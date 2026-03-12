import time
import msgpack
import redis
import fastf1
import math
import argparse
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description="S.C.I.T.T.S.E. Live Telemetry Simulator")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--track", type=str, default="Silverstone")
    parser.add_argument("--session", type=str, default="R")
    parser.add_argument("--driver", type=str, default="4")
    return parser.parse_args()

try:
    r = redis.Redis(host='localhost', port=6379, db=0)
    r.ping()
    print("🟢 S.C.I.T.T.S.E. Data Bus Connected.")
except redis.ConnectionError:
    print("🔴 Redis Connection Failed.")
    exit(1)

def safe_cast(val, cast_type, default):
    try:
        if math.isnan(val): return default
        return cast_type(val)
    except:
        return default

def get_current_turn(car_distance, corners):
    if corners is None or car_distance is None or math.isnan(car_distance):
        return "Straight"
    for _, corner in corners.iterrows():
        if abs(corner['Distance'] - car_distance) < 100:
            return str(corner['Number'])
    return "Straight"

def stream_telemetry(year, track, session_type, driver):
    print(f"📡 Initializing Link: {year} {track} [{session_type}] - Driver: {driver}")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cache_dir = os.path.join(project_root, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    
    try:
        session = fastf1.get_session(year, track, session_type)
        session.load(telemetry=True, weather=False, messages=False)
        
        try:
            circuit_info = session.get_circuit_info()
            corners = circuit_info.corners
        except Exception:
            corners = None

        print(f"🏎️ Locking onto Car {driver}...")
        driver_laps = session.laps.pick_drivers(str(driver))
        
        # --- Dynamic Track Scaling Math ---
        print("📐 Calculating dynamic track boundaries...")
        full_telemetry = driver_laps.get_telemetry()
        min_x, max_x = full_telemetry['X'].min(), full_telemetry['X'].max()
        min_y, max_y = full_telemetry['Y'].min(), full_telemetry['Y'].max()
        
        track_width = max_x - min_x
        track_height = max_y - min_y
        max_dim = max(track_width, track_height)
        
        offset_x = (max_dim - track_width) / 2
        offset_y = (max_dim - track_height) / 2
        # ----------------------------------

        # --- Virtual Brake Sensor Calibration ---
        # Runs once on full session data to find peak mechanical braking G.
        # This sets the "100%" mark so real-time normalization is stable.
        print("🧠 Calibrating Virtual Brake Sensor...")
        _speeds_ms = full_telemetry['Speed'] / 3.6
        _times_s   = full_telemetry['Time'].dt.total_seconds()
        _accels    = _speeds_ms.diff() / _times_s.diff()
        _gs        = _accels / 9.81

        # Drag model: ~1G at 300 km/h (83.33 m/s), scales with v²
        _drags     = (_speeds_ms / 83.33) ** 2

        # Clamp to 0 — prevents drag overcorrection on slow corners
        # producing negative mech_g that would corrupt max_mech_g
        _mech_gs   = (_gs.abs() - _drags).clip(lower=0)

        # Gate: only sample moments where brake pedal is physically pressed
        _valid_brakes = _mech_gs[full_telemetry['Brake'] > 0]

        # Fallback: 3.0G is realistic post-drag peak (not 4.0 which compressed the scale)
        max_mech_g = (
            _valid_brakes.max()
            if not _valid_brakes.empty and not math.isnan(_valid_brakes.max())
            else 3.0
        )
        print(f"🛑 Max Mechanical Braking Force Calibrated at: {max_mech_g:.2f} G")
        # ----------------------------------------
        
    except Exception as e:
        print(f"\n🔴 ERROR: Failed to load data.\nDetails: {e}")
        exit(1)

    print(f"🚀 Blasting MsgPack stream to Redis (synced to real telemetry timestamps)...")
    
    for _, lap in driver_laps.iterrows():
        lap_num = safe_cast(lap['LapNumber'], int, 0)
        telemetry = lap.get_telemetry()
        
        prev_time     = None
        prev_speed_ms = None
        brake_history = []  # Reset per lap — smoothing state must not bleed between laps

        for index, row in telemetry.iterrows():

            current_time      = row['Time'].total_seconds() if hasattr(row['Time'], 'total_seconds') else 0
            current_speed_kmh = safe_cast(row['Speed'], float, 0.0)
            current_speed_ms  = current_speed_kmh / 3.6
            long_g            = 0.0  # First row of each lap always 0 — correct, car hasn't moved yet
            time_delta        = 0.0

            if prev_time is not None and prev_speed_ms is not None:
                time_delta = current_time - prev_time
                if time_delta > 0:
                    accel_ms2 = (current_speed_ms - prev_speed_ms) / time_delta
                    long_g    = accel_ms2 / 9.81

            prev_time     = current_time
            prev_speed_ms = current_speed_ms

            # === VIRTUAL BRAKE SENSOR PIPELINE ===

            brake_binary = safe_cast(row['Brake'], int, 0)

            # 1. Gate — require both pedal signal AND physical deceleration
            #    This blocks engine braking (long_g < 0 but brake_binary = 0)
            #    and false positives on the first frame (long_g = 0.0)
            if brake_binary > 0 and long_g < 0:

                # 2. Subtract drag — clamp to 0 so slow-corner braking
                #    (where drag ≈ total decel) never produces negative pressure
                drag_g = (current_speed_ms / 83.33) ** 2
                mech_g = max(0.0, abs(long_g) - drag_g)

                # 3. Normalize against session peak and clamp to 0–100
                raw_brake_pct = (mech_g / max_mech_g) * 100
                raw_brake_pct = max(0.0, min(100.0, raw_brake_pct))

            else:
                raw_brake_pct = 0.0

            # 4. Smooth — window of 9 gives a fluid trace without
            #    losing the sharpness of a hard late-braking event
            brake_history.append(raw_brake_pct)
            if len(brake_history) > 9:
                brake_history.pop(0)
            smooth_brake_pct = sum(brake_history) / len(brake_history) if brake_history else 0.0

            # ======================================

            car_distance = safe_cast(row['Distance'], float, 0.0)
            current_turn = get_current_turn(car_distance, corners)
            
            raw_x  = safe_cast(row['X'], float, 0.0)
            raw_y  = safe_cast(row['Y'], float, 0.0)
            norm_x = (raw_x - min_x + offset_x) / max_dim if max_dim > 0 else 0
            norm_y = (raw_y - min_y + offset_y) / max_dim if max_dim > 0 else 0
            
            payload = {
                "rpm":      safe_cast(row['RPM'], int, 0),
                "speed":    int(current_speed_kmh),   # Use already-computed value, not a second safe_cast
                "gear":     safe_cast(row['nGear'], int, 0),
                "throttle": safe_cast(row['Throttle'], int, 0),
                "brake":    int(smooth_brake_pct),    # Virtual pressure, not binary*100
                "g_force":  round(long_g, 2),
                "x":        norm_x,
                "y":        norm_y,
                "lap":      lap_num,
                "turn":     current_turn
            }
            
            packed_data = msgpack.packb(payload)
            r.publish('race:telemetry', packed_data)
            
            # Sync stream to real timestamp cadence
            if time_delta > 0:
                time.sleep(max(0.001, time_delta))

if __name__ == "__main__":
    args = parse_arguments()
    stream_telemetry(args.year, args.track, args.session, args.driver)