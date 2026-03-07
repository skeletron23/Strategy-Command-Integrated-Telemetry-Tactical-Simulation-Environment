import time
import msgpack
import redis
import fastf1
import math
import argparse

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
    fastf1.Cache.enable_cache('./')
    
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
        
        # --- NEW: Dynamic Track Scaling Math ---
        print("📐 Calculating dynamic track boundaries...")
        full_telemetry = driver_laps.get_telemetry()
        min_x, max_x = full_telemetry['X'].min(), full_telemetry['X'].max()
        min_y, max_y = full_telemetry['Y'].min(), full_telemetry['Y'].max()
        
        # Find the largest dimension to maintain aspect ratio without stretching
        track_width = max_x - min_x
        track_height = max_y - min_y
        max_dim = max(track_width, track_height)
        
        # Offsets to perfectly center the track in the square
        offset_x = (max_dim - track_width) / 2
        offset_y = (max_dim - track_height) / 2
        # ---------------------------------------
        
    except Exception as e:
        print(f"\n🔴 ERROR: Failed to load data.\nDetails: {e}")
        exit(1)

    print(f"🚀 Blasting MsgPack stream to Redis (synced to real telemetry timestamps)...")
    
    for _, lap in driver_laps.iterrows():
        lap_num = safe_cast(lap['LapNumber'], int, 0)
        telemetry = lap.get_telemetry()
        
        prev_time = None
        for index, row in telemetry.iterrows():
            start_time = time.time()
            
            # --- NEW: Use actual telemetry timestamp delta ---
            current_time = row['Time'].total_seconds() if hasattr(row['Time'], 'total_seconds') else 0
            
            if prev_time is not None:
                time_delta = current_time - prev_time
                # Sleep for the actual time delta between telemetry points
                sleep_time = max(0.001, time_delta)  # Minimum 1ms to avoid busy-waiting
                time.sleep(sleep_time)
            
            prev_time = current_time
            # -----------------------------------------------
            
            car_distance = safe_cast(row['Distance'], float, 0.0)
            current_turn = get_current_turn(car_distance, corners)
            
            raw_x = safe_cast(row['X'], float, 0.0)
            raw_y = safe_cast(row['Y'], float, 0.0)
            
            # --- NEW: Normalize X and Y to a 0.0 - 1.0 percentage ---
            norm_x = (raw_x - min_x + offset_x) / max_dim if max_dim > 0 else 0
            norm_y = (raw_y - min_y + offset_y) / max_dim if max_dim > 0 else 0
            
            payload = {
                "rpm": safe_cast(row['RPM'], int, 0),
                "speed": safe_cast(row['Speed'], int, 0),
                "gear": safe_cast(row['nGear'], int, 0),
                "throttle": safe_cast(row['Throttle'], int, 0),
                "brake": safe_cast(row['Brake'], int, 0),
                "x": norm_x,             # Pushing a percentage instead of meters
                "y": norm_y,             # Pushing a percentage instead of meters
                "lap": lap_num,
                "turn": current_turn
            }
            
            packed_data = msgpack.packb(payload)
            r.publish('race:telemetry', packed_data)

if __name__ == "__main__":
    args = parse_arguments()
    stream_telemetry(args.year, args.track, args.session, args.driver)