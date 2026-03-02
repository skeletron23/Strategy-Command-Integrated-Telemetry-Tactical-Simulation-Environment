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
    """Calculates if the car's distance matches a known corner apex."""
    if corners is None or car_distance is None or math.isnan(car_distance):
        return "Straight"
    
    # Check if the car is within 100 meters of any corner
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
        
        # Load the Track Blueprint to map the corners
        try:
            circuit_info = session.get_circuit_info()
            corners = circuit_info.corners
            print("🗺️ Track blueprint loaded for corner detection.")
        except Exception:
            corners = None
            print("⚠️ Could not load track blueprint. Turn numbers disabled.")

        print(f"🏎️ Locking onto Car {driver}...")
        driver_laps = session.laps.pick_drivers(str(driver))
        
    except Exception as e:
        print(f"\n🔴 ERROR: Failed to load data.\nDetails: {e}")
        exit(1)

    print(f"🚀 Blasting 60Hz MsgPack stream to Redis...")
    
    # Iterate lap-by-lap so we know exactly which lap we are on
    for _, lap in driver_laps.iterrows():
        lap_num = safe_cast(lap['LapNumber'], int, 0)
        telemetry = lap.get_telemetry()
        
        for index, row in telemetry.iterrows():
            start_time = time.time()
            
            # Calculate current distance and turn
            car_distance = safe_cast(row['Distance'], float, 0.0)
            current_turn = get_current_turn(car_distance, corners)
            
            # The upgraded payload
            payload = {
                "rpm": safe_cast(row['RPM'], int, 0),
                "speed": safe_cast(row['Speed'], int, 0),
                "gear": safe_cast(row['nGear'], int, 0),
                "throttle": safe_cast(row['Throttle'], int, 0),
                "brake": safe_cast(row['Brake'], int, 0),
                "x": safe_cast(row['X'], float, 0.0),
                "y": safe_cast(row['Y'], float, 0.0),
                "lap": lap_num,          # <--- NEW
                "turn": current_turn     # <--- NEW
            }
            
            packed_data = msgpack.packb(payload)
            r.publish('race:telemetry', packed_data)
            
            elapsed = time.time() - start_time
            time.sleep(max(0, (1/60) - elapsed))

if __name__ == "__main__":
    args = parse_arguments()
    stream_telemetry(args.year, args.track, args.session, args.driver)