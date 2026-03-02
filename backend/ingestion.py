import time
import msgpack
import redis
import fastf1
import math
import argparse

# 1. Setup Command Line Arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="S.C.I.T.T.S.E. Live Telemetry Simulator")
    parser.add_argument("--year", type=int, default=2023, help="Championship Year (e.g., 2023, 2024)")
    parser.add_argument("--track", type=str, default="Monaco", help="Track Name (e.g., Monaco, Monza, Silverstone)")
    parser.add_argument("--session", type=str, default="Q", help="Session Type (FP1, FP2, FP3, Q, SQ, S, R)")
    parser.add_argument("--driver", type=str, default="16", help="Driver Number or 3-Letter Code (e.g., 16, LEC, 4, NOR)")
    return parser.parse_args()

# 2. Connect to Data Bus
try:
    r = redis.Redis(host='localhost', port=6379, db=0)
    r.ping()
    print("🟢 S.C.I.T.T.S.E. Data Bus Connected.")
except redis.ConnectionError:
    print("🔴 Redis Connection Failed. Is the Docker container running?")
    exit(1)

def safe_cast(val, cast_type, default):
    """Prevents NaN values from crashing the binary packer."""
    try:
        if math.isnan(val): return default
        return cast_type(val)
    except:
        return default

def stream_telemetry(year, track, session_type, driver):
    print(f"📡 Initializing Link: {year} {track} [{session_type}] - Driver: {driver}")
    
    fastf1.Cache.enable_cache('./')
    
    try:
        session = fastf1.get_session(year, track, session_type)
        session.load(telemetry=True, weather=False, messages=False)
        
        print(f"🏎️ Locking onto Car {driver} and syncing GPS + Engine telemetry...")
        
        # This is the "Smart Merge" fix
        driver_laps = session.laps.pick_drivers(str(driver))
        telemetry = driver_laps.get_telemetry()
        
    except Exception as e:
        print(f"\n🔴 ERROR: Failed to load data.\nDetails: {e}")
        exit(1)

    print(f"🚀 Blasting 60Hz MsgPack stream to Redis channel: 'race:telemetry'...")
    
    for index, row in telemetry.iterrows():
        start_time = time.time()
        
        # The payload now has access to 'X' and 'Y' from the smart merge
        payload = {
            "rpm": safe_cast(row['RPM'], int, 0),
            "speed": safe_cast(row['Speed'], int, 0),
            "gear": safe_cast(row['nGear'], int, 0),
            "throttle": safe_cast(row['Throttle'], int, 0),
            "brake": safe_cast(row['Brake'], int, 0),
            "x": safe_cast(row['X'], float, 0.0),
            "y": safe_cast(row['Y'], float, 0.0)
        }
        
        packed_data = msgpack.packb(payload)
        r.publish('race:telemetry', packed_data)
        
        # Enforce the 60Hz frequency
        elapsed = time.time() - start_time
        time.sleep(max(0, (1/60) - elapsed))

if __name__ == "__main__":
    args = parse_arguments()
    stream_telemetry(args.year, args.track, args.session, args.driver)