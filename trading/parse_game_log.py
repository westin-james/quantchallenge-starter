import sys
import json
import argparse

BASE_KEYS = [
    "home_away","home_score","away_score","event_type","player_name",
    "substituted_player_name","shot_type","assist_player","rebound_type",
    "coordinate_x","coordinate_y","time_seconds"
]

def parse_stream_to_events(io):
    events = []
    for raw in io:
        line = raw.strip()
        if not line:
            continue

        if line.startswith("algo_print:"):
            line = line[len("algo_print:"):].strip()
        
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        if isinstance(obj, dict):
            events.append(obj)
        elif isinstance(obj, list):
            events.extend([x for x in obj if isinstance(x, dict)])
        else:
            continue
    return events

def main():
    ap = argparse.ArgumentParser(description="Convert strategy log lines into JSON array")
    ap.add_argument("-i","--input", type=str, default=None, help="Path to log file (default: stdin)")
    ap.add_argument("-o","--output", type=str, default="game_events.json", help="Path to write JSON file")
    ap.add_argument("--only-base-keys", action="store_true",
                    help="If set, drop extra fields and keep only base event keys")
    args = ap.parse_args()

    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            events = parse_stream_to_events(f)
    else:
        events = parse_stream_to_events(sys.stdin)
    
    if args.only_base_keys:
        cleaned = []
        for ev in events:
            cleaned.append({k: ev.get(k, None) for k in BASE_KEYS})
        events = cleaned
    
    # Write as array
    with open(args.output, "w", encoding="utf-8") as out:
        json.dump(events, out, ensure_ascii=False, indet=2)
    
    print(f"Wrote {len(events)} events to {args.output}")

if __name__ == "__main__":
    main()