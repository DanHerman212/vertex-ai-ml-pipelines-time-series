import glob
import json
from streaming.transform import ParseVehicleUpdates

def test_parsing():
    files = glob.glob("streaming/json_files/*.json")
    print(f"Found {len(files)} files.")
    
    # Test with a stop we know exists in the sample file (A15N on route A)
    print("Testing with Route A, Stop A15N...")
    parser = ParseVehicleUpdates(target_route_id="A", target_stop_id="A15N")
    
    for file_path in files:
        print(f"Processing {file_path}...")
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Simulate the DoFn process
        results = list(parser.process(content))
        
        if results:
            print(f"  Found {len(results)} matching updates:")
            for res in results:
                print(f"    - {res}")
        else:
            print("  No matching updates found.")

    # Also test the user's target (E / F11S) to confirm it returns nothing (as expected for this data)
    print("\nTesting with Route E, Stop F11S...")
    parser_target = ParseVehicleUpdates(target_route_id="E", target_stop_id="F11S")
    for file_path in files:
         with open(file_path, 'r') as f:
            content = f.read()
            results = list(parser_target.process(content))
            if results:
                print(f"  Found {len(results)} matching updates in {file_path}")

if __name__ == "__main__":
    test_parsing()
