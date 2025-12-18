import json
import logging
import apache_beam as beam
from apache_beam.transforms.userstate import BagStateSpec
from apache_beam.coders import PickleCoder
from datetime import datetime

class ParseVehicleUpdates(beam.DoFn):
    """
    Parses raw JSON messages from the subway feed and extracts relevant vehicle updates.
    Filters for specific route and stop ID.
    """
    def __init__(self, target_route_id="E", target_stop_id="F11S"):
        self.target_route_id = target_route_id
        self.target_stop_id = target_stop_id

    def process(self, element):
        """
        element: Raw JSON string or bytes.
        """
        try:
            if isinstance(element, bytes):
                element = element.decode('utf-8')
            
            data = json.loads(element)
            
            # Check if 'entity' list exists
            if 'entity' not in data:
                return

            for entity in data['entity']:
                # We only care about 'vehicle' updates
                if 'vehicle' not in entity:
                    continue
                
                vehicle = entity['vehicle']
                trip = vehicle.get('trip', {})
                
                route_id = trip.get('route_id')
                stop_id = vehicle.get('stop_id')
                
                # Filter for target route and stop
                if route_id == self.target_route_id and stop_id == self.target_stop_id:
                    timestamp = vehicle.get('timestamp')
                    
                    # If timestamp is missing in vehicle, try header (though vehicle usually has it)
                    if not timestamp:
                        timestamp = data.get('header', {}).get('timestamp')

                    if timestamp:
                        # Key by route_id and stop_id to allow stateful processing per station
                        key = f"{route_id}_{stop_id}"
                        yield (key, {
                            'route_id': route_id,
                            'stop_id': stop_id,
                            'timestamp': int(timestamp),
                            'trip_id': trip.get('trip_id'),
                            'status': vehicle.get('current_status')
                        })

        except json.JSONDecodeError:
            logging.error("Failed to decode JSON message")
        except Exception as e:
            logging.error(f"Error processing message: {e}")

class AccumulateArrivals(beam.DoFn):
    """
    Stateful DoFn that accumulates the last 150 arrival timestamps.
    """
    # State to hold the list of arrival timestamps
    ARRIVAL_HISTORY = BagStateSpec('arrival_history', PickleCoder())

    def process(self, element, arrival_history=beam.DoFn.StateParam(ARRIVAL_HISTORY)):
        """
        element: (key, update_dict)
        """
        key, update = element
        new_timestamp = update['timestamp']
        
        # Read current history
        # Note: BagState is unordered and append-only. 
        # For a sliding window, we might need to read all, sort, and prune.
        # Since we need strict order for time series, we read all into memory.
        current_history = list(arrival_history.read())
        
        # Add new timestamp if it's not already there (deduplication)
        if new_timestamp not in current_history:
            current_history.append(new_timestamp)
            current_history.sort()
            
            # Keep only the last 150
            if len(current_history) > 150:
                current_history = current_history[-150:]
            
            # Clear and rewrite state
            arrival_history.clear()
            for ts in current_history:
                arrival_history.add(ts)
            
            # If we have enough data, emit the window for prediction
            if len(current_history) == 150:
                yield {
                    'key': key,
                    'timestamps': current_history,
                    'last_timestamp': new_timestamp
                }

                            'stop_id': stop_id,
                            'timestamp': int(timestamp),
                            'trip_id': trip.get('trip_id'),
                            'status': vehicle.get('current_status')
                        }

        except json.JSONDecodeError:
            logging.error("Failed to decode JSON message")
        except Exception as e:
            logging.error(f"Error processing message: {e}")
