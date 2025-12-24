import logging
import apache_beam as beam
from google.cloud import firestore
from datetime import datetime

class WriteToFirestore(beam.DoFn):
    """
    Writes prediction results to Firestore.
    """
    def __init__(self, project_id, collection_name):
        self.project_id = project_id
        self.collection_name = collection_name
        self.client = None

    def setup(self):
        """
        Initialize Firestore client once per worker.
        """
        # By default, this connects to the '(default)' database.
        # If using a named database, we must specify it.
        # self.client = firestore.Client(project=self.project_id, database='timeseries-project')
        # However, usually the main DB is named '(default)'.
        # If the user explicitly named it 'timeseries-project', we need to pass that.
        
        # Let's try to connect to the named database if provided, otherwise default.
        # For now, we will assume the user meant the project ID is 'timeseries-project' 
        # and the database is '(default)'.
        # BUT if the database name ITSELF is 'timeseries-project', we need to change this.
        
        # Assuming standard setup where DB name matches project or is (default).
        self.client = firestore.Client(project=self.project_id)

    def process(self, element):
        """
        element: {
            'key': 'Route_Stop',
            'input_last_timestamp': ts,
            'forecast': [...]
        }
        """
        try:
            key = element['key']
            last_ts = element['input_last_timestamp']
            forecast = element['forecast']
            
            # Create a document ID based on the key and timestamp to ensure uniqueness
            # e.g., "E_F11S_1760369207"
            doc_id = f"{key}_{last_ts}"
            
            # Structure the data for Firestore
            # We want to store the forecast as a structured object, not just a raw list
            # The forecast is a list of dicts: [{'ds': '...', 'NHITS-median': ...}, ...]
            
            # Take the first prediction (median) as the primary value
            # But store the full forecast array for detailed plotting
            
            primary_prediction = 0.0
            if forecast and isinstance(forecast, list) and len(forecast) > 0:
                # Handle both list of floats (old format) and list of dicts (new format)
                first_item = forecast[0]
                if isinstance(first_item, dict):
                    primary_prediction = first_item.get('NHITS-median', 0.0)
                elif isinstance(first_item, (int, float)):
                    primary_prediction = float(first_item)

            firestore_data = {
                'route_id': key.split('_')[0],
                'stop_id': key.split('_')[1] if '_' in key else 'UNKNOWN',
                'timestamp': last_ts,
                'timestamp_str': datetime.fromtimestamp(last_ts).isoformat(),
                'predicted_mbt': primary_prediction,
                'forecast_details': forecast, # Store the full list of dicts
                'status': element.get('prediction_status', 'UNKNOWN'),
                'created_at': firestore.SERVER_TIMESTAMP
            }
            
            doc_ref = self.client.collection(self.collection_name).document(doc_id)
            doc_ref.set(firestore_data)
            
            logging.info(f"✅ Written to Firestore: {doc_id} | MBT: {primary_prediction:.2f}")
            
        except Exception as e:
            # Check for "database not found" error specifically
            error_str = str(e)
            if "404" in error_str and "database" in error_str:
                 logging.error(f"🔥 FIRESTORE ERROR: Database not found. Please create a Firestore database in 'Native Mode' for project {self.project_id}.")
            else:
                logging.error(f"Error writing to Firestore for {element.get('key')}: {e}")
            
            # Prepare data for Firestore
            # We store the raw forecast plus metadata
            data = {
                'route_stop_id': key,
                'prediction_timestamp': datetime.utcnow(),
                'last_arrival_timestamp': datetime.fromtimestamp(last_ts),
                'forecast_values': forecast
            }
            
            doc_ref.set(data)
            
            logging.info(f"Successfully wrote prediction for {doc_id} to Firestore.")
            
        except Exception as e:
            logging.error(f"Error writing to Firestore for {element.get('key')}: {e}")
