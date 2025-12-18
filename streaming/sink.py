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
            
            doc_ref = self.client.collection(self.collection_name).document(doc_id)
            
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
