import time
import os
import json
import logging
import requests
from google.cloud import pubsub_v1
from google.protobuf.json_format import MessageToDict
from google.transit import gtfs_realtime_pb2

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configuration (Env Variables)
PROJECT_ID = os.getenv('PROJECT_ID', 'time-series-478616')
TOPIC_ID = os.getenv('TOPIC_ID', 'vehicle-position-updates')
MTA_API_KEY = os.getenv('MTA_API_KEY')
# Default to ACE train feed, but can be overridden
# See https://api.mta.info/#/subwayRealTimeFeeds
FEED_URL = os.getenv('FEED_URL', 'https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-ace') 
POLL_INTERVAL = int(os.getenv('POLL_INTERVAL', '30')) # Seconds

def get_pubsub_publisher():
    return pubsub_v1.PublisherClient()

def get_topic_path(publisher):
    return publisher.topic_path(PROJECT_ID, TOPIC_ID)

def fetch_feed():
    headers = {'x-api-key': MTA_API_KEY} if MTA_API_KEY else {}
    try:
        response = requests.get(FEED_URL, headers=headers, timeout=10)
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        logging.error(f"Error fetching feed: {e}")
        return None

def process_feed(content, publisher, topic_path):
    try:
        # Parse Protobuf
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(content)
        
        # Convert to JSON (Dict)
        # preserving_proto_field_name=True keeps 'route_id' instead of 'routeId'
        # Check your downstream parser expectations! 
        # streaming/transform.py uses camelCase (e.g. routeId) in some places? 
        # Let's check the sample data or transform.py.
        # transform.py uses: entity['vehicle']['trip']['routeId']
        # So we likely want camelCase (default for MessageToDict is camelCase)
        feed_dict = MessageToDict(feed)
        
        # Serialize to JSON string
        message_json = json.dumps(feed_dict)
        message_bytes = message_json.encode('utf-8')
        
        # Publish to Pub/Sub
        future = publisher.publish(topic_path, message_bytes)
        message_id = future.result()
        
        logging.info(f"Published message ID: {message_id} with {len(feed.entity)} entities.")
        
    except Exception as e:
        logging.error(f"Error processing/publishing feed: {e}")

def main():
    # if not MTA_API_KEY:
    #     logging.warning("MTA_API_KEY is not set. Requests might fail if auth is required.")

    publisher = get_pubsub_publisher()
    topic_path = get_topic_path(publisher)
    
    logging.info(f"Starting ingestion loop. Feed: {FEED_URL}")
    logging.info(f"Publishing to: {topic_path}")
    logging.info(f"Poll Interval: {POLL_INTERVAL}s")

    while True:
        start_time = time.time()
        
        content = fetch_feed()
        if content:
            process_feed(content, publisher, topic_path)
        
        # Sleep for the remainder of the interval
        elapsed = time.time() - start_time
        sleep_time = max(0, POLL_INTERVAL - elapsed)
        time.sleep(sleep_time)

if __name__ == "__main__":
    main()
