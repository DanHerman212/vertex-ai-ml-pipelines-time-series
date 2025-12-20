#!/bin/bash
set -e

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '#' | awk '/=/ {print $1}')
else
    echo "Error: .env file not found."
    exit 1
fi

# Required Variables
REQUIRED_VARS=("PROJECT_ID" "ZONE" "INSTANCE_NAME" "MTA_API_KEY")
for VAR in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!VAR}" ]; then
        echo "Error: $VAR is not set in .env"
        exit 1
    fi
done

echo "Creating/Verifying GCE instance..."
if gcloud compute instances describe $INSTANCE_NAME --project=$PROJECT_ID --zone=$ZONE > /dev/null 2>&1; then
    echo "Instance $INSTANCE_NAME already exists."
else
    gcloud compute instances create $INSTANCE_NAME \
        --project=$PROJECT_ID \
        --zone=$ZONE \
        --machine-type=e2-standard-2 \
        --scopes=https://www.googleapis.com/auth/cloud-platform \
        --image-family=debian-11 \
        --image-project=debian-cloud
fi

echo "Waiting for instance..."
sleep 10

echo "Copying workspace..."
tar -czf workspace.tar.gz --exclude='venv' --exclude='.git' --exclude='__pycache__' .
gcloud compute scp workspace.tar.gz $INSTANCE_NAME:~/ --project=$PROJECT_ID --zone=$ZONE
rm workspace.tar.gz

echo "Running Ingestion Test on VM..."
gcloud compute ssh $INSTANCE_NAME --project=$PROJECT_ID --zone=$ZONE --command="
    sudo apt-get update -q
    sudo apt-get install -yq python3-pip python3-venv
    
    mkdir -p ingestion_test
    tar -xzf workspace.tar.gz -C ingestion_test
    cd ingestion_test
    
    python3 -m venv venv
    source venv/bin/activate
    
    pip install -r ingestion/requirements.txt
    
    # Load env vars from the copied .env file
    if [ -f .env ]; then
        export \$(cat .env | grep -v '#' | awk '/=/ {print \$1}')
    fi
    
    echo '--- Starting Dry Run ---'
    python3 ingestion/ingest_feed.py --dry-run
    echo '--- Dry Run Complete ---'
"
