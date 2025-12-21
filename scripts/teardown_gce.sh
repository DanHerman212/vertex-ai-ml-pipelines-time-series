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
REQUIRED_VARS=("PROJECT_ID" "ZONE" "INSTANCE_NAME" "TOPIC_ID" "SUBSCRIPTION_ID")
for VAR in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!VAR}" ]; then
        echo "Error: $VAR is not set in .env"
        exit 1
    fi
done

echo "========================================================"
echo "TEARDOWN: Deleting GCE and Pub/Sub Resources"
echo "Project: $PROJECT_ID"
echo "Instance: $INSTANCE_NAME ($ZONE)"
echo "Topic: $TOPIC_ID"
echo "Subscription: $SUBSCRIPTION_ID"
echo "========================================================"

read -p "Are you sure you want to delete these resources? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Teardown cancelled."
    exit 1
fi

# 1. Delete GCE Instance
echo "Deleting GCE instance $INSTANCE_NAME..."
if gcloud compute instances describe $INSTANCE_NAME --project=$PROJECT_ID --zone=$ZONE > /dev/null 2>&1; then
    gcloud compute instances delete $INSTANCE_NAME --project=$PROJECT_ID --zone=$ZONE --quiet
    echo "Instance deleted."
else
    echo "Instance $INSTANCE_NAME not found, skipping."
fi

# 2. Delete Pub/Sub Subscription
echo "Deleting Pub/Sub subscription $SUBSCRIPTION_ID..."
if gcloud pubsub subscriptions describe $SUBSCRIPTION_ID --project=$PROJECT_ID > /dev/null 2>&1; then
    gcloud pubsub subscriptions delete $SUBSCRIPTION_ID --project=$PROJECT_ID --quiet
    echo "Subscription deleted."
else
    echo "Subscription $SUBSCRIPTION_ID not found, skipping."
fi

# 3. Delete Pub/Sub Topic
echo "Deleting Pub/Sub topic $TOPIC_ID..."
if gcloud pubsub topics describe $TOPIC_ID --project=$PROJECT_ID > /dev/null 2>&1; then
    gcloud pubsub topics delete $TOPIC_ID --project=$PROJECT_ID --quiet
    echo "Topic deleted."
else
    echo "Topic $TOPIC_ID not found, skipping."
fi

echo "========================================================"
echo "Teardown complete."
echo "========================================================"
