#!/bin/bash
set -e

# Configuration
PROJECT_ID=${PROJECT_ID:-"time-series-478616"}
REGION=${REGION:-"us-east1"}
TOPIC_ID="vehicle-position-updates"
SUBSCRIPTION_ID="vehicle-position-updates-sub"

echo "========================================================"
echo "Setting up Infrastructure for Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Topic: $TOPIC_ID"
echo "Subscription: $SUBSCRIPTION_ID"
echo "========================================================"

# 1. Enable Required APIs
echo ""
echo "[1/3] Enabling Google Cloud APIs..."
gcloud services enable \
    pubsub.googleapis.com \
    dataflow.googleapis.com \
    firestore.googleapis.com \
    aiplatform.googleapis.com \
    --project "$PROJECT_ID"

# 2. Create Pub/Sub Topic
echo ""
echo "[2/3] Checking Pub/Sub Topic..."
if ! gcloud pubsub topics describe "$TOPIC_ID" --project "$PROJECT_ID" &>/dev/null; then
    echo "Creating topic: $TOPIC_ID"
    gcloud pubsub topics create "$TOPIC_ID" --project "$PROJECT_ID"
else
    echo "Topic $TOPIC_ID already exists."
fi

# 3. Create Pub/Sub Subscription
# We create a Pull subscription for Dataflow
echo ""
echo "[3/3] Checking Pub/Sub Subscription..."
if ! gcloud pubsub subscriptions describe "$SUBSCRIPTION_ID" --project "$PROJECT_ID" &>/dev/null; then
    echo "Creating subscription: $SUBSCRIPTION_ID"
    gcloud pubsub subscriptions create "$SUBSCRIPTION_ID" \
        --topic "$TOPIC_ID" \
        --project "$PROJECT_ID" \
        --ack-deadline 60
else
    echo "Subscription $SUBSCRIPTION_ID already exists."
fi

echo ""
echo "Infrastructure setup complete."
