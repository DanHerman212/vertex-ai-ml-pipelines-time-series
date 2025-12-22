#!/bin/bash
set -e

IMAGE_NAME="nhits-serving-local"
CONTAINER_NAME="nhits-serving-test"
PORT=8080

echo "========================================================"
echo "TEST: Building Serving Image Locally"
echo "========================================================"

# Build the image using the serving Dockerfile
# We use the same Dockerfile as production to ensure fidelity
docker build -t $IMAGE_NAME -f docker/Dockerfile.serving .

echo "========================================================"
echo "TEST: Starting Container"
echo "========================================================"

# Stop any existing container
docker rm -f $CONTAINER_NAME 2>/dev/null || true

# Run the container
# We mount the local model artifacts to /app/nhits_model because serve.py defaults to looking there
# if AIP_STORAGE_URI is not set.
docker run -d \
    --name $CONTAINER_NAME \
    -p $PORT:8080 \
    -v "$(pwd)/local_test_artifacts/nhits_model:/app/nhits_model" \
    $IMAGE_NAME

echo "Waiting for container to start (10s)..."
sleep 10

# Check if container is still running
if ! docker ps | grep -q $CONTAINER_NAME; then
    echo "Error: Container died immediately. Logs:"
    docker logs $CONTAINER_NAME
    exit 1
fi

echo "========================================================"
echo "TEST: Health Check"
echo "========================================================"
curl -v http://localhost:$PORT/health

echo ""
echo "========================================================"
echo "TEST: Prediction"
echo "========================================================"

# Create a dummy payload
# NeuralForecast expects a list of dicts or a dict with "instances" key
# We need to provide enough history for the model to predict? 
# Actually, NHITS usually needs the input window.
# Let's try sending a small payload. If it fails with shape error, at least we know the server is up.

cat <<EOF > payload.json
{
  "instances": [
    {"ds": "2025-01-01 10:00:00", "y": 10.0, "unique_id": "L"},
    {"ds": "2025-01-01 10:15:00", "y": 12.0, "unique_id": "L"},
    {"ds": "2025-01-01 10:30:00", "y": 15.0, "unique_id": "L"}
  ]
}
EOF

curl -X POST \
    -H "Content-Type: application/json" \
    -d @payload.json \
    http://localhost:$PORT/predict

echo ""
echo "========================================================"
echo "Logs:"
docker logs $CONTAINER_NAME | head -n 20

echo "========================================================"
echo "Cleaning up..."
docker rm -f $CONTAINER_NAME
rm payload.json
