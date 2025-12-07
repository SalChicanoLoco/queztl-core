#!/bin/bash

# Gen3D Quick Start Script
# Launches the standalone Gen3D application

set -e

echo "🚀 Starting Gen3D - AI 3D Model Generation"
echo "=========================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running"
    echo "Please start Docker Desktop and try again"
    exit 1
fi

# Navigate to gen3d-app directory
cd "$(dirname "$0")"

echo ""
echo "📦 Building containers..."
docker-compose build

echo ""
echo "🔄 Starting services..."
docker-compose up -d

echo ""
echo "⏳ Waiting for services to be ready..."
sleep 5

# Check backend health
echo "🏥 Checking backend health..."
for i in {1..10}; do
    if curl -s http://localhost:8001/health > /dev/null 2>&1; then
        echo "✅ Backend is ready!"
        break
    fi
    if [ $i -eq 10 ]; then
        echo "❌ Backend failed to start"
        docker-compose logs backend
        exit 1
    fi
    echo "   Attempt $i/10..."
    sleep 2
done

echo ""
echo "✨ Gen3D is ready!"
echo ""
echo "🌐 Access the application:"
echo "   Frontend:  http://localhost:3001"
echo "   Backend:   http://localhost:8001"
echo "   API Docs:  http://localhost:8001/docs"
echo ""
echo "📊 Useful commands:"
echo "   View logs:     docker-compose logs -f"
echo "   Stop services: docker-compose down"
echo "   Restart:       docker-compose restart"
echo ""
echo "🎨 Try these example prompts:"
echo "   • Futuristic spacecraft"
echo "   • Medieval castle tower"
echo "   • Cyberpunk character"
echo "   • Ancient tree"
echo ""
