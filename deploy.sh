#!/bin/bash
# Deploy script for Megan Spanish Bot

set -e

echo "🚀 Deploying Megan Spanish Bot..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo "Please copy .env.example to .env and fill in your credentials."
    exit 1
fi

# Create data directory
mkdir -p data

# Pull latest image
echo "📥 Pulling latest image..."
docker-compose pull

# Start services
echo "▶️ Starting services..."
docker-compose up -d

# Wait for health check
echo "⏳ Waiting for health check..."
sleep 5

# Check if running
if docker-compose ps | grep -q "Up"; then
    echo "✅ Bot is running!"
    docker-compose logs --tail=20
else
    echo "❌ Something went wrong. Check logs:"
    docker-compose logs
    exit 1
fi

echo ""
echo "🎉 Deploy complete! The bot is now running."
echo "📊 View logs: docker-compose logs -f"
echo "🛑 Stop: docker-compose down"
