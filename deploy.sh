#!/bin/bash
# Deploy script for Megan Spanish Bot

set -e

echo "🚀 Deploying Megan Spanish Bot..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ .env file not found!"
    echo "Please copy .env.example to .env and configure your tokens:"
    echo "  cp .env.example .env"
    exit 1
fi

# Check required env vars
if ! grep -q "TELEGRAM_TOKEN=your_" .env && grep -q "TELEGRAM_TOKEN=" .env; then
    echo "✅ TELEGRAM_TOKEN found"
else
    echo "❌ TELEGRAM_TOKEN not configured in .env"
    exit 1
fi

if ! grep -q "OPENAI_API_KEY=your_" .env && grep -q "OPENAI_API_KEY=" .env; then
    echo "✅ OPENAI_API_KEY found"
else
    echo "❌ OPENAI_API_KEY not configured in .env"
    exit 1
fi

# Create data directory
mkdir -p data

# Check for voice file
if [ -f "MI voz.wav" ]; then
    echo "✅ Voice file found"
else
    echo "⚠️  Warning: MI voz.wav not found. Voice cloning will not work."
    echo "   Add a Spanish voice sample (3-10 seconds, WAV format) to enable TTS."
fi

# Pull latest image and start
echo "📦 Pulling latest image..."
docker-compose pull

echo "🚀 Starting services..."
docker-compose up -d

echo ""
echo "✅ Bot deployed successfully!"
echo ""
echo "📊 Check status:"
echo "  docker-compose ps"
echo ""
echo "📝 View logs:"
echo "  docker-compose logs -f megan-spanish-bot"
echo ""
echo "🛑 Stop bot:"
echo "  docker-compose down"
