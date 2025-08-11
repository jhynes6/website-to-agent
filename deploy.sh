#!/bin/bash

echo "🚀 Deploying Website-to-Agent to Digital Ocean App Platform..."

# Check if doctl is installed
if ! command -v doctl &> /dev/null; then
    echo "❌ doctl CLI is not installed. Please install it first:"
    echo "   brew install doctl"
    echo "   or visit: https://docs.digitalocean.com/reference/doctl/how-to/install/"
    exit 1
fi

# Check if user is authenticated
if ! doctl account get &> /dev/null; then
    echo "❌ You're not authenticated with Digital Ocean."
    echo "   Please run: doctl auth init"
    exit 1
fi

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ OPENAI_API_KEY environment variable is not set."
    echo "   Please export your OpenAI API key:"
    echo "   export OPENAI_API_KEY='your_api_key_here'"
    exit 1
fi

# Replace the placeholder in the app.yaml file
sed -i.bak "s/YOUR_OPENAI_API_KEY_HERE/$OPENAI_API_KEY/" .do/app.yaml

echo "✅ Creating Digital Ocean App..."

# Create the app
doctl apps create .do/app.yaml

# Restore the original app.yaml file
mv .do/app.yaml.bak .do/app.yaml

echo "✅ App deployment initiated!"
echo "🌐 Your app will be available at: https://chat.lauragonzales.com"
echo "📊 Check deployment status with: doctl apps list"

echo ""
echo "🔧 Next steps:"
echo "1. Wait for deployment to complete (check in DO dashboard)"
echo "2. Configure DNS for chat.lauragonzales.com to point to your app"
echo "3. Test your application!"
