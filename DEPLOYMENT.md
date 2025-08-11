# Deployment Guide: Website-to-Agent on Digital Ocean App Platform

This guide will help you deploy your Streamlit application to Digital Ocean App Platform so it's accessible at `https://chat.lauragonzales.com`.

## Prerequisites

1. **Digital Ocean Account**: Ensure you have a Digital Ocean account with App Platform access
2. **doctl CLI**: Digital Ocean command-line tool
3. **OpenAI API Key**: Required for the application to function
4. **Domain Access**: Ability to modify DNS records for `lauragonzales.com`

## Step-by-Step Deployment

### 1. Install Digital Ocean CLI

```bash
# On macOS
brew install doctl

# Or download from: https://docs.digitalocean.com/reference/doctl/how-to/install/
```

### 2. Authenticate with Digital Ocean

```bash
# Initialize authentication
doctl auth init

# Follow the prompts to enter your API token
# You can get your API token from: https://cloud.digitalocean.com/account/api/tokens
```

### 3. Set Up Environment Variables

```bash
# Export your OpenAI API key
export OPENAI_API_KEY='your_openai_api_key_here'

# Verify it's set
echo $OPENAI_API_KEY
```

### 4. Deploy the Application

**Option A: Use the automated script**

```bash
./deploy.sh
```

**Option B: Manual deployment**

```bash
# Update the app.yaml with your API key
sed -i.bak "s/YOUR_OPENAI_API_KEY_HERE/$OPENAI_API_KEY/" .do/app.yaml

# Create the app
doctl apps create .do/app.yaml

# Restore original file
mv .do/app.yaml.bak .do/app.yaml
```

### 5. Monitor Deployment

```bash
# List your apps
doctl apps list

# Check deployment status (replace APP_ID with your actual app ID)
doctl apps get APP_ID

# View app logs
doctl apps logs APP_ID
```

### 6. Configure DNS

After deployment, you need to configure DNS for `chat.lauragonzales.com`:

1. Get your app's URL from the Digital Ocean dashboard
2. Create a CNAME record:
   - **Name**: `chat`
   - **Target**: Your app's Digital Ocean URL (something like `your-app-name-xxxxx.ondigitalocean.app`)

### 7. Verify Deployment

Once DNS propagation is complete (usually 5-15 minutes), visit:
- `https://chat.lauragonzales.com`

## Configuration Details

### App Specifications
- **Region**: NYC1
- **Instance Type**: Basic XXS (suitable for light traffic)
- **Port**: 8501 (Streamlit default)
- **Health Check**: HTTP on root path `/`
- **Auto-Deploy**: Enabled on push to main branch

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key (stored as secret)
- `STREAMLIT_SERVER_PORT`: 8501
- `STREAMLIT_SERVER_ADDRESS`: 0.0.0.0

## Scaling and Updates

### Scaling Up
To handle more traffic, you can upgrade the instance size:

```bash
# Update the app.yaml file with a larger instance_size_slug
# Options: basic-xxs, basic-xs, basic-s, basic-m, etc.
doctl apps update APP_ID --spec .do/app.yaml
```

### Updating the App
The app will automatically redeploy when you push changes to the main branch on GitHub.

## Troubleshooting

### Common Issues

1. **Authentication Failed**
   - Ensure your Digital Ocean API token is valid
   - Run `doctl auth init` to re-authenticate

2. **Deployment Fails**
   - Check app logs: `doctl apps logs APP_ID`
   - Verify all environment variables are set correctly
   - Ensure your GitHub repository is accessible

3. **App Not Responding**
   - Check health check configuration
   - Verify port 8501 is exposed and accessible
   - Review Streamlit server configuration

4. **Domain Not Working**
   - Verify DNS configuration
   - Allow time for DNS propagation (up to 48 hours)
   - Check SSL certificate provisioning in DO dashboard

### Useful Commands

```bash
# List apps
doctl apps list

# Get app details
doctl apps get APP_ID

# View logs
doctl apps logs APP_ID --type build
doctl apps logs APP_ID --type deploy
doctl apps logs APP_ID --type run

# Delete app (if needed)
doctl apps delete APP_ID
```

## Cost Estimation

- Basic XXS instance: ~$5-10/month
- Domain management: Included
- SSL certificate: Free (Let's Encrypt)

## Support

For issues:
1. Check the [Digital Ocean App Platform documentation](https://docs.digitalocean.com/products/app-platform/)
2. Review the [Streamlit deployment guide](https://docs.streamlit.io/streamlit-cloud/get-started/deploy-an-app)
3. Contact Digital Ocean support through their dashboard

---

Your website-to-agent application will be live at `https://chat.lauragonzales.com` once deployment is complete!
