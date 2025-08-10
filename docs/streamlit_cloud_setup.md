# Streamlit Community Cloud Setup Guide

This guide provides specific instructions for setting up your AI Data Analysis Assistant on Streamlit Community Cloud after you've created your GitHub repository.

## Creating Your Streamlit Community Cloud Account

1. Go to [Streamlit Community Cloud](https://streamlit.io/cloud)
2. Click "Sign in with GitHub"
3. Authorize Streamlit to access your GitHub account
4. Complete any additional registration steps if prompted

## Deploying Your Application

### Step 1: Access Your Dashboard
After signing in, you'll be taken to your Streamlit Community Cloud dashboard.

### Step 2: Create a New App
1. Click the "New app" button in the top-right corner
2. You'll see a deployment form with the following fields:

### Step 3: Configure Your App
Fill in the deployment form:
1. **Repository**: Select your GitHub account and the `ai-data-assistant` repository
2. **Branch**: Select "main"
3. **Main file path**: Enter "app.py"
4. **App URL**: This will be auto-generated based on your repository name, but you can customize it if desired

### Step 4: Add Your Secrets
1. Expand the "Advanced settings" section
2. Click "Add secrets"
3. In the text area that appears, enter the following:
   ```
   GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
   ```
   (Replace with your actual API key if different)
4. Click "Save"

### Step 5: Deploy
1. Click the "Deploy!" button
2. Streamlit will now build and deploy your app (this may take a few minutes)
3. You'll see a progress indicator during the deployment process

## After Deployment

### Accessing Your App
Once deployment is complete:
1. You'll see a success message with a link to your app
2. Click the link to open your app in a new browser tab
3. Your app now has a permanent URL that you can share with others

### Managing Your App
From your Streamlit Community Cloud dashboard, you can:
1. **View app details**: Click on your app name to see deployment details
2. **View logs**: Click "Manage app" then "Logs" to see application logs
3. **Modify settings**: Click "Manage app" then "Settings" to change app configuration
4. **Restart app**: Click "Manage app" then "Settings" and use the "Reboot app" option

### Updating Your App
To update your application:
1. Make changes to your files in the GitHub repository
2. Commit and push the changes
3. Streamlit Community Cloud will automatically detect the changes and rebuild your app

## Troubleshooting Common Issues

### App Fails to Deploy
If your app fails to deploy, check the build logs for errors:
1. Common issues include:
   - Missing dependencies in requirements.txt
   - Syntax errors in your Python code
   - Incorrect file paths

### App Deploys But Doesn't Work
If your app deploys but doesn't function correctly:
1. Check that your GEMINI_API_KEY secret is correctly set
2. Verify that all required files are in the correct locations
3. Check the app logs for runtime errors

### API Key Issues
If you see errors related to the Gemini API:
1. Verify your API key is correct
2. Check that the key has been properly added to the secrets
3. Ensure your API key has the necessary permissions

## Resource Limits

Streamlit Community Cloud provides free hosting with some limitations:
1. Apps go to sleep after a period of inactivity
2. Limited computational resources
3. Public visibility (anyone with the link can access your app)

For production use cases with higher requirements, consider upgrading to a paid plan or using alternative deployment options.

## Getting Help

If you encounter issues not covered in this guide:
1. Check the [Streamlit Community Cloud documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud)
2. Visit the [Streamlit forum](https://discuss.streamlit.io/) for community support
3. Contact Streamlit support through their website
