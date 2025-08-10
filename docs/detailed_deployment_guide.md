# Detailed Deployment Guide for AI Data Analysis Assistant

This guide will walk you through the process of deploying your AI Data Analysis Assistant application to Streamlit Community Cloud for permanent hosting.

## Prerequisites

- A GitHub account
- A web browser

## Step 1: Create a GitHub Repository

1. Go to [GitHub](https://github.com) and sign in to your account
2. Click on the "+" icon in the top-right corner and select "New repository"
3. Fill in the repository details:
   - Repository name: `ai-data-assistant`
   - Description: `AI-driven data analysis assistant using Streamlit and Gemini API`
   - Visibility: Public (required for free Streamlit Community Cloud deployment)
   - Initialize with a README: No (we'll upload our own files)
4. Click "Create repository"

## Step 2: Upload Files to GitHub

### Option 1: Using GitHub Web Interface (Easiest)

1. In your new repository, click on "uploading an existing file" link
2. Unzip the `ai_data_assistant.zip` file on your computer
3. Drag and drop all the files and folders from the unzipped directory to the GitHub upload area
   - Make sure to include: `app.py`, `requirements.txt`, `.gitignore`, `README.md`, and the `src` and `docs` folders
   - Do NOT upload the `.env` file (it contains your API key)
4. Add a commit message like "Initial commit" and click "Commit changes"

### Option 2: Using Git Command Line (For Advanced Users)

If you're familiar with Git, you can clone the repository and push the files:

```bash
git clone https://github.com/yourusername/ai-data-assistant.git
cd ai-data-assistant
# Copy all files from the unzipped ai_data_assistant.zip here
git add .
git commit -m "Initial commit"
git push origin main
```

## Step 3: Create a Streamlit Community Cloud Account

1. Go to [Streamlit Community Cloud](https://streamlit.io/cloud)
2. Click "Sign in with GitHub"
3. Authorize Streamlit to access your GitHub account
4. Complete any additional registration steps if prompted

## Step 4: Deploy Your App

1. In Streamlit Community Cloud, click "New app" button
2. Select your GitHub account and the `ai-data-assistant` repository
3. For the branch, select "main"
4. For the main file path, enter "app.py"
5. Under "Advanced settings", add your Gemini API key:
   - Click "Add secrets"
   - Enter the following in the text area:
     ```
     GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
     ```
   - Click "Save"
6. Click "Deploy!"

## Step 5: Access Your Deployed App

1. Streamlit will now build and deploy your app (this may take a few minutes)
2. Once deployed, you'll see a success message and a link to your app
3. Click the link to open your app in a new tab
4. Your app now has a permanent URL that you can share with others

## Troubleshooting

### If the app fails to deploy:

1. Check the build logs for errors
2. Common issues include:
   - Missing dependencies: Make sure requirements.txt is correct
   - API key not set: Verify you added the GEMINI_API_KEY secret
   - File path issues: Ensure app.py is in the root directory

### If the app deploys but doesn't work correctly:

1. Check the app logs for errors
2. Verify the API key is correctly set in the secrets
3. Make sure all required files are in the repository

## Updating Your App

To make changes to your app:

1. Update the files in your GitHub repository
2. Streamlit Community Cloud will automatically detect the changes and rebuild your app

## Security Notes

- Never commit your API key directly in the code
- Always use Streamlit's secrets management for sensitive information
- Regularly rotate your API keys for better security
