# AI Data Analysis Assistant

An AI-driven data analysis assistant that simulates a team of AI personas (Manager, Analyst, Associate) to guide users through a structured data analysis process based on uploaded CSV data.

## Features

- **AI Manager**: Creates structured analysis plans based on your project goals
- **AI Analyst**: Examines data details and executes specific analysis tasks
- **AI Associate**: Provides guidance, hypotheses, and insights for analysis
- **Interactive Reports**: Generate comprehensive HTML reports of your analysis
- **CSV Data Processing**: Upload and analyze your CSV data files
- **Conversation History**: Track interactions with different AI personas

## Requirements

- Python 3.10 or higher
- Gemini API key (free tier)
- Plotly for interactive visualizations

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/ai-data-assistant.git
   cd ai-data-assistant
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your Gemini API key:
   
   Create a `.env` file in the project root directory with the following content:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
   
   Replace `your_gemini_api_key_here` with your actual Gemini API key.

## Running the Application

Run the Streamlit application:
```
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Deployment Options

### Option 1: Streamlit Community Cloud (Recommended for non-technical users)

1. Create a free account on [Streamlit Community Cloud](https://streamlit.io/cloud)
2. Connect your GitHub repository
3. Add your Gemini API key as a secret in the Streamlit deployment settings
4. Deploy the application with a single click

This option provides a public URL that you can share with others without requiring them to install anything.

### Option 2: Local Installation

Follow the installation and running instructions above. This requires Python knowledge and installing dependencies locally.

## How to Use

1. **Start a New Project**:
   - Enter a project name and problem statement
   - Upload one or more CSV files
   - Optionally provide context about your data

2. **Review the Manager's Plan**:
   - The AI Manager will create a structured analysis plan
   - You can provide feedback to refine the plan

3. **Understand Your Data**:
   - The AI Analyst will examine your data and provide a summary
   - You can ask questions about your data

4. **Get Analysis Guidance**:
   - The AI Associate will provide specific hypotheses and tasks
   - You can provide feedback to refine the guidance

5. **Execute Analysis Tasks**:
   - The AI Analyst will execute specific analysis tasks
   - You can request additional analyses

6. **Generate a Final Report**:
   - The AI Manager will synthesize all findings into a comprehensive report
   - You can download the report as an HTML file

## Project Structure

- `app.py`: Main Streamlit application
- `src/utils.py`: Utility functions for Gemini API integration and data processing
- `requirements.txt`: Required Python dependencies
- `.env`: Environment variables (not included in repository)

## Security Note

This application uses environment variables to securely store your Gemini API key. Never hardcode your API key directly in the source code or commit it to version control.

## Getting a Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy the key and add it to your `.env` file

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Google Gemini API for AI capabilities
- Streamlit for the web application framework
- Pandas for data processing
