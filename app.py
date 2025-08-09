import streamlit as st
import os
import json
import polars as pl
from pathlib import Path
import asyncio
import subprocess
import venv
import plotly.express as px


class SecurityManager:
    """Basic security checks for code execution."""

    def validate_code(self, code: str) -> bool:
        """Validate code for security concerns."""
        forbidden_imports = ["os.system", "subprocess"]
        return all(imp not in code for imp in forbidden_imports)


class VirtualEnvManager:
    """Handles creation of isolated Python environments."""

    def __init__(self):
        self.base_dir = Path.home() / ".ai-data-analysis"
        self.envs_dir = self.base_dir / "environments"
        self.packages_dir = self.base_dir / "packages"
        self.envs_dir.mkdir(parents=True, exist_ok=True)
        self.packages_dir.mkdir(parents=True, exist_ok=True)

    def create_environment(self, name: str = "default") -> Path:
        """Create a virtual environment with pip enabled."""
        env_path = self.envs_dir / name
        if not env_path.exists():
            venv.create(env_path, with_pip=True)
        return env_path

    def install_package(self, package: str, env_name: str = "default") -> subprocess.CompletedProcess:
        """Install a package into the specified environment."""
        env_path = self.create_environment(env_name)
        pip_executable = env_path / "bin" / "pip"
        result = subprocess.run(
            [str(pip_executable), "install", package],
            capture_output=True,
            text=True,
        )
        return result


class CodeExecutor:
    """Execute arbitrary Python code inside a managed virtualenv."""

    def __init__(self, env_manager: VirtualEnvManager):
        self.env_manager = env_manager
        self.timeout = 30

    async def execute_code(self, code: str, env_name: str = "default") -> subprocess.CompletedProcess:
        """Execute the provided code asynchronously.

        Parameters
        ----------
        code: str
            Python code to run.
        env_name: str
            Name of the virtual environment in which to run the code.
        """
        env_path = self.env_manager.create_environment(env_name)
        python_executable = env_path / "bin" / "python"
        process = await asyncio.create_subprocess_exec(
            str(python_executable),
            "-c",
            code,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.timeout)
        except asyncio.TimeoutError:
            process.kill()
            stdout, stderr = await process.communicate()
            stderr += b"\nExecution timed out"
        return subprocess.CompletedProcess(
            args=[str(python_executable), "-c", code],
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
        )

    def execute(self, code: str, env_name: str = "default") -> subprocess.CompletedProcess:
        """Convenience wrapper to run ``execute_code`` synchronously."""
        return asyncio.run(self.execute_code(code, env_name))


class AnalysisDashboard:
    """UI components for interactive code execution and visualization."""

    def __init__(self):
        self.env_manager = VirtualEnvManager()
        self.executor = CodeExecutor(self.env_manager)
        self.security = SecurityManager()

    def render_code_section(self):
        with st.container():
            st.markdown("### 📝 Code Editor")

            # Package Management
            with st.expander("📦 Manage Dependencies"):
                self._render_package_manager()

            # Code Editor
            self._render_code_editor()

            # Execution Controls
            self._render_execution_controls()

    def _render_package_manager(self):
        col1, col2 = st.columns([3, 1])
        with col1:
            pkg = st.text_input("Package Name", key="pkg_name", placeholder="e.g., pandas==2.0.0")
        with col2:
            if st.button("Install", key="install_pkg_btn") and pkg:
                result = self.env_manager.install_package(pkg)
                st.text(result.stdout)
                if result.stderr:
                    st.text(result.stderr)

    def _render_code_editor(self):
        st.text_area("Python Code", height=300, key="code_editor")

    def _render_execution_controls(self):
        if st.button("Run Code", key="run_code_btn"):
            code = st.session_state.get("code_editor", "")
            if not code:
                st.warning("Please enter code to run.")
                return
            if not self.security.validate_code(code):
                st.error("Security check failed for the provided code.")
                return
            result = self.executor.execute(code)
            st.text_area("stdout", result.stdout.decode(), height=150)
            if result.stderr:
                st.text_area("stderr", result.stderr.decode(), height=150)


class UserPreferences:
    """Manages user customization settings."""

    def __init__(self):
        self.base_path = Path.home() / ".ai-data-analysis" / "preferences.json"
        self.theme = "light"
        self.shortcuts = {}
        self.load()

    def load(self):
        if self.base_path.exists():
            data = json.loads(self.base_path.read_text())
            self.theme = data.get("theme", "light")
            self.shortcuts = data.get("shortcuts", {})

    def save(self):
        self.base_path.parent.mkdir(parents=True, exist_ok=True)
        self.base_path.write_text(
            json.dumps({"theme": self.theme, "shortcuts": self.shortcuts}, indent=2)
        )

from datetime import datetime


class ProjectManager:
    """Manage saving and loading of project states."""

    def __init__(self):
        self.base_dir = Path.home() / ".ai-data-analysis" / "projects"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_state(self, name: str = "default") -> Path:
        """Persist current project state to disk."""
        state = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user": st.session_state.get("user", "anonymous"),
            "analysis_state": st.session_state.to_dict(),
        }
        path = self.base_dir / f"{name}.json"
        path.write_text(json.dumps(state, indent=2))
        return path

    def load_state(self, name: str = "default"):
        """Load a saved project state into ``st.session_state``."""
        path = self.base_dir / f"{name}.json"
        if not path.exists():
            return None
        state = json.loads(path.read_text())
        for key, value in state.get("analysis_state", {}).items():
            st.session_state[key] = value
        return state


HELP_TEXT = {
    "setup": "Fill in project details and upload your data to begin.",
    "manager_planning": "The manager persona will outline a strategy for the analysis.",
    "data_understanding": "Review uploaded datasets and their basic statistics.",
    "analysis_execution": "Generate and optionally run analysis code here.",
    "final_report": "Compile all results into a final summary report.",
}


class HelpSystem:
    """Displays contextual help in the sidebar."""

    def show_contextual_help(self, context: str):
        help_content = self.get_help_content(context)
        with st.sidebar:
            st.markdown(f"### Help: {context}")
            st.markdown(help_content)

    def get_help_content(self, context: str) -> str:
        return HELP_TEXT.get(context, "No help available yet.")

# Import functions from utils, prompts, and helpers
import google.generativeai as genai
from dotenv import load_dotenv
import docx
from PyPDF2 import PdfReader

# Load environment variables
load_dotenv()

def escape_curly_braces(text: str) -> str:
    """Escape curly braces in text for safe use with str.format."""
    return text.replace("{", "{{").replace("}", "}}") if isinstance(text, str) else text

# Configure Gemini API
def configure_genai(api_key=None):
    """
    Configure the Gemini API with the provided key or environment variable.
    """
    # Prioritize explicitly passed key, then session state, then env var
    key_to_use = api_key if api_key \
        else st.session_state.get("gemini_api_key", os.getenv("GEMINI_API_KEY"))

    if not key_to_use:
        # Let the main app handle UI warnings/errors for missing keys during runtime
        # This function might be called before session_state is fully available initially
        print("Warning: No Gemini API key found during configure_genai call.")
        return False
    if genai is None:
        error_message = (
            "google-generativeai package is not installed. "
            "Install dependencies with `pip install -r requirements.txt`."
        )
        try:
            st.error(error_message)
        except Exception:
            print(error_message)
        return False
    try:
        genai.configure(api_key=key_to_use)
        # print("Gemini API Configured Successfully.") # Optional debug print
        return True
    except Exception as e:
        # Use st.error only if Streamlit context is guaranteed, otherwise print
        error_message = f"Error configuring Gemini API: {str(e)}"
        try:
            st.error(error_message)
        except Exception:
            print(error_message)
        return False


# Function to generate response from Gemini API
def get_gemini_response(prompt: str, persona: str = "general", model: str | None = None, api_key: str | None = None) -> str:
    """
    Get a response from the Gemini API using the provided complete prompt.

    Args:
        prompt (str): The full prompt (including context, instructions, persona)
                      formatted by the calling application.
        persona (str): Informational only, not used for system prompt here.
        model (str | None): The specific model ID to use (e.g., "gemini-1.5-flash-latest").
                             Defaults to session state or "gemini-1.5-flash-latest".
        api_key (str | None): The API key. Defaults to session state or env var.

    Returns:
        str: The text response from the API or an error message.
    """
    # Use API key and model from args, session state, or defaults
    current_api_key = api_key if api_key \
        else st.session_state.get("gemini_api_key", os.getenv("GEMINI_API_KEY"))
    # Use a known good default if session state isn't set or model arg is None
    current_model = model if model \
        else st.session_state.get("gemini_model", "gemini-1.5-flash-latest")

    if not current_api_key:
        return "Error: Gemini API key not configured. Please add it in the sidebar settings."

    # Ensure API is configured with the current key for this call
    if not configure_genai(current_api_key):
         # Configuration failed, error likely shown by configure_genai
         return "Error: Failed to configure Gemini API with the provided key." \
                " or missing dependencies."

    if genai is None:
        return (
            "Error: google-generativeai package not installed. "
            "Run `pip install -r requirements.txt`."
        )

    try:
        # Initialize the model - No system_instruction here, it's part of the main 'prompt'
        # The main app prepares the prompt using the templates from session state.
        genai_model = genai.GenerativeModel(current_model)

        # Generate response using the full prompt passed from the main app
        response = genai_model.generate_content(
            prompt,
            generation_config={"temperature": 0.2} # Keep temperature low for consistency
        )

        # Safely access response text
        return response.text if hasattr(response, 'text') else "Error: Received empty response from API."

    except Exception as e:
        error_message = f"Error generating Gemini response: {str(e)}"
        try:
            st.error(error_message) # Show error in Streamlit UI if possible
        except Exception:
            print(error_message) # Fallback to console print
        return error_message


# --- File Processing Functions ---

def _generate_polars_profile(df: pl.DataFrame) -> dict:
    """Helper function to generate profile dict from a Polars DataFrame."""
    profile = {
        "file_type": "tabular",
        "columns": df.columns,
        "shape": df.shape, # Returns (height, width)
        "dtypes": {col: str(dtype) for col, dtype in df.schema.items()},
        "missing_summary": None, # Placeholder for missing summary DataFrame
        "numeric_summary": None # Placeholder for describe DataFrame
    }

    try:
        # Calculate missing values (returns a DataFrame)
        missing_df = df.null_count()
        # Convert the null count DataFrame to a more usable format if needed,
        # e.g., a dictionary {column: count}. For now, store the DF itself.
        # Check if the null_count DF has the expected structure before storing
        if missing_df.shape == (1, df.width): # Expected shape: 1 row, N columns
            profile["missing_summary"] = missing_df
        else:
             # Handle unexpected shape or create dict manually if needed
             profile["missing_summary"] = pl.DataFrame({"error": ["Unexpected null_count shape"]})


    except Exception as e:
        print(f"Error calculating missing values: {e}")
        profile["missing_summary"] = pl.DataFrame({"error": [f"Missing value calculation error: {e}"]})


    try:
        # Generate summary statistics for all columns (Polars describe works on all)
        # Note: Polars describe includes non-numeric stats like 'unique', 'null_count'
        describe_df = df.describe()
        profile["numeric_summary"] = describe_df # Store the full describe DF
    except Exception as e:
        print(f"Error generating describe summary: {e}")
        profile["numeric_summary"] = pl.DataFrame({"error": [f"Describe calculation error: {e}"]})


    return profile

# Function to read and process CSV files using Polars
def process_csv_file(uploaded_file):
    """
    Process an uploaded CSV file using Polars.

    Args:
        uploaded_file: The uploaded file object from Streamlit.

    Returns:
        pl.DataFrame | None: The processed Polars DataFrame or None on error.
        dict | None: A profile dict of the data or None on error.
        str: Extracted text content (empty for CSV).
    """
    try:
        # Read the CSV file content into bytes, then let Polars handle it
        file_content = io.BytesIO(uploaded_file.getvalue())
        df = pl.read_csv(file_content)
        profile = _generate_polars_profile(df)
        return df, profile, ""
    except Exception as e:
        error_message = f"Error processing CSV file '{uploaded_file.name}': {str(e)}"
        try:
            st.error(error_message)
        except Exception:
            print(error_message)
        return None, None, ""

# Function to read and process Excel files using Polars
def process_excel_file(uploaded_file):
    """
    Process an uploaded Excel file using Polars.

    Args:
        uploaded_file: The uploaded file object from Streamlit.

    Returns:
        pl.DataFrame | None: The processed Polars DataFrame or None on error.
        dict | None: A profile dict of the data or None on error.
        str: Extracted text content (empty for Excel).
    """
    try:
        # Read the Excel file content into bytes
        file_content = io.BytesIO(uploaded_file.getvalue())
        # Polars read_excel might need engine specification if 'xlsx'
        # It often uses 'xlsx2csv' or connectorx internally. Ensure engine is installed if needed.
        df = pl.read_excel(file_content, engine='openpyxl') # Specify engine if needed
        profile = _generate_polars_profile(df)
        return df, profile, ""
    except ImportError:
         # Handle missing engine specifically
         err_msg = "Error: Missing engine for Excel processing. Try `pip install openpyxl`."
         try: st.error(err_msg)
         except: print(err_msg)
         return None, None, ""
    except Exception as e:
        error_message = f"Error processing Excel file '{uploaded_file.name}': {str(e)}"
        try:
            st.error(error_message)
        except Exception:
            print(error_message)
        return None, None, ""

# Function to extract text from DOCX files
def extract_text_from_docx(uploaded_file):
    """
    Extract text content from a DOCX file.

    Args:
        uploaded_file: The uploaded file object from Streamlit.

    Returns:
        str: The extracted text content, or empty string on error.
    """
    try:
        # Read directly from the uploaded file object
        doc = docx.Document(uploaded_file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        error_message = f"Error extracting text from DOCX '{uploaded_file.name}': {str(e)}"
        try:
            st.error(error_message)
        except Exception:
            print(error_message)
        return ""

# Function to extract text from PDF files
def extract_text_from_pdf(uploaded_file):
    """
    Extract text content from a PDF file.

    Args:
        uploaded_file: The uploaded file object from Streamlit.

    Returns:
        str: The extracted text content, or empty string on error.
    """
    try:
        # PdfReader works directly with the file-like object
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text: # Add text only if extraction was successful
                text += page_text + "\n"
        return text
    except Exception as e:
        error_message = f"Error extracting text from PDF '{uploaded_file.name}': {str(e)}"
        try:
            st.error(error_message)
        except Exception:
            print(error_message)
        return ""

# Central function to process uploaded files based on type
def process_uploaded_file(uploaded_file):
    """
    Process an uploaded file based on its type (CSV, XLSX, DOCX, PDF).

    Args:
        uploaded_file: The uploaded file object from Streamlit.

    Returns:
        pl.DataFrame | None: Processed Polars DataFrame (None if not tabular/error).
        dict | None: Profile dict (None if not tabular/error).
        str: Extracted text content (empty if tabular/error).
    """
    if not uploaded_file:
        return None, None, ""

    try:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()

        if file_extension == ".csv":
            df, profile, text = process_csv_file(uploaded_file)
            return df, profile, text # df/profile might be None on error
        elif file_extension in [".xlsx", ".xls"]:
            df, profile, text = process_excel_file(uploaded_file)
            return df, profile, text # df/profile might be None on error
        elif file_extension == ".docx":
            text = extract_text_from_docx(uploaded_file)
            # Create a simple profile for text files
            profile = {"file_type": "docx", "text_length": len(text)}
            return None, profile, text
        elif file_extension == ".pdf":
            text = extract_text_from_pdf(uploaded_file)
            profile = {"file_type": "pdf", "text_length": len(text)}
            return None, profile, text
        else:
            st.warning(f"Unsupported file type: {file_extension} for file '{uploaded_file.name}'")
            return None, None, ""
    except Exception as e:
        # Catch-all for unexpected errors during dispatch
        error_message = f"Unexpected error processing file '{uploaded_file.name}': {str(e)}"
        try:
            st.error(error_message)
        except Exception:
            print(error_message)
        return None, None, ""


# Function to generate a structured data profile dictionary
def generate_data_profile_summary(profile: dict | None) -> dict:
    """
    Generate a structured data profile dictionary from the raw profile.

    Args:
        profile (dict | None): The raw data profile dictionary generated by
                               _generate_polars_profile or for text files.

    Returns:
        dict: A structured dictionary containing profile information,
              or an empty dict if no profile is available.
    """
    if not profile:
        return {}

    file_type = profile.get("file_type", "unknown")
    structured_profile = {"file_type": file_type}

    if file_type == "tabular":
        structured_profile["shape"] = profile.get('shape', ('N/A', 'N/A'))
        structured_profile["columns"] = profile.get('columns', [])
        structured_profile["dtypes"] = profile.get('dtypes', {})
        structured_profile["missing_summary"] = profile.get('missing_summary') # Keep as Polars DF
        structured_profile["numeric_summary"] = profile.get('numeric_summary') # Keep as Polars DF

    elif file_type in ["docx", "pdf"]:
         structured_profile["text_length"] = profile.get("text_length", "N/A")
         structured_profile["text_snippet"] = profile.get("text_snippet", "") # Assuming snippet might be added later

    return structured_profile
from prompts import (
    MANAGER_PROMPT_TEMPLATE, ANALYST_PROMPT_TEMPLATE, ASSOCIATE_PROMPT_TEMPLATE,
    ANALYST_TASK_PROMPT_TEMPLATE, ASSOCIATE_REVIEW_PROMPT_TEMPLATE,
    MANAGER_REPORT_PROMPT_TEMPLATE, REVIEWER_PROMPT_TEMPLATE
)
import base64

def add_download_buttons(step_name):
    """Adds download buttons for session state data."""
    st.sidebar.markdown("---")
    st.sidebar.subheader(f"{step_name} Data Download")

    # Prepare data for download
    download_data = {}
    if st.session_state.get('dataframes'):
        # Convert Polars DataFrames to Pandas for to_csv/to_excel compatibility
        download_data['dataframes'] = {
            name: df.to_pandas().to_csv(index=False).encode('utf-8')
            for name, df in st.session_state.dataframes.items()
        }
        download_data['excel_dataframes'] = {}
        for name, df in st.session_state.dataframes.items():
            # Use BytesIO to write to memory
            output = io.BytesIO()
            df.to_pandas().to_excel(output, index=False)
            processed_data = output.getvalue()
            download_data['excel_dataframes'][name] = processed_data

    if st.session_state.get('data_profiles'):
        # Prepare data_profiles for JSON serialization by converting Polars DataFrames to strings
        serializable_profiles = {}
        for filename, profile in st.session_state.data_profiles.items():
            serializable_profile = profile.copy() # Create a copy to modify
            if 'missing_summary' in serializable_profile and isinstance(serializable_profile['missing_summary'], pl.DataFrame):
                try:
                    # Convert Polars DataFrame to string representation
                    serializable_profile['missing_summary'] = serializable_profile['missing_summary'].to_pandas().to_string()
                except Exception as e:
                    serializable_profile['missing_summary'] = f"Error converting missing_summary to string: {e}"

            if 'numeric_summary' in serializable_profile and isinstance(serializable_profile['numeric_summary'], pl.DataFrame):
                 try:
                     # Convert Polars DataFrame to string representation
                     serializable_profile['numeric_summary'] = serializable_profile['numeric_summary'].to_pandas().to_string()
                 except Exception as e:
                     serializable_profile['numeric_summary'] = f"Error converting numeric_summary to string: {e}"

            serializable_profiles[filename] = serializable_profile

        download_data['data_profiles'] = json.dumps(serializable_profiles, indent=2).encode('utf-8')
    if st.session_state.get('data_texts'):
        download_data['data_texts'] = json.dumps(st.session_state.data_texts, indent=2).encode('utf-8')
    if st.session_state.get('analysis_results'):
         # Convert analysis_results to a JSON string
         analysis_results_json = json.dumps(st.session_state.analysis_results, indent=2)
         download_data['analysis_results'] = analysis_results_json.encode('utf-8')
    if st.session_state.get('manager_plan'):
         download_data['manager_plan'] = st.session_state.manager_plan.encode('utf-8')
    if st.session_state.get('analyst_summary'):
         download_data['analyst_summary'] = st.session_state.analyst_summary.encode('utf-8')
    if st.session_state.get('associate_guidance'):
         download_data['associate_guidance'] = st.session_state.associate_guidance.encode('utf-8')
    if st.session_state.get('final_report'):
         download_data['final_report'] = st.session_state.final_report.encode('utf-8')
    if st.session_state.get('conversation_history'):
         conversation_history_json = json.dumps(st.session_state.conversation_history, indent=2)
         download_data['conversation_history'] = conversation_history_json.encode('utf-8')
    if st.session_state.get('consultation_response'):
         download_data['consultation_response'] = st.session_state.consultation_response.encode('utf-8')
    if st.session_state.get('reviewer_response'):
         download_data['reviewer_response'] = st.session_state.reviewer_response.encode('utf-8')


    if download_data:
        # Create a zip file in memory
        # Note: Streamlit's built-in download_button is simpler for single files.
        # For multiple files, a zip is better, but requires more complex handling
        # or directing the user to save individual files. Let's stick to individual
        # downloads for simplicity with Streamlit's native widget.

        # Provide download buttons for each item
        if 'dataframes' in download_data:
            for name, data in download_data['dataframes'].items():
                st.sidebar.download_button(
                    label=f"Download {name} (CSV)",
                    data=data,
                    file_name=f"{step_name}_{name}.csv",
                    mime="text/csv",
                    key=f"download_csv_{step_name}_{name}"
                )
            for name, data in download_data['excel_dataframes'].items():
                 st.sidebar.download_button(
                     label=f"Download {name} (XLSX)",
                     data=data,
                     file_name=f"{step_name}_{name}.xlsx",
                     mime="application/vnd.openxmlformats-officedocument.spreadsheet.sheet",
                     key=f"download_xlsx_{step_name}_{name}"
                 )

        if 'data_profiles' in download_data:
            st.sidebar.download_button(
                label="Download Data Profiles (JSON)",
                data=download_data['data_profiles'],
                file_name=f"{step_name}_data_profiles.json",
                mime="application/json",
                key=f"download_profiles_{step_name}"
            )
        if 'data_texts' in download_data:
            st.sidebar.download_button(
                label="Download Text Data (JSON)",
                data=download_data['data_texts'],
                file_name=f"{step_name}_data_texts.json",
                mime="application/json",
                key=f"download_texts_{step_name}"
            )
        if 'analysis_results' in download_data:
             st.sidebar.download_button(
                 label="Download Analysis Results (JSON)",
                 data=download_data['analysis_results'],
                 file_name=f"{step_name}_analysis_results.json",
                 mime="application/json",
                 key=f"download_analysis_results_{step_name}"
             )
        if 'manager_plan' in download_data:
             st.sidebar.download_button(
                 label="Download Manager Plan (TXT)",
                 data=download_data['manager_plan'],
                 file_name=f"{step_name}_manager_plan.txt",
                 mime="text/plain",
                 key=f"download_manager_plan_{step_name}"
             )
        if 'analyst_summary' in download_data:
             st.sidebar.download_button(
                 label="Download Analyst Summary (TXT)",
                 data=download_data['analyst_summary'],
                 file_name=f"{step_name}_analyst_summary.txt",
                 mime="text/plain",
                 key=f"download_analyst_summary_{step_name}"
             )
        if 'associate_guidance' in download_data:
             st.sidebar.download_button(
                 label="Download Associate Guidance (TXT)",
                 data=download_data['associate_guidance'],
                 file_name=f"{step_name}_associate_guidance.txt",
                 mime="text/plain",
                 key=f"download_associate_guidance_{step_name}"
             )
        if 'final_report' in download_data:
             st.sidebar.download_button(
                 label="Download Final Report (TXT)",
                 data=download_data['final_report'],
                 file_name=f"{step_name}_final_report.txt",
                 mime="text/plain",
                 key=f"download_final_report_{step_name}"
             )
        if 'conversation_history' in download_data:
             st.sidebar.download_button(
                 label="Download Conversation History (JSON)",
                 data=download_data['conversation_history'],
                 file_name=f"{step_name}_conversation_history.json",
                 mime="application/json",
                 key=f"download_conversation_history_{step_name}"
             )
        if 'consultation_response' in download_data:
             st.sidebar.download_button(
                 label="Download Consultation Response (TXT)",
                 data=download_data['consultation_response'],
                 file_name=f"{step_name}_consultation_response.txt",
                 mime="text/plain",
                 key=f"download_consultation_response_{step_name}"
             )
        if 'reviewer_response' in download_data:
             st.sidebar.download_button(
                 label="Download Reviewer Response (TXT)",
                 data=download_data['reviewer_response'],
                 file_name=f"{step_name}_reviewer_response.txt",
                 mime="text/plain",
                 key=f"download_reviewer_response_{step_name}"
             )

def reset_session():
    """Resets the session state to initial values."""
    # Keep API key and potentially model/prompts if user wants to reuse them
    current_api_key = st.session_state.get('gemini_api_key', os.getenv("GEMINI_API_KEY", ""))
    current_model = st.session_state.get('gemini_model', "gemini-2.5-flash") # Use updated default
    current_lib_mgmt = st.session_state.get('library_management', "Manual")
    # Store prompts before clearing
    prompts = {k: v for k, v in st.session_state.items() if k.endswith('_prompt_template')}

    # Clear all session state
    st.session_state.clear()

    # Re-apply defaults
    defaults = {
        'project_initialized': False,
        'current_step': 0,
        'data_uploaded': False,
        'dataframes': {},        # Stores Polars DataFrames {filename: pl.DataFrame}
        'data_profiles': {},     # Stores basic profiles {filename: profile_dict}
        'data_texts': {},        # Stores text from non-tabular files {filename: text_string}
        'project_name': "Default Project",
        'problem_statement': "",
        'data_context': "",
        'manager_plan': None,
        'analyst_summary': None,
        'associate_guidance': None,
        'analysis_results': [],  # List to store dicts: [{"task": ..., "approach": ..., "code": ..., "results_text": ..., "insights": ...}]
        'final_report': None,
        'conversation_history': [], # List of {"role": ..., "content": ...}
        'consultation_response': None,
        'consultation_persona': None,
        'reviewer_response': None,
        'reviewer_specific_request': None,
        'gemini_api_key': os.getenv("GEMINI_API_KEY", ""), # Load from env var if available
        'gemini_model': "gemini-2.5-flash", # Updated default model
        'library_management': "Manual", # New setting: Manual / Automated
        # --- Prompt Templates --- (Load from prompts.py)
        # Note: These will be loaded from prompts.py in the main app,
        # but we include them here for completeness if this file were run standalone
        'manager_prompt_template': "", # Placeholder
        'analyst_prompt_template': "", # Placeholder
        'associate_prompt_template': "", # Placeholder
        'analyst_task_prompt_template': "", # Placeholder
        'associate_review_prompt_template': "", # Placeholder
        'manager_report_prompt_template': "", # Placeholder
        'reviewer_prompt_template': "" # Placeholder
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Restore persistent settings
    st.session_state.gemini_api_key = current_api_key
    st.session_state.gemini_model = current_model
    st.session_state.library_management = current_lib_mgmt
    st.session_state.update(prompts) # Restore prompts

    st.success("Project Reset!")


def add_to_conversation(role, content):
    """Adds a message to the conversation history."""
    st.session_state.conversation_history.append({
        "role": role,
        "content": str(content) # Ensure content is string
    })

def format_results_markdown(analysis_results):
    """Format a list of analysis results into Markdown."""

    if not analysis_results:
        return "No analysis results available."

    lines = ["## Analysis Results Summary", ""]

    for idx, result in enumerate(analysis_results, start=1):
        task_title = result.get("task", f"Task {idx}")
        lines.append(f"### Task {idx}: {task_title}")

        files = result.get("files")
        if files:
            if isinstance(files, (list, tuple)):
                file_list = ", ".join(str(f) for f in files)
            else:
                file_list = str(files)
            lines.append(f"**Files Used:** {file_list}")

        approach = result.get("approach")
        if approach:
            lines.append("**Approach:**")
            lines.append(approach.strip())

        code = result.get("code")
        if code:
            lines.append("**Python Code:**")
            lines.append(f"```python\n{code.strip()}\n```")

        results_text = result.get("results_text") or result.get("results")
        if results_text:
            lines.append("**Results:**")
            lines.append(results_text.strip())

        insights = result.get("insights")
        if insights:
            lines.append("**Key Insights:**")
            lines.append(insights.strip())

        # Include any figure or plot references if present
        fig_value = None
        for key in result.keys():
            lower_key = key.lower()
            if lower_key in {"figure_file", "figure", "plot_file", "plot_html"} or "figure" in lower_key or "plot" in lower_key:
                fig_value = result.get(key)
                if fig_value:
                    break

        if fig_value:
            if isinstance(fig_value, (list, tuple)):
                for fig in fig_value:
                    lines.append(f"**Figure File:** {fig}")
            else:
                lines.append(f"**Figure File:** {fig_value}")

        lines.append("\n---\n")

    return "\n".join(lines).strip()

# Function to check API key before AI calls
def check_api_key():
    if not st.session_state.gemini_api_key:
        st.error("Gemini API Key missing. Please enter it in the sidebar settings.")
        return False
    # Optional: Add a quick test call here if desired
    return True
import traceback

LOG_FILE = "crash_log.txt"


def log_exception(exc: Exception) -> None:
    """Append exception info and traceback to the crash log."""
    timestamp = datetime.utcnow().isoformat()
    trace = traceback.format_exc()
    entry = f"{timestamp} - {exc}\n{trace}\n\n"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry)


def read_log() -> str:
    """Return contents of the crash log if it exists."""
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def clear_log() -> None:
    """Delete the crash log file."""
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
def parse_associate_tasks(guidance_text):
    """
    Attempts to parse actionable tasks from the Associate's guidance.
    Extracts blocks starting with **Task N:** and stops at "5. Develop Narrative:".
    """
    tasks = []
    if not guidance_text:
        return tasks

    # Split the guidance text into lines, keeping line endings to preserve structure within blocks
    lines = guidance_text.strip().splitlines(keepends=True)

    # Regex to identify the start of a new Task block.
    # This pattern looks for lines starting with optional whitespace, followed by optional bolding (**),
    # then "Task", a space, a number, optional period, optional bolding, and a colon.
    task_header_pattern = re.compile(r"^\s*\**\s*Task\s*\d+\.?\s*\**:", re.IGNORECASE)

    current_task_block = []
    formatted_tasks = []

    for line in lines:
        # Check for the stop line first
        if "5. Develop Narrative:" in line:
            # If we were processing a task block, add its current content and stop
            if current_task_block: # Ensure block is not empty before appending
                formatted_tasks.append("".join(current_task_block).strip())
            break # Stop processing lines

        line_stripped = line.strip()
        header_match = task_header_pattern.match(line_stripped)

        if header_match:
            # If we were processing a previous task block, add it to the list
            if current_task_block: # Ensure block is not empty before appending
                formatted_tasks.append("".join(current_task_block).strip())

            # Start a new task block with the current line
            current_task_block = [line]
        # If it's not a task header but we are inside a task block, append the line
        elif current_task_block:
            current_task_block.append(line)

    # Add the last processed task block if it exists after the loop finishes
    if current_task_block:
         formatted_tasks.append("".join(current_task_block).strip())

    # Clean up empty strings that might result from parsing
    formatted_tasks = [task for task in formatted_tasks if task]

    # If no specific Task items were parsed, return the whole guidance as a single option
    # This might happen if the guidance doesn't use "Task N:" format
    if not formatted_tasks and guidance_text.strip():
         # Check if the guidance contains any lines that look like potential tasks (e.g., starting with *, -, or numbers)
         # If so, maybe return the whole guidance as one block, otherwise return default.
         # For now, let's just return the whole guidance if no specific tasks were found.
         formatted_tasks = [guidance_text.strip()]


    # If still empty, provide the default prompt
    if not formatted_tasks:
        try:
            st.warning("Could not automatically parse specific tasks from Associate guidance. Please manually define the task below.")
        except Exception:
            print("Warning: Could not automatically parse specific tasks from Associate guidance.")
        return ["Manually define task based on guidance above."] # Provide a default prompt

    # Ensure the "Manually define task below" option is always available
    if "Manually define task below" not in formatted_tasks:
        formatted_tasks.append("Manually define task below")

    # Remove duplicates while preserving order as much as possible (simple approach)
    seen = set()
    unique_formatted_tasks = []
    for task in formatted_tasks:
        if task not in seen:
            seen.add(task)
            unique_formatted_tasks.append(task)

    formatted_tasks = unique_formatted_tasks

    # Move "Manually define task below" to the end if it exists
    if "Manually define task below" in formatted_tasks:
        formatted_tasks.remove("Manually define task below")
        formatted_tasks.append("Manually define task below")

    # The final check for empty formatted_tasks was redundant and has been removed.
    # If the list was empty, it would have been handled by the check before adding "Manually define task below".

    return formatted_tasks

def parse_analyst_task_response(response_text):
    """
    Parses the Analyst's response into Approach, Code, Results, and Insights.
    Uses more robust header matching and content extraction.
    """
    if not response_text:
        print("Parsing Error: No response text provided.")
        return {"approach": "Error: No response from Analyst.", "code": "", "results_text": "", "insights": ""}

    # Define headers and their potential variations (case-insensitive, flexible spacing/numbering, optional markdown bolding)
    # Make bolding optional and handle variations in header text
    headers = {
        "approach": r"^\s*\d*\.?\s*\**approach\**:", # Optional number, period, optional bolding
        "code": r"^\s*\d*\.?\s*\**python?\s*code\**:", # Optional number, period, optional bolding, optional 'python'
        "results_text": r"^\s*\d*\.?\s*\**results\**:", # Optional number, period, optional bolding
        "insights": r"^\s*\d*\.?\s*\**key?\s*insights\**:", # Optional number, period, optional bolding, optional 'key'
    }

    parts = {
        "approach": "Could not parse 'approach' section.",
        "code": "Could not parse 'code' section.",
        "results_text": "Could not parse 'results_text' section.",
        "insights": "Could not parse 'insights' section."
    }

    # Find the start index of each section using the more flexible patterns
    section_matches = []
    for key, pattern in headers.items():
        for match in re.finditer(pattern, response_text, re.MULTILINE | re.IGNORECASE):
            section_matches.append({"key": key, "start": match.start(), "end": match.end()})

    # Sort found sections by their start index
    sorted_sections = sorted(section_matches, key=lambda x: x["start"])

    print(f"Found {len(sorted_sections)} potential section headers:")
    for match in sorted_sections:
        print(f"- Key: {match['key']}, Start: {match['start']}, End: {match['end']}")


    # Extract content for each section
    for i, section_match in enumerate(sorted_sections):
        key = section_match["key"]
        content_start_index = section_match["end"] # Content starts immediately after the header match

        # The content for the current section goes from content_start_index
        # up to the start index of the next *any* subsequent recognized header,
        # or the end of the text if this is the last recognized header.
        content_end_index = len(response_text)
        if i + 1 < len(sorted_sections):
            content_end_index = sorted_sections[i+1]["start"] # Start of the next found header

        content = response_text[content_start_index:content_end_index].strip()

        # Remove leading/trailing markdown bolding from content
        content = re.sub(r"^\s*\*\*", "", content).strip()
        content = re.sub(r"\*\*\s*$", "", content).strip()


        # Specific cleanup for code block
        if key == 'code':
            # Remove markdown code fences and language specifier
            content = re.sub(r"```python\n?|```", "", content, flags=re.IGNORECASE).strip()

        parts[key] = content
        print(f"Extracted content for '{key}':\n---\n{content[:200]}...\n---\n") # Print snippet of extracted content


    # Ensure all expected keys are present, even if parsing failed
    expected_keys = ["approach", "code", "results_text", "insights"]
    for key in expected_keys:
        if key not in parts:
             parts[key] = f"Could not parse '{key}' section."


    return parts

# Import feature functions

def display_final_report_step():
    """Displays the Final Report step."""
    st.title("📝 6. AI Manager - Final Report")
    if not check_api_key(): st.stop()

    st.info("This step synthesizes the project findings into a final report using the AI Manager.")

    # Check if prerequisites are met
    if not st.session_state.manager_plan or not st.session_state.analyst_summary or not st.session_state.analysis_results:
        missing = [] # Correct indentation for this block
        if not st.session_state.manager_plan: missing.append("Manager Plan (Step 2)")
        if not st.session_state.analyst_summary: missing.append("Analyst Summary (Step 3)")
        if not st.session_state.analysis_results: missing.append("Analysis Results (Step 5)")
        st.warning(f"Requires: {', '.join(missing)} to generate the report. Please complete previous steps.")
        # Add buttons to go back?
        if st.button("Go to Manager Planning"): st.session_state.current_step = 1; st.rerun()
        if st.button("Go to Data Understanding"): st.session_state.current_step = 2; st.rerun()
        if st.button("Go to Analysis Execution"): st.session_state.current_step = 4; st.rerun()
        st.stop()

    # Generate Report Button
    if st.session_state.final_report is None:
        if st.button("Generate Final Report"):
            if not check_api_key(): st.stop()
            with st.spinner("AI Manager is drafting the final report..."):
                # Prepare context for report generation
                try:
                    results_summary = format_results_markdown(st.session_state.analysis_results) # Use markdown formatter

                    prompt = st.session_state.manager_report_prompt_template.format(
                        project_name=st.session_state.project_name,
                        problem_statement=st.session_state.problem_statement,
                        manager_plan=st.session_state.manager_plan,
                        analyst_summary=st.session_state.analyst_summary,
                        analysis_results_summary=results_summary
                    )

                    report_response = get_gemini_response(prompt, persona="manager", model=st.session_state.gemini_model)

                    if report_response and not report_response.startswith("Error:"):
                        st.session_state.final_report = report_response
                        add_to_conversation("manager", f"Generated Final Report:\n{report_response}")
                        st.success("Final report generated!")
                        st.rerun() # Rerun to display the report
                    else:
                        st.error(f"Failed to generate report: {report_response}")
                        add_to_conversation("system", f"Error generating final report: {report_response}")

                except KeyError as e:
                    st.error(f"Prompt Formatting Error: Missing key {e} in Manager Report Prompt template. Please check the template in sidebar settings.")
                    add_to_conversation("system", f"Error formatting Manager Report prompt: Missing key {e}")
                except Exception as e:
                    st.error(f"An unexpected error occurred during report generation: {e}")
                    add_to_conversation("system", f"Error during report generation: {e}")

    # Display Report and Download
    if st.session_state.final_report:
        st.markdown("### Final Report Draft")
        st.markdown(st.session_state.final_report)

        # Allow download of the report itself
        try:
            st.download_button(
                 label="Download Report (Markdown)",
                 data=st.session_state.final_report,
                 file_name=f"{st.session_state.project_name}_Final_Report.md",
                 mime="text/markdown",
                 key="download_report_md"
            )
        except Exception as e:
            st.error(f"Error creating report download button: {e}")

        # Optional: Convert Markdown to HTML for better preview or other formats?
        try:
            html_report = markdown.markdown(st.session_state.final_report)
            st.download_button(
               label="Download Report (HTML)",
               data=html_report,
               file_name=f"{st.session_state.project_name}_Final_Report.html",
               mime="text/html",
               key="download_report_html"
               )
        except Exception as e:
            st.warning(f"Could not generate HTML download: {e}")


    add_download_buttons("FinalReport")

def display_analysis_execution_step():
    """Displays the Analysis Execution step."""
    st.title("⚙️ 5. AI Analyst - Analysis Execution")
    if not check_api_key(): st.stop()

    if not st.session_state.associate_guidance:
        st.warning("Associate Guidance not available. Please complete Step 4 first.")
        if st.button("Go back to Analysis Guidance"): st.session_state.current_step = 3; st.rerun()
        st.stop()

    st.markdown("### Task Execution")
    st.markdown("Based on the Associate's guidance, select or define a task for the Analyst to execute.")

    # Initialize expander state in session state if it doesn't exist
    if 'show_guidance_expander_state' not in st.session_state:
        st.session_state.show_guidance_expander_state = False # Default to collapsed

    # Display guidance for context, linked to session state
    # Removed 'key' as it's not supported in this Streamlit version
    with st.expander("Show Associate Guidance", expanded=st.session_state.show_guidance_expander_state):
         st.markdown(st.session_state.associate_guidance)
         # Removed explicit state update as it's not needed without 'key'

    # --- TEMPORARY: Display Raw Associate Guidance Markdown ---
    with st.expander("🔍 Raw Associate Guidance Markdown (Temporary for Debugging)"):
        st.text(st.session_state.associate_guidance)
    # --- END TEMPORARY SECTION ---


    # Suggest tasks based on parsing Associate guidance
    suggested_tasks = parse_associate_tasks(st.session_state.associate_guidance)

    # Select or define task
    # Check if 'selected_task_execution' exists, otherwise set default
    if 'selected_task_execution' not in st.session_state:
        st.session_state.selected_task_execution = suggested_tasks[0] if suggested_tasks else "Manually define task below"

    st.session_state.selected_task_execution = st.selectbox(
         "Select suggested task or define manually:",
         options=suggested_tasks + ["Manually define task below"],
         # Try to keep selection, but handle cases where the stored value is not in the current list
         index=(suggested_tasks + ["Manually define task below"]).index(st.session_state.selected_task_execution)
               if st.session_state.selected_task_execution in (suggested_tasks + ["Manually define task below"])
               else (suggested_tasks + ["Manually define task below"]).index("Manually define task below"), # Default to manual if stored value is invalid
         key="task_selector"
    )

    # Text area for the task to be executed
    # Pre-fill based on selection or leave empty for manual entry
    default_task_value = ""
    if st.session_state.selected_task_execution != "Manually define task below":
        default_task_value = st.session_state.selected_task_execution
    elif 'manual_task_input' in st.session_state: # Persist manual input if user switches back and forth
         default_task_value = st.session_state.manual_task_input

    task_to_run = st.text_area(
         "Task for Analyst:",
         value=default_task_value,
         height=100,
         key="task_input_area",
         help="Confirm the task the Analyst should perform using Polars/Plotly. Edit if needed."
         )
    # Store manual input separately if needed
    if st.session_state.selected_task_execution == "Manually define task below":
        st.session_state.manual_task_input = task_to_run

    # --- File/Data Selection ---
    st.markdown("**Select Relevant Data File(s) for this Task:**")
    available_files = list(st.session_state.dataframes.keys())
    if not available_files:
         st.error("No tabular data files loaded for analysis.")
         st.stop()

    # Try to maintain selection if key exists
    default_selection = st.session_state.get('task_file_select', [available_files[0]] if available_files else [])
    # Ensure default selection only contains available files
    default_selection = [f for f in default_selection if f in available_files]
    if not default_selection and available_files: # If previous selection is invalid, default to first file
        default_selection = [available_files[0]]


    selected_files = st.multiselect(
         "File(s):",
         options=available_files,
         default=default_selection,
         key="task_file_select"
    )

    if not selected_files:
         st.warning("Please select at least one data file relevant to the task.")
         # Don't stop here, allow button click but handle missing files later

    # --- Execute Task Button ---
    if st.button("🤖 Generate Analysis Code & Insights", key="execute_task_btn"):
        if not task_to_run:
            st.error("Please define the task for the Analyst.")
        elif not selected_files:
            st.error("Please select at least one data file for the task.")
        elif not check_api_key():
             st.stop() # Already checked, but good practice
        else:
            # --- Prepare Context for Analyst Task Prompt ---
            # Select the *first* selected dataframe for sample and columns
            # Future enhancement: Allow specifying which file for sample/columns if multiple selected
            target_file = selected_files[0]
            df_pl = st.session_state.dataframes.get(target_file)

            if df_pl is None:
                st.error(f"Selected file '{target_file}' not found in loaded data. Please reset or check uploads.")
                st.stop()

            # Get data sample (Polars to JSON) - use write_json for better compatibility
            try:
                # Limit columns in sample if too many? For now, take head(5)
                data_sample_json = json.dumps(df_pl.head(5).to_dicts())
            except Exception as e:
                st.warning(f"Could not generate JSON sample for {target_file}: {e}")
                data_sample_json = json.dumps({'error': f'Could not generate sample: {e}'}) # Ensure valid JSON string

            # Get available columns
            available_columns_str = ", ".join(df_pl.columns)

            # Summarize previous results (simple summary)
            previous_results_summary = "\n".join([f"- Task {i+1}: {res.get('task', 'N/A')[:60]}..." for i, res in enumerate(st.session_state.analysis_results)])
            if not previous_results_summary:
                previous_results_summary = "No previous analysis tasks completed in this session."
            else:
                previous_results_summary = "Summary of Previous Tasks:\n" + previous_results_summary


            # Format the prompt - **CRITICAL POINT FOR THE ERROR**
            try:
                prompt = st.session_state.analyst_task_prompt_template.format(
                    project_name=st.session_state.project_name,
                    problem_statement=st.session_state.problem_statement,
                    previous_results_summary=previous_results_summary,
                    task_to_execute=task_to_run,
                    file_names=", ".join(selected_files), # Use selected files (PLURAL)
                    available_columns=available_columns_str,
                    data_sample=data_sample_json
                )

                # Call LLM
                with st.spinner(f"AI Analyst is working on task: {task_to_run[:50]}..."):
                    analyst_response = get_gemini_response(prompt, persona="analyst", model=st.session_state.gemini_model)

                    if analyst_response and not analyst_response.startswith("Error:"):
                        # --- TEMPORARY: Display Raw LLM Output ---
                        with st.expander("🔍 Raw LLM Output (Temporary for Debugging)"):
                            st.text(analyst_response)
                        # --- END TEMPORARY SECTION ---

                        add_to_conversation("user", f"Requested Analyst Task: {task_to_run} on files: {', '.join(selected_files)}")
                        add_to_conversation("analyst", f"Generated Analysis for Task:\n{analyst_response}")

                        # Parse the response
                        parsed_result = parse_analyst_task_response(analyst_response)

                        # Store the result along with the task and files
                        current_result = {
                            "task": task_to_run,
                            "files": selected_files, # Store which files were selected
                            "approach": parsed_result["approach"],
                            "code": parsed_result["code"],
                            "results_text": parsed_result["results_text"], # LLM's description
                            "insights": parsed_result["insights"]
                        }
                        st.session_state.analysis_results.append(current_result)
                        st.success("Analyst finished task!")
                        # Don't rerun immediately, results are displayed below
                        # We need rerun() if we want the results section to update *instantly* without another interaction

                    else:
                        st.error(f"Failed to get analysis from Analyst: {analyst_response}")
                        add_to_conversation("system", f"Error executing task '{task_to_run}': {analyst_response}")

            except KeyError as e:
                 # ***** THIS IS WHERE THE 'file-name' KeyError WOULD BE CAUGHT *****
                 st.error(f"Prompt Formatting Error: Missing key '{e}' in Analyst Task Prompt template. ")
                 st.error(f"Please check the 'Analyst Task Prompt' in the sidebar settings. It should likely use '{{file_names}}' (plural) instead of '{{file-name}}'.")
                 add_to_conversation("system", f"Error formatting Analyst Task prompt: Missing key {e}")
            except Exception as e:
                 st.error(f"An unexpected error occurred during task execution: {e}")
                 add_to_conversation("system", f"Error during task execution: {e}")


    # --- Display Results ---
    st.markdown("---")
    st.subheader("Analysis Task Results")
    if not st.session_state.analysis_results:
         st.info("No analysis tasks have been executed yet. Click 'Generate Analysis Code & Insights' above.")
    else:
         # Display results of the last task prominently
         last_result = st.session_state.analysis_results[-1]
         st.markdown(f"#### Last Task ({len(st.session_state.analysis_results)}): {last_result.get('task', 'N/A')}")
         st.markdown(f"**Files Used:** {', '.join(last_result.get('files', []))}")

         # Use columns for better layout
         col_app, col_code = st.columns(2)
         with col_app:
             st.markdown("**Approach:**")
             # Display Approach as regular markdown
             st.markdown(last_result.get('approach', '123.Could not parse "Approach" section.'))
             st.markdown("**Key Insights:**")
             # Display Insights as regular markdown
             st.markdown(last_result.get('insights', 'Could not parse "Key Insights" section.'))
         with col_code:
             st.markdown("**Python Code (Polars + Plotly):**")
             # Use st.text_area for editable code
             st.session_state.last_generated_code = last_result.get('code', '# Could not parse "Python Code" section.') # Store code in session state for editing
             edited_code = st.text_area(
                 "Generated Code:",
                 value=st.session_state.last_generated_code,
                 height=400, # Increase height slightly for better visibility
                 key="editable_code_area",
                 help="Edit the code if needed before running it locally. The text area is scrollable for longer code." # Add help text about scrolling
             )
             # Update the stored code if edited
             st.session_state.last_generated_code = edited_code


         st.markdown("**Results (Description from AI):**")
         # Use st.text to preserve formatting of text-based tables from code output
         st.text(last_result.get('results_text', 'Could not parse "Results" section.'))

         # --- Section for External Execution Output ---
         st.markdown("---")
         st.subheader("External Execution Output")
         st.info("Run the generated code in your local Python environment. If the code produces text output, paste it below. If it generates a Plotly figure object named `fig`, the code should include `fig.write_html('plot_output.html')` to save it. Upload the saved plot file below.")

         # Text area for pasting output
         pasted_output = st.text_area(
             "Paste Code Output Here:",
             height=200,
             key="pasted_output_area",
             help="Paste the text output from running the generated code locally."
         )

         # Button to get insights from pasted output
         if st.button("Get Insights from Pasted Output", key="get_insights_from_output_btn"):
             if pasted_output:
                 if not check_api_key(): st.stop()
                 with st.spinner("AI Analyst is generating insights from the output..."):
                     # Prepare prompt for Analyst to get insights from output
                     output_insights_prompt = st.session_state.analyst_task_prompt_template.format(
                         project_name=st.session_state.project_name,
                         problem_statement=st.session_state.problem_statement,
                         previous_results_summary="Context: Analyzing output from a previous task.", # Provide context
                         task_to_execute=f"Analyze the following output based on the original task: {last_result.get('task', 'N/A')}",
                         file_names=", ".join(last_result.get('files', [])), # Use files from the task
                         available_columns="N/A (Analyzing output, not raw data)",
                         data_sample=f"Original Code:\n```python\n{edited_code}\n```\n\nPasted Output:\n```\n{pasted_output}\n```" # Include code and output as sample
                     )
                     # Modify prompt slightly to focus on interpreting output
                     output_insights_prompt += "\n\nBased on the 'Pasted Output' provided, interpret the results and provide key insights related to the original task. Focus on explaining what the output means in the context of the analysis."


                     insights_response = get_gemini_response(output_insights_prompt, persona="analyst", model=st.session_state.gemini_model)

                     if insights_response and not insights_response.startswith("Error:"):
                         # Store the new insights (maybe append or replace a specific insights field?)
                         # For now, let's just display it below the button
                         st.session_state.output_insights = insights_response
                         add_to_conversation("analyst", f"Insights from Pasted Output:\n{insights_response}")
                         st.success("Insights generated!")
                         st.rerun() # Rerun to display insights
                     else:
                         st.error(f"Failed to get insights from output: {insights_response}")
                         add_to_conversation("system", f"Error getting insights from output: {insights_response}")
             else:
                 st.warning("Please paste the code output first.")

         # Display generated insights from output if available
         if st.session_state.get('output_insights'):
             st.markdown("#### Insights from Output")
             st.markdown(st.session_state.output_insights)
             # Clear after displaying
             # del st.session_state.output_insights # Or keep it? Let's keep for now.


         # File uploader for plots
         uploaded_plot_file = st.file_uploader(
             "Upload Saved Plot File (HTML, PNG, JPG):",
             type=["html", "png", "jpg", "jpeg"],
             key="plot_uploader"
         )

         # Display uploaded plot
         if uploaded_plot_file is not None:
             file_extension = os.path.splitext(uploaded_plot_file.name)[1].lower()
             if file_extension == ".html":
                 # Read HTML content and display
                 html_content = uploaded_plot_file.getvalue().decode("utf-8")
                 st.components.v1.html(html_content, height=500, scrolling=True)
             elif file_extension in [".png", ".jpg", ".jpeg"]:
                 # Display image
                 st.image(uploaded_plot_file)
             else:
                 st.warning(f"Unsupported plot file type for display: {file_extension}")

             # Button to get insights from uploaded plot (based on description)
             if st.button("Get Insights from Uploaded Plot", key="get_insights_from_plot_btn"):
                 if not check_api_key(): st.stop()
                 with st.spinner("AI Analyst is generating insights from the plot description..."):
                     # Prepare prompt for Analyst to get insights from plot description
                     # Send original task, code, and AI's original results_text (plot description)
                     plot_insights_prompt = st.session_state.analyst_task_prompt_template.format(
                         project_name=st.session_state.project_name,
                         problem_statement=st.session_state.problem_statement,
                         previous_results_summary="Context: Analyzing a generated plot based on its description.", # Provide context
                         task_to_execute=f"Analyze the plot described below based on the original task: {last_result.get('task', 'N/A')}",
                         file_names=", ".join(last_result.get('files', [])), # Use files from the task
                         available_columns="N/A (Analyzing plot description)",
                         data_sample=f"Original Code:\n```python\n{edited_code}\n```\n\nAI's Description of Plot:\n```\n{last_result.get('results_text', 'N/A')}\n```" # Include code and AI's description
                     )
                     # Modify prompt slightly to focus on interpreting the described plot
                     plot_insights_prompt += "\n\nBased on the 'AI's Description of Plot' provided, interpret the visualization and provide key insights related to the original task. Focus on explaining what the described plot suggests about the data."

                     plot_insights_response = get_gemini_response(plot_insights_prompt, persona="analyst", model=st.session_state.gemini_model)

                     if plot_insights_response and not plot_insights_response.startswith("Error:"):
                         # Store the new insights (maybe append or replace a specific insights field?)
                         # For now, let's just display it below the button
                         st.session_state.plot_insights = plot_insights_response
                         add_to_conversation("analyst", f"Insights from Uploaded Plot Description:\n{plot_insights_response}")
                         st.success("Insights generated!")
                         st.rerun() # Rerun to display insights
                     else:
                         st.error(f"Failed to get insights from plot description: {plot_insights_response}")
                         add_to_conversation("system", f"Error getting insights from plot description: {plot_insights_response}")


         # Display generated insights from plot description if available
         if st.session_state.get('plot_insights'):
             st.markdown("#### Insights from Plot Description")
             st.markdown(st.session_state.plot_insights)
             # Clear after displaying
             # del st.session_state.plot_insights # Or keep it? Let's keep for now.


         # Expander for all previous results
         with st.expander("View All Task Results", expanded=True): # Default to expanded
              # Iterate in reverse to show newest first
              for i, result in enumerate(reversed(st.session_state.analysis_results)):
                   task_num = len(st.session_state.analysis_results) - i
                   # Use a container with a border for each task for better visual separation
                   with st.container(border=True):
                        st.markdown(f"#### Task {task_num}: {result.get('task', 'N/A')}")
                        st.markdown(f"**Files Used:** `{', '.join(result.get('files', []))}`")

                        st.markdown("**Approach:**")
                        # Use st.markdown to correctly render formatted text from the LLM
                        st.markdown(result.get('approach', '_No approach was parsed._'))

                        st.markdown("**Python Code:**")
                        # Use st.code for proper syntax highlighting
                        st.code(result.get('code', '# No code was parsed.'), language='python')

                        st.markdown("**Results Description:**")
                        # Use st.markdown to correctly render formatted text
                        st.markdown(result.get('results_text', '_No results description was parsed._'))

                        st.markdown("**Key Insights:**")
                        # Use st.markdown to correctly render formatted text
                        st.markdown(result.get('insights', '_No insights were parsed._'))
                   st.markdown("---") # Add a separator after each container


    # --- Navigation to Next Step ---
    st.markdown("---")
    col1, col2 = st.columns([1, 4])
    with col1:
         # Add Associate Review Button (placeholder action)
         # if st.button("Review with Associate", key="review_task_btn"):
         #     st.info("Associate Review Feature Placeholder")

         if st.button("Next: Final Report"):
              if not st.session_state.analysis_results:
                   st.warning("Please execute at least one analysis task before generating the final report.")
              else:
                   st.session_state.current_step = 5 # Index 5 = Step 6
                   st.rerun()

    add_download_buttons("FinalReport")

def display_analysis_guidance_step():
    """Displays the Analysis Guidance step."""
    st.title("🔍 4. AI Associate - Analysis Guidance")
    if not check_api_key(): st.stop()

    # Generate guidance if not exists
    if st.session_state.associate_guidance is None:
         if not st.session_state.analyst_summary:
              st.warning("Analyst Summary not available. Please complete Step 3 first.")
              if st.button("Go back to Data Understanding"): st.session_state.current_step = 2; st.rerun()
              st.stop()
         elif not st.session_state.manager_plan:
               st.warning("Manager Plan not available. Please complete Step 2 first.")
               if st.button("Go back to Manager Planning"): st.session_state.current_step = 1; st.rerun()
               st.stop()
         else:
              with st.spinner("AI Associate is generating guidance and next steps..."):
                    try:
                        prompt = st.session_state.associate_prompt_template.format(
                             problem_statement=st.session_state.problem_statement,
                             manager_plan=st.session_state.manager_plan,
                             analyst_summary=st.session_state.analyst_summary
                        )
                        assoc_response = get_gemini_response(prompt, persona="associate", model=st.session_state.gemini_model)
                        if assoc_response and not assoc_response.startswith("Error:"):
                             st.session_state.associate_guidance = assoc_response
                             add_to_conversation("associate", f"Generated Analysis Guidance:\n{assoc_response}")
                             st.rerun()
                        else:
                             st.error(f"Failed to get guidance: {assoc_response}")
                             add_to_conversation("system", f"Error getting Associate guidance: {assoc_response}")
                    except KeyError as e:
                        st.error(f"Prompt Formatting Error: Missing key {e} in Associate Guidance Prompt template. Please check the template in sidebar settings.")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")


    # Display guidance
    if st.session_state.associate_guidance:
        st.markdown("### Analysis Guidance & Next Tasks")
        st.markdown(st.session_state.associate_guidance)

        # --- Consultation/Review Section ---
        with st.expander("💬 Consult with AI Persona"):
            st.markdown("Select an AI persona to consult with regarding the analysis guidance.")

            persona_options = ["Manager", "Analyst", "Associate", "Reviewer"]
            selected_consult_persona = st.selectbox(
                "Select Persona:",
                options=persona_options,
                key="consult_persona_select_analysis_guidance"
            )

            consultation_request = st.text_area(f"Your message to the {selected_consult_persona}:", key="consult_request_analysis_guidance")

            if st.button(f"Send to {selected_consult_persona}", key="consult_button_analysis_guidance"):
                if consultation_request:
                    if not check_api_key(): st.stop()

                    # Determine which prompt template and persona name to use
                    persona_key = selected_consult_persona.lower()
                    prompt_template_key = f"{persona_key}_prompt_template"

                    # Special case for Reviewer, use the specific reviewer template
                    if persona_key == "reviewer":
                         prompt_template_key = "reviewer_prompt_template"
                         # For reviewer, format the template with specific context
                         project_artifacts = f"Associate's Guidance:\n{st.session_state.associate_guidance}"
                         project_artifacts = escape_curly_braces(project_artifacts)
                         consult_prompt = st.session_state[prompt_template_key].format(
                             project_name=st.session_state.project_name,
                             problem_statement=st.session_state.problem_statement,
                             current_stage="Analysis Guidance",
                             project_artifacts=project_artifacts,
                             specific_request=consultation_request
                         )
                    else:
                         # For other personas, use a generic consultation format
                         try:
                             generic_consult_wrapper = f"""
                             You are the AI {selected_consult_persona}. A user is consulting with you about the current analysis guidance.

                             **Associate's Guidance:**
                             {st.session_state.associate_guidance}

                             **User's Question/Request:**
                             {consultation_request}

                             **Your Task:** Respond to the user's request based on your persona's expertise regarding the analysis guidance.
                             """
                             consult_prompt = generic_consult_wrapper

                         except KeyError as e:
                             st.error(f"Error preparing prompt for {selected_consult_persona}: Missing key {e}. Template might not support this type of consultation.")
                             add_to_conversation("system", f"Error preparing consultation prompt for {selected_consult_persona}: Missing key {e}")
                             st.stop()
                         except Exception as e:
                             st.error(f"An unexpected error occurred preparing prompt for {selected_consult_persona}: {e}")
                             add_to_conversation("system", f"Error preparing consultation prompt for {selected_consult_persona}: {e}")
                             st.stop()


                    with st.spinner(f"Consulting with AI {selected_consult_persona}..."):
                        consult_response = get_gemini_response(consult_prompt, persona=persona_key, model=st.session_state.gemini_model)

                        if consult_response and not consult_response.startswith("Error:"):
                            st.session_state.consultation_response = consult_response
                            st.session_state.consultation_persona = selected_consult_persona # Store persona for display
                            add_to_conversation(persona_key, f"Consultation Request ({selected_consult_persona}): {consultation_request}\n\nResponse:\n{consult_response}")
                            st.rerun() # Rerun to display the response
                        elif consult_response:
                            st.error(f"{selected_consult_persona} Error: {consult_response}")
                            add_to_conversation("system", f"Error getting consultation response from {selected_consult_persona}: {consult_response}")
                        else:
                            st.error(f"Failed to get response from {selected_consult_persona}.")
                            add_to_conversation("system", f"Failed to get consultation response from {selected_consult_persona}.")
                else:
                    st.warning("Please enter a message for the consultation.")

        # Display consultation response if available
        if st.session_state.get('consultation_response'):
            st.markdown(f"#### 💬 AI {st.session_state.get('consultation_persona', 'Persona')}'s Response")
            st.markdown(st.session_state.consultation_response)
            # Clear after displaying once
            # del st.session_state.consultation_response
            # del st.session_state.consultation_persona
        # --- End Consultation/Review Section ---

        col1, col2 = st.columns([1,4])
        with col1:
            if st.button("Next: Analysis Execution"):
                st.session_state.current_step = 4 # Index 4 is Step 5
                st.rerun()

        add_download_buttons("AnalysisGuidance")

def display_data_understanding_step():
    """Displays the Data Understanding step."""
    st.title("📊 3. AI Analyst - Data Understanding")
    if not check_api_key(): st.stop()

    # Generate summary if not exists
    if st.session_state.analyst_summary is None:
         if not st.session_state.manager_plan:
              st.warning("Manager Plan not available. Please complete Step 2 first.")
              if st.button("Go back to Manager Planning"): st.session_state.current_step = 1; st.rerun()
              st.stop()
         else:
              with st.spinner("AI Analyst is examining data profiles..."):
                    # Generate combined profile summary
                    all_profiles_summary = ""
                    for file_name, profile in st.session_state.data_profiles.items():
                        try:
                            profile_summary = generate_data_profile_summary(profile) # From utils
                            all_profiles_summary += f"\n## Profile: {file_name}\n{profile_summary}\n"
                        except Exception as e:
                             all_profiles_summary += f"\n## Profile: {file_name}\nError generating summary: {e}\n"
                             st.warning(f"Could not generate profile summary for {file_name}: {e}")

                    for file_name, text in st.session_state.data_texts.items():
                        text_snippet = text[:200] + "..." if len(text) > 200 else text
                        all_profiles_summary += f"\n## Text Document: {file_name}\nSnippet: {text_snippet}\n" # Include text snippets

                    if not all_profiles_summary.strip():
                         all_profiles_summary = "No detailed data profiles or text snippets available."
                         st.warning("No data profiles or text content found to provide to Analyst.")

                    try:
                        prompt = st.session_state.analyst_prompt_template.format(
                             problem_statement=st.session_state.problem_statement,
                             manager_plan=st.session_state.manager_plan,
                             data_profiles_summary=all_profiles_summary
                        )
                        analyst_response = get_gemini_response(prompt, persona="analyst", model=st.session_state.gemini_model)
                        if analyst_response and not analyst_response.startswith("Error:"):
                             st.session_state.analyst_summary = analyst_response
                             add_to_conversation("analyst", f"Generated Data Summary:\n{analyst_response}")
                             st.rerun()
                        else:
                             st.error(f"Failed to get data summary: {analyst_response}")
                             add_to_conversation("system", f"Error getting Analyst summary: {analyst_response}")
                    except KeyError as e:
                        st.error(f"Prompt Formatting Error: Missing key {e} in Analyst Summary Prompt template. Please check the template in sidebar settings.")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")

    # Display summary and data details
    if st.session_state.analyst_summary:
        # Display data profiles expander
        with st.expander("View Data Details", expanded=True): # Expanded by default for better visibility
            if st.session_state.get('dataframes'):
                st.markdown("#### Tabular Data Profiles")
                for file_name, df in st.session_state.dataframes.items():
                    st.subheader(f"File: {file_name}")
                    st.dataframe(df.head(10)) # Display head as Pandas for better Streamlit rendering

                    profile = st.session_state.data_profiles.get(file_name)
                    if profile and profile.get("file_type") == "tabular":
                        st.write(f"**Dimensions:** {profile.get('shape', ('N/A', 'N/A'))}")
                        st.write(f"**Columns:** {', '.join(profile.get('columns', []))}")

                        st.markdown("##### Data Types:")
                        dtypes_dict = profile.get('dtypes', {})
                        if dtypes_dict:
                            for col, dtype in dtypes_dict.items():
                                st.write(f"- **{col}**: {dtype}")
                        else:
                            st.write("Data type information not available.")

                        # Display missing summary
                        missing_summary = profile.get('missing_summary')
                        st.markdown("##### Missing Values:")
                        if isinstance(missing_summary, pl.DataFrame) and not missing_summary.is_empty():
                            st.dataframe(missing_summary.to_pandas()) # Display as Pandas
                        elif isinstance(missing_summary, pd.DataFrame) and not missing_summary.empty: # Handle pandas case if profile generated it
                             st.dataframe(missing_summary)
                        elif missing_summary is None:
                             st.write("Missing value information not generated in profile.")
                        else:
                            st.write("No missing values detected.")

                        # Display descriptive statistics
                        describe_df = profile.get('numeric_summary')
                        st.markdown("##### Descriptive Statistics:")
                        if isinstance(describe_df, pl.DataFrame) and not describe_df.is_empty():
                             st.dataframe(describe_df.to_pandas()) # Display as Pandas
                        elif isinstance(describe_df, pd.DataFrame) and not describe_df.empty: # Handle pandas case
                             st.dataframe(describe_df)
                        elif describe_df is None:
                             st.write("Descriptive statistics not generated in profile.")
                        else:
                             st.write("Descriptive statistics not available.")

                    else:
                         st.write("Detailed tabular profile not available.")
                    st.markdown("---")

            if st.session_state.get('data_texts'):
                 st.markdown("#### Text Document Snippets")
                 for file_name, text_content in st.session_state.data_texts.items():
                      st.subheader(f"Text Document: {file_name}")
                      text_snippet = text_content[:1000] + "..." if len(text_content) > 1000 else text_content
                      st.text_area("Content Snippet", text_snippet, height=150, disabled=True, key=f"text_snippet_{file_name}")
                      profile = st.session_state.data_profiles.get(file_name)
                      if profile and profile.get("file_type") in ["docx", "pdf"]:
                           st.write(f"**Extracted Text Length:** {profile.get('text_length', 'N/A')} characters")
                      st.markdown("---")

        st.markdown("### Data Summary & Assessment")
        # Display the Analyst's narrative summary as Markdown
        st.markdown(st.session_state.analyst_summary)

        # --- Consultation/Review Section ---
        with st.expander("💬 Consult with AI Persona"):
            st.markdown("Select an AI persona to consult with regarding the data understanding.")

            persona_options = ["Manager", "Analyst", "Associate", "Reviewer"]
            selected_consult_persona = st.selectbox(
                "Select Persona:",
                options=persona_options,
                key="consult_persona_select_data_understanding"
            )

            consultation_request = st.text_area(f"Your message to the {selected_consult_persona}:", key="consult_request_data_understanding")

            if st.button(f"Send to {selected_consult_persona}", key="consult_button_data_understanding"):
                if consultation_request:
                    if not check_api_key(): st.stop()

                    # Determine which prompt template and persona name to use
                    persona_key = selected_consult_persona.lower()
                    prompt_template_key = f"{persona_key}_prompt_template"

                    # Special case for Reviewer, use the specific reviewer template
                    if persona_key == "reviewer":
                         prompt_template_key = "reviewer_prompt_template"
                         # For reviewer, format the template with specific context
                         project_artifacts = f"Analyst's Data Summary:\n{st.session_state.analyst_summary}"
                         project_artifacts = escape_curly_braces(project_artifacts)
                         consult_prompt = st.session_state[prompt_template_key].format(
                             project_name=st.session_state.project_name,
                             problem_statement=st.session_state.problem_statement,
                             current_stage="Data Understanding",
                             project_artifacts=project_artifacts,
                             specific_request=consultation_request
                         )
                    else:
                         # For other personas, use a generic consultation format
                         try:
                             generic_consult_wrapper = f"""
                             You are the AI {selected_consult_persona}. A user is consulting with you about the current data understanding.

                             **Analyst's Data Summary:**
                             {st.session_state.analyst_summary}

                             **User's Question/Request:**
                             {consultation_request}

                             **Your Task:** Respond to the user's request based on your persona's expertise regarding the data understanding.
                             """
                             consult_prompt = generic_consult_wrapper

                         except KeyError as e:
                             st.error(f"Error preparing prompt for {selected_consult_persona}: Missing key {e}. Template might not support this type of consultation.")
                             add_to_conversation("system", f"Error preparing consultation prompt for {selected_consult_persona}: Missing key {e}")
                             st.stop()
                         except Exception as e:
                             st.error(f"An unexpected error occurred preparing prompt for {selected_consult_persona}: {e}")
                             add_to_conversation("system", f"Error preparing consultation prompt for {selected_consult_persona}: {e}")
                             st.stop()


                    with st.spinner(f"Consulting with AI {selected_consult_persona}..."):
                        consult_response = get_gemini_response(consult_prompt, persona=persona_key, model=st.session_state.gemini_model)

                        if consult_response and not consult_response.startswith("Error:"):
                            st.session_state.consultation_response = consult_response
                            st.session_state.consultation_persona = selected_consult_persona # Store persona for display
                            add_to_conversation(persona_key, f"Consultation Request ({selected_consult_persona}): {consultation_request}\n\nResponse:\n{consult_response}")
                            st.rerun() # Rerun to display the response
                        elif consult_response:
                            st.error(f"{selected_consult_persona} Error: {consult_response}")
                            add_to_conversation("system", f"Error getting consultation response from {selected_consult_persona}: {consult_response}")
                        else:
                            st.error(f"Failed to get response from {selected_consult_persona}.")
                            add_to_conversation("system", f"Failed to get consultation response from {selected_consult_persona}.")
                else:
                    st.warning("Please enter a message for the consultation.")

        # Display consultation response if available
        if st.session_state.get('consultation_response'):
            st.markdown(f"#### 💬 AI {st.session_state.get('consultation_persona', 'Persona')}'s Response")
            st.markdown(st.session_state.consultation_response)
            # Clear after displaying once
            # del st.session_state.consultation_response
            # del st.session_state.consultation_persona
        # --- End Consultation/Review Section ---

        col1, col2 = st.columns([1,4])
        with col1:
            if st.button("Next: Analysis Guidance"):
                st.session_state.current_step = 3
                st.rerun()

        add_download_buttons("DataUnderstanding")

def display_manager_planning_step():
    """Displays the Manager Planning step."""
    st.title("👨‍💼 2. AI Manager - Analysis Planning")

    if not check_api_key(): st.stop()

    # Generate plan if not exists
    if st.session_state.manager_plan is None:
        with st.spinner("AI Manager is generating the analysis plan..."):
            # Prepare context for Manager
            file_info = ""
            for file_name, profile in st.session_state.data_profiles.items():
                file_info += f"\nFile: {file_name}\n"
                if profile: # Check if profile exists
                    file_info += f"- Columns: {profile.get('columns', 'N/A')}\n"
                    file_info += f"- Shape: {profile.get('shape', 'N/A')}\n"
                else:
                    file_info += "- Profile: Not available (check file processing)\n"
            for file_name, text in st.session_state.data_texts.items():
                text_snippet = text[:100] + "..." if len(text) > 100 else text
                file_info += f"\nFile: {file_name}\n- Type: Text Document\n- Snippet: {text_snippet}\n" # Show snippet

            # Use the editable prompt template
            try:
                prompt = st.session_state.manager_prompt_template.format(
                    project_name=st.session_state.project_name,
                    problem_statement=st.session_state.problem_statement,
                    data_context=st.session_state.data_context,
                    file_info=file_info if file_info else "No data files loaded."
                )
                manager_response = get_gemini_response(prompt, persona="manager", model=st.session_state.gemini_model)
                if manager_response and not manager_response.startswith("Error:"):
                    st.session_state.manager_plan = manager_response
                    add_to_conversation("manager", f"Generated Analysis Plan:\n{manager_response}")
                    st.rerun() # Rerun to display the plan
                else:
                    st.error(f"Failed to get plan from Manager: {manager_response}")
                    add_to_conversation("system", f"Error getting Manager plan: {manager_response}")
            except KeyError as e:
                st.error(f"Prompt Formatting Error: Missing key {e} in Manager Prompt template. Please check the template in sidebar settings.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")


    # Display plan and interaction options
    if st.session_state.manager_plan:
        st.markdown("### Analysis Plan")
        st.markdown(st.session_state.manager_plan)

        # --- Feedback Expander ---
        with st.expander("Provide Feedback to Manager"):
             feedback = st.text_area("Your feedback on the plan:", key="manager_feedback_input")
             if st.button("Send Feedback", key="manager_feedback_btn"):
                  if feedback:
                       if not check_api_key(): st.stop()
                       add_to_conversation("user", f"Feedback on Manager Plan: {feedback}")
                       with st.spinner("Manager is revising plan..."):
                            # Create revision prompt using the Manager's persona prompt structure
                            revision_prompt = f"""
                            **Original Context:**
                            Project Name: {st.session_state.project_name}
                            Problem Statement: {st.session_state.problem_statement}
                            Data Context: {st.session_state.data_context}
                            Available Data Files: [Details omitted for brevity, assume original context available]

                            **Original Plan:**
                            {st.session_state.manager_plan}

                            **User Feedback:**
                            {feedback}

                            **Your Task (as AI Manager):**
                            Revise the original analysis plan based ONLY on the user feedback provided. Maintain the structured, step-by-step format. Output only the revised plan.
                            """
                            try:
                                revised_plan = get_gemini_response(revision_prompt, persona="manager", model=st.session_state.gemini_model)
                                if revised_plan and not revised_plan.startswith("Error:"):
                                     st.session_state.manager_plan = revised_plan
                                     add_to_conversation("manager", f"Revised Plan based on feedback:\n{revised_plan}")
                                     st.success("Plan updated!")
                                     st.rerun()
                                else:
                                     st.error(f"Failed to revise plan: {revised_plan}")
                                     add_to_conversation("system", f"Error revising Manager plan: {revised_plan}")
                            except Exception as e:
                                st.error(f"An error occurred during plan revision: {e}")
                  else:
                       st.warning("Please enter feedback.")

        # --- Consultation/Review Section ---
        with st.expander("💬 Consult with AI Persona"):
            st.markdown("Select an AI persona to consult with regarding the current plan.")

            persona_options = ["Manager", "Analyst", "Associate", "Reviewer"]
            selected_consult_persona = st.selectbox(
                "Select Persona:",
                options=persona_options,
                key="consult_persona_select_manager_planning"
            )

            consultation_request = st.text_area(f"Your message to the {selected_consult_persona}:", key="consult_request_manager_planning")

            if st.button(f"Send to {selected_consult_persona}", key="consult_button_manager_planning"):
                if consultation_request:
                    if not check_api_key(): st.stop()

                    # Determine which prompt template and persona name to use
                    persona_key = selected_consult_persona.lower()
                    prompt_template_key = f"{persona_key}_prompt_template"

                    # Special case for Reviewer, use the specific reviewer template
                    if persona_key == "reviewer":
                         prompt_template_key = "reviewer_prompt_template"
                         # For reviewer, format the template with specific context
                         project_artifacts = f"Current Analysis Plan:\n{st.session_state.manager_plan}"
                         project_artifacts = escape_curly_braces(project_artifacts)
                         consult_prompt = st.session_state[prompt_template_key].format(
                             project_name=st.session_state.project_name,
                             problem_statement=st.session_state.problem_statement,
                             current_stage="Manager Planning",
                             project_artifacts=project_artifacts,
                             specific_request=consultation_request
                         )
                    else:
                         # For other personas, use a generic consultation format
                         # This might need refinement based on how each persona should respond to arbitrary questions
                         # For now, format using their main template if possible, or a generic wrapper
                         try:
                             # Attempt to use the persona's main template, providing relevant context
                             # This assumes the template can handle additional context like a specific question
                             # This might fail if the template expects specific keys not available here.
                             # A more robust approach might be a dedicated 'consultation_prompt_template' per persona.
                             # For now, let's try a generic wrapper + their main template content
                             generic_consult_wrapper = f"""
                             You are the AI {selected_consult_persona}. A user is consulting with you about the current analysis plan.

                             **Current Analysis Plan:**
                             {st.session_state.manager_plan}

                             **User's Question/Request:**
                             {consultation_request}

                             **Your Task:** Respond to the user's request based on your persona's expertise regarding the plan.
                             """
                             consult_prompt = generic_consult_wrapper # Use the wrapper for now

                         except KeyError as e:
                             st.error(f"Error preparing prompt for {selected_consult_persona}: Missing key {e}. Template might not support this type of consultation.")
                             add_to_conversation("system", f"Error preparing consultation prompt for {selected_consult_persona}: Missing key {e}")
                             st.stop() # Stop execution on prompt error
                         except Exception as e:
                             st.error(f"An unexpected error occurred preparing prompt for {selected_consult_persona}: {e}")
                             add_to_conversation("system", f"Error preparing consultation prompt for {selected_consult_persona}: {e}")
                             st.stop() # Stop execution on prompt error


                    with st.spinner(f"Consulting with AI {selected_consult_persona}..."):
                        consult_response = get_gemini_response(consult_prompt, persona=persona_key, model=st.session_state.gemini_model)

                        if consult_response and not consult_response.startswith("Error:"):
                            st.session_state.consultation_response = consult_response
                            st.session_state.consultation_persona = selected_consult_persona # Store persona for display
                            add_to_conversation(persona_key, f"Consultation Request ({selected_consult_persona}): {consultation_request}\n\nResponse:\n{consult_response}")
                            st.rerun() # Rerun to display the response
                        elif consult_response:
                            st.error(f"{selected_consult_persona} Error: {consult_response}")
                            add_to_conversation("system", f"Error getting consultation response from {selected_consult_persona}: {consult_response}")
                        else:
                            st.error(f"Failed to get response from {selected_consult_persona}.")
                            add_to_conversation("system", f"Failed to get consultation response from {selected_consult_persona}.")
                else:
                    st.warning("Please enter a message for the consultation.")

        # Display consultation response if available
        if st.session_state.get('consultation_response'):
            st.markdown(f"#### 💬 AI {st.session_state.get('consultation_persona', 'Persona')}'s Response")
            st.markdown(st.session_state.consultation_response)
            # Clear after displaying once
            # del st.session_state.consultation_response
            # del st.session_state.consultation_persona
        # --- End Consultation/Review Section ---


        col1, col2 = st.columns([1,4])
        with col1:
            if st.button("Next: Data Understanding"):
                st.session_state.current_step = 2
                st.rerun()

        add_download_buttons("ManagerPlanning")

def display_setup_step():
    """Displays the Project Setup step."""
    st.title("🚀 Start New Analysis Project")
    with st.form("project_setup_form"):
        st.subheader("1. Project Details")
        project_name = st.text_input("Project Name", st.session_state.get("project_name", "Analysis Project"))
        problem_statement = st.text_area("Problem Statement / Goal", st.session_state.get("problem_statement", ""), placeholder="Describe what you want to achieve...")
        data_context = st.text_area("Data Context (Optional)", st.session_state.get("data_context", ""), placeholder="Background info about the data...")

        st.subheader("2. Upload Data")
        uploaded_files = st.file_uploader(
            "Upload CSV, XLSX, DOCX, or PDF files",
            type=["csv", "xlsx", "docx", "pdf"],
            accept_multiple_files=True,
            key="file_uploader"
        )

        submit_button = st.form_submit_button("🚀 Start Analysis")

        if submit_button:
            if not st.session_state.gemini_api_key:
                 st.error("Please enter your Gemini API Key in the sidebar first!")
            elif not project_name or not problem_statement or not uploaded_files:
                st.error("Project Name, Problem Statement, and at least one Data File are required.")
            else:
                # Reset previous project data if any
                st.session_state.dataframes = {}
                st.session_state.data_profiles = {}
                st.session_state.data_texts = {}
                st.session_state.analysis_results = []
                st.session_state.conversation_history = []
                # Clear previous AI outputs
                st.session_state.manager_plan = None
                st.session_state.analyst_summary = None
                st.session_state.associate_guidance = None
                st.session_state.final_report = None

                with st.spinner("Processing uploaded files..."):
                    success_count = 0
                    error_messages = []
                    for uploaded_file in uploaded_files:
                        try:
                            # Use Polars for CSV/Excel directly
                            df, profile, text_content = process_uploaded_file(uploaded_file)
                            if df is not None: # df is now a Polars DataFrame
                                st.session_state.dataframes[uploaded_file.name] = df
                                st.session_state.data_profiles[uploaded_file.name] = profile
                                success_count += 1
                            if text_content:
                                st.session_state.data_texts[uploaded_file.name] = text_content
                                # Only count as success if no dataframe was extracted for this file
                                if df is None:
                                    success_count +=1
                        except Exception as e:
                            error_messages.append(f"Error processing {uploaded_file.name}: {e}")

                    if error_messages:
                        for msg in error_messages:
                            st.error(msg)

                    if success_count > 0:
                        st.session_state.data_uploaded = True
                        st.session_state.project_initialized = True
                        st.session_state.current_step = 1 # Move to Manager Planning
                        st.session_state.project_name = project_name
                        st.session_state.problem_statement = problem_statement
                        st.session_state.data_context = data_context

                        # Add initial info to conversation
                        file_summary = "Uploaded Files:\n"
                        for name in st.session_state.dataframes.keys(): file_summary += f"- Tabular: {name} ({st.session_state.dataframes[name].height} rows, {st.session_state.dataframes[name].width} cols)\n" # Use Polars attributes
                        for name in st.session_state.data_texts.keys(): file_summary += f"- Text: {name}\n"
                        init_msg = f"Project: {project_name}\nProblem: {problem_statement}\nContext: {data_context}\n\n{file_summary}"
                        add_to_conversation("user", init_msg)

                        st.success("Project initialized!")
                        st.rerun()
                    else:
                        st.error("No usable data or text content could be extracted. Please check file formats or content.")

    # Display summary after form submission or on reload if initialized
    if st.session_state.project_initialized:
        st.subheader("Project Summary")
        st.write(f"**Project Name:** {st.session_state.project_name}")
        st.write(f"**Problem Statement:** {st.session_state.problem_statement}")
        if st.session_state.data_context: st.write(f"**Data Context:** {st.session_state.data_context}")

        st.subheader("Uploaded Data Summary")
        if st.session_state.dataframes:
            for name, df in st.session_state.dataframes.items(): st.write(f"- Tabular: {name} ({df.height} rows, {df.width} cols)") # Use Polars attributes
        if st.session_state.data_texts:
            for name in st.session_state.data_texts.keys(): st.write(f"- Text Document: {name}")

        col1, col2 = st.columns([1,4])
        with col1:
            if st.button("Next: Manager Planning"):
                st.session_state.current_step = 1
                st.rerun()
        add_download_buttons("Setup")

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Data Analysis Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Configure Gemini API ---
# Initial check, actual calls use session state key
# Ensure API key is available before configuring
if st.session_state.get("gemini_api_key") or os.getenv("GEMINI_API_KEY"):
    configure_genai(api_key=st.session_state.get("gemini_api_key", os.getenv("GEMINI_API_KEY")))
else:
    # Display a warning or placeholder if the key is not yet set,
    # but allow the app to load the initial UI elements.
    pass # Configuration will happen properly once the key is entered in the sidebar


# --- Initialize Session State Variables ---
# (Initialize only if they don't exist)
defaults = {
    'project_initialized': False,
    'current_step': 0,
    'data_uploaded': False,
    'dataframes': {},        # Stores Polars DataFrames {filename: pl.DataFrame}
    'data_profiles': {},     # Stores basic profiles {filename: profile_dict}
    'data_texts': {},        # Stores text from non-tabular files {filename: text_string}
    'project_name': "Default Project",
    'problem_statement': "",
    'data_context': "",
    'manager_plan': None,
    'analyst_summary': None,
    'associate_guidance': None,
    'analysis_results': [],  # List to store dicts: [{"task": ..., "approach": ..., "code": ..., "results_text": ..., "insights": ...}]
    'final_report': None,
    'conversation_history': [], # List of {"role": ..., "content": ...}
    'consultation_response': None,
    'consultation_persona': None,
    'reviewer_response': None,
    'reviewer_specific_request': None,
    'gemini_api_key': os.getenv("GEMINI_API_KEY", ""), # Load from env var if available
    'gemini_model': "gemini-2.5-flash", # Updated default model
    'library_management': "Manual", # New setting: Manual / Automated
    # --- Prompt Templates --- (Load from prompts.py)
    'manager_prompt_template': MANAGER_PROMPT_TEMPLATE,
    'analyst_prompt_template': ANALYST_PROMPT_TEMPLATE,
    'associate_prompt_template': ASSOCIATE_PROMPT_TEMPLATE,
    'analyst_task_prompt_template': ANALYST_TASK_PROMPT_TEMPLATE,
    'associate_review_prompt_template': ASSOCIATE_REVIEW_PROMPT_TEMPLATE,
    'manager_report_prompt_template': MANAGER_REPORT_PROMPT_TEMPLATE,
    'reviewer_prompt_template': REVIEWER_PROMPT_TEMPLATE
}

# Apply defaults
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Helper Functions (Centralized ones remain, others moved) ---

# --- End Helper Functions ---


# Import feature functions

# --- Main Application Logic ---
def main():
    project_manager = ProjectManager()
    help_system = HelpSystem()
    user_prefs = UserPreferences()
    dashboard = AnalysisDashboard()
    # --- Sidebar ---
    with st.sidebar:
        st.title("📊 AI Analysis Assistant")
        st.markdown("---")

        # API Key Check & Configuration
        st.session_state.gemini_api_key = st.text_input(
            "Gemini API Key",
            value=st.session_state.gemini_api_key,
            type="password",
            help="Enter your Google Gemini API key."
        )
        # Configure API only if key is present
        if st.session_state.gemini_api_key:
             try:
                 configure_genai(api_key=st.session_state.gemini_api_key)
                 # st.success("Gemini API Configured.") # Optional: feedback
             except Exception as e:
                 st.error(f"API Key Error: {e}. Please check your key.")
                 # Don't stop here, allow UI interaction but show error
        elif not st.session_state.project_initialized:
             # Allow initial setup screen without key, but show warning
             st.warning("Please enter your Gemini API Key in the sidebar to start.")


        st.subheader("Navigation")
        if st.session_state.project_initialized:
            step_options = [
                "1. Project Setup",
                "2. Manager Planning",
                "3. Data Understanding",
                "4. Analysis Guidance",
                "5. Analysis Execution", # Step index 4
                "6. Final Report",     # Step index 5
            ]
            # Ensure current step is valid index
            current_idx = st.session_state.current_step if 0 <= st.session_state.current_step < len(step_options) else 0

            # Use st.radio for clearer step indication
            selected_step_label = st.radio(
                "Current Step:",
                step_options,
                index=current_idx,
                key="step_navigation_radio"
                )
            new_step_index = step_options.index(selected_step_label)
            if new_step_index != st.session_state.current_step:
                st.session_state.current_step = new_step_index
                st.rerun() # Rerun when step changes

            if st.button("💾 Save Project"):
                project_manager.save_state(st.session_state.project_name)
            if st.button("🔄 Reset Project"):
                reset_session()
                st.rerun()
        else:
            st.info("Start a new project to enable navigation.")

        st.markdown("---")
        st.markdown("### AI Team")
        st.markdown("🧠 **Manager**: Plans & Reports")
        st.markdown("📊 **Analyst**: Examines Data & Executes Tasks")
        st.markdown("🔍 **Associate**: Guides Execution & Reviews")
        st.markdown("⭐ **Reviewer**: Strategic Oversight")

        st.markdown("---")
        st.subheader("Settings")

        theme_option = st.selectbox(
            "Theme",
            ["light", "dark"],
            index=0 if user_prefs.theme == "light" else 1,
            key="theme_select",
        )
        if theme_option != user_prefs.theme:
            user_prefs.theme = theme_option
            user_prefs.save()

        # Library Management Choice
        st.session_state.library_management = st.radio(
            "Python Library Management",
            options=["Manual", "Automated (Experimental)"],
            index=0 if st.session_state.library_management == "Manual" else 1,
            key="library_management_radio",
            help="Manual: You must install libraries via requirements.txt. Automated: Attempts to install libraries via pip (Requires permissions, use with caution - NOT YET IMPLEMENTED)."
        )
        if st.session_state.library_management == "Automated (Experimental)":
            st.warning("Automated installation is experimental and not yet functional. Please use Manual for now.", icon="⚠️")


        # Model Selection
        model_options = [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
             "Custom"
        ]
        # Ensure current model is in options, or set to custom
        if st.session_state.gemini_model in model_options and st.session_state.gemini_model != "Custom":
             current_model_idx = model_options.index(st.session_state.gemini_model)
             current_custom_value = ""
        else: # Either it's custom or not in the list
             current_model_idx = model_options.index("Custom")
             current_custom_value = st.session_state.gemini_model # Keep the current custom value


        selected_model_option = st.selectbox(
            "Select Gemini Model",
            model_options,
            index=current_model_idx,
            key="model_select"
        )
        if selected_model_option == "Custom":
            st.session_state.gemini_model = st.text_input(
                "Enter Custom Model Name",
                value=current_custom_value, # Use the stored custom value if Custom is selected
                key="custom_model_input"
            )
        else:
            st.session_state.gemini_model = selected_model_option


        with st.expander("Edit Persona Prompts"):
             # These will be loaded from prompts.py, but can be edited here
             st.session_state.manager_prompt_template = st.text_area("Manager Prompt", value=st.session_state.manager_prompt_template, height=150, key="manager_prompt_edit")
             st.session_state.analyst_prompt_template = st.text_area("Analyst Summary Prompt", value=st.session_state.analyst_prompt_template, height=150, key="analyst_summary_prompt_edit")
             st.session_state.associate_prompt_template = st.text_area("Associate Guidance Prompt", value=st.session_state.associate_prompt_template, height=150, key="associate_guidance_prompt_edit")
             st.session_state.analyst_task_prompt_template = st.text_area("Analyst Task Prompt", value=st.session_state.analyst_task_prompt_template, height=150, key="analyst_task_prompt_edit")
             st.session_state.associate_review_prompt_template = st.text_area("Associate Review Prompt", value=st.session_state.associate_review_prompt_template, height=150, key="associate_review_prompt_edit")
             st.session_state.manager_report_prompt_template = st.text_area("Manager Report Prompt", value=st.session_state.manager_report_prompt_template, height=150, key="manager_report_prompt_edit")
        st.session_state.reviewer_prompt_template = st.text_area("Reviewer Prompt", value=st.session_state.reviewer_prompt_template, height=150, key="reviewer_prompt_edit")

        with st.expander("Crash Log"):
            log_text = read_log()
            if log_text:
                st.text_area("Latest Crash Details", log_text, height=150)
                st.download_button("Download Crash Log", log_text, "crash_log.txt")
                if st.button("Clear Crash Log"):
                    clear_log()
                    st.rerun()
            else:
                st.write("No crash log found.")


    # --- Main Content Area ---
    if not st.session_state.gemini_api_key and not st.session_state.project_initialized:
         st.error("Please enter your Gemini API Key in the sidebar to begin.")
         st.stop() # Halt if no key and no project started

    active_step = st.session_state.current_step

    # Call functions from features directory based on active_step
    if not st.session_state.project_initialized:
         # Display setup form if project not initialized
         setup.display_setup_step()
    elif active_step == 0:
        # This case might be redundant if setup handles initialization check,
        # but keep for clarity or if setup needs to display differently post-init.
        setup.display_setup_step() # Or a specific post-init view if needed
        # Placeholder content removed
    elif active_step == 1:
        manager_planning.display_manager_planning_step()
        # Placeholder content removed
    elif active_step == 2:
        data_understanding.display_data_understanding_step()
        # Placeholder content removed
    elif active_step == 3:
        analysis_guidance.display_analysis_guidance_step()
        # Placeholder content removed
    elif active_step == 4:
        analysis_execution.display_analysis_execution_step()
        dashboard.render_code_section()
    elif active_step == 5:
        final_report.display_final_report_step()
        # Placeholder content removed
    elif active_step == 6:
        st.error("This feature is currently disabled.")

    step_contexts = {
        0: "setup",
        1: "manager_planning",
        2: "data_understanding",
        3: "analysis_guidance",
        4: "analysis_execution",
        5: "final_report",
    }
    help_system.show_contextual_help(step_contexts.get(active_step, ""))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred in the main application: {e}")
        st.exception(e)
        log_exception(e)
