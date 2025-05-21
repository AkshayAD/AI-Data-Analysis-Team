import os
import pandas as pd
import polars as pl # Import Polars
try:
    import google.generativeai as genai
except ModuleNotFoundError:  # pragma: no cover - environment may not have deps
    genai = None
    # When running outside Streamlit context we can't use st.error directly
    # Store a flag so the main app can surface a helpful message
    st.session_state['genai_import_error'] = (
        st.session_state.get('genai_import_error', False) or True
    )
from dotenv import load_dotenv
import streamlit as st
import docx
from PyPDF2 import PdfReader
import io # Needed for reading uploaded file content with Polars

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
