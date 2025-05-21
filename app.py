import streamlit as st
import os
import json
import io
import markdown
import re
import openpyxl
import pandas as pd
import polars as pl

from src.ui.components import AnalysisDashboard
from src.project_management.project_manager import ProjectManager
from src.ui.help_system import HelpSystem
from src.ui.user_preferences import UserPreferences

# Import functions from utils, prompts, and helpers
from src.utils import configure_genai, get_gemini_response, process_uploaded_file
from prompts import (
    MANAGER_PROMPT_TEMPLATE, ANALYST_PROMPT_TEMPLATE, ASSOCIATE_PROMPT_TEMPLATE,
    ANALYST_TASK_PROMPT_TEMPLATE, ASSOCIATE_REVIEW_PROMPT_TEMPLATE,
    MANAGER_REPORT_PROMPT_TEMPLATE, REVIEWER_PROMPT_TEMPLATE
)
# Import feature functions
from src.ui_helpers import add_download_buttons, reset_session, add_to_conversation, check_api_key
from src.processing_helpers import parse_associate_tasks, parse_analyst_task_response
from features import setup, manager_planning, data_understanding, analysis_guidance, analysis_execution, final_report

# Alert user immediately if required deps are missing
if st.session_state.get("genai_import_error", False):
    st.error(
        "google-generativeai package is not installed. "
        "Install dependencies with `pip install -r requirements.txt`."
    )
    st.stop()

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
    'gemini_model': "gemini-2.5-flash-preview-04-17", # Updated default model (using latest flash)
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
from features import setup, manager_planning, data_understanding, analysis_guidance, analysis_execution, final_report

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
                "6. Final Report"     # Step index 5
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
            "gemini-2.5-flash-preview-04-17",
            "gemini-2.5-pro-preview-03-25",
            "gemini-2.5-pro-exp-03-25",
            "gemini-2.0-flash",
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

    step_contexts = {
        0: "setup",
        1: "manager_planning",
        2: "data_understanding",
        3: "analysis_guidance",
        4: "analysis_execution",
        5: "final_report",
    }
    help_system.show_contextual_help(step_contexts.get(active_step, ""))

# The following block of code from the original file (lines approx 279-335)
# has been identified as redundant or outdated placeholder code and is removed.
# This includes the commented out "Handle initial state" section and subsequent
# elif active_step == ... blocks that were either duplicating calls to feature
# modules or contained placeholder UI.

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred in the main application: {e}")
        st.exception(e)
