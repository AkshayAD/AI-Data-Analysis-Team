import streamlit as st
import os
import json
import io
import markdown
import re
import openpyxl
import pandas as pd
import polars as pl

from src.utils import configure_genai, get_gemini_response, process_uploaded_file
from prompts import (
    MANAGER_PROMPT_TEMPLATE, ANALYST_PROMPT_TEMPLATE, ASSOCIATE_PROMPT_TEMPLATE,
    ANALYST_TASK_PROMPT_TEMPLATE, ASSOCIATE_REVIEW_PROMPT_TEMPLATE,
    MANAGER_REPORT_PROMPT_TEMPLATE, REVIEWER_PROMPT_TEMPLATE
)
from src.ui_helpers import add_download_buttons, add_to_conversation, display_navigation_buttons # Import necessary helpers

def display_setup_step():
    """Displays the Project Setup step."""
    st.title("🚀 1. Project Setup")

    # Show a warning if the API key is missing
    if not st.session_state.gemini_api_key:
        st.warning("Please enter your Gemini API Key in the sidebar to begin.", icon="🔑")

    with st.form("project_setup_form"):
        st.header("Project Details")
        project_name = st.text_input("Project Name", st.session_state.get("project_name", "AI Analysis Project"))
        problem_statement = st.text_area(
            "Problem Statement / Goal",
            st.session_state.get("problem_statement", ""),
            height=100,
            placeholder="Example: Analyze customer churn to identify key drivers and predict future churn risk."
        )
        data_context = st.text_area(
            "Data Context (Optional)",
            st.session_state.get("data_context", ""),
            height=100,
            placeholder="Example: The dataset includes customer demographics, usage patterns, and subscription details from the last quarter."
        )

        st.header("Upload Data")
        uploaded_files = st.file_uploader(
            "Upload CSV, XLSX, DOCX, or PDF files",
            type=["csv", "xlsx", "docx", "pdf"],
            accept_multiple_files=True,
            key="file_uploader"
        )

        submit_button = st.form_submit_button("🚀 Initialize Project")

        if submit_button:
            # Re-check API key on submission
            if not st.session_state.gemini_api_key:
                 st.error("Please enter your Gemini API Key in the sidebar before initializing.", icon="🚨")
            elif not project_name or not problem_statement or not uploaded_files:
                st.error("Project Name, Problem Statement, and at least one Data File are required.", icon="🚨")
            else:
                # Reset project state before initializing a new one
                st.session_state.dataframes = {}
                st.session_state.data_profiles = {}
                st.session_state.data_texts = {}
                st.session_state.analysis_results = []
                st.session_state.conversation_history = []
                st.session_state.manager_plan = None
                st.session_state.analyst_summary = None
                st.session_state.associate_guidance = None
                st.session_state.final_report = None
                st.session_state.project_initialized = False # Set to false until success

                with st.spinner("Processing uploaded files... This may take a moment."):
                    success_count = 0
                    error_messages = []
                    for uploaded_file in uploaded_files:
                        try:
                            df, profile, text_content = process_uploaded_file(uploaded_file)
                            if df is not None:
                                st.session_state.dataframes[uploaded_file.name] = df
                                st.session_state.data_profiles[uploaded_file.name] = profile
                                success_count += 1
                            if text_content:
                                st.session_state.data_texts[uploaded_file.name] = text_content
                                if df is None:
                                    success_count += 1
                        except Exception as e:
                            error_messages.append(f"Error processing {uploaded_file.name}: {e}")

                    if error_messages:
                        for msg in error_messages:
                            st.error(msg)

                    if success_count > 0:
                        st.session_state.data_uploaded = True
                        st.session_state.project_name = project_name
                        st.session_state.problem_statement = problem_statement
                        st.session_state.data_context = data_context
                        st.session_state.project_initialized = True # Set to True on success

                        # Create and add initial conversation message
                        file_summary = "Uploaded Files:\n"
                        for name, df in st.session_state.dataframes.items():
                            file_summary += f"- Tabular: {name} ({df.height} rows, {df.width} cols)\n"
                        for name in st.session_state.data_texts.keys():
                            file_summary += f"- Text: {name}\n"
                        init_msg = (
                            f"Project: {project_name}\n"
                            f"Problem: {problem_statement}\n"
                            f"Context: {data_context}\n\n"
                            f"{file_summary}"
                        )
                        add_to_conversation("user", init_msg)

                        st.success("Project initialized successfully! You can now proceed to the next step.", icon="✅")
                        # No rerun needed here, the page will update naturally.
                    else:
                        st.error("Initialization failed. No usable data could be extracted. Please check file formats or content.", icon="❌")

    # Display summary and navigation buttons if the project is initialized
    if st.session_state.project_initialized:
        st.success("Project is initialized and ready. Use the buttons below to navigate.", icon="👍")
        with st.expander("View Project Summary", expanded=False):
            st.write(f"**Project Name:** {st.session_state.project_name}")
            st.write(f"**Problem Statement:** {st.session_state.problem_statement}")
            if st.session_state.data_context:
                st.write(f"**Data Context:** {st.session_state.data_context}")

            st.subheader("Uploaded Data Summary")
            if st.session_state.dataframes:
                for name, df in st.session_state.dataframes.items():
                    st.write(f"- **Tabular**: {name} ({df.height} rows, {df.width} cols)")
            if st.session_state.data_texts:
                for name in st.session_state.data_texts.keys():
                    st.write(f"- **Text Document**: {name}")

    # Display navigation buttons at the end of the page content
    # The "Next" button is disabled until the project is successfully initialized.
    display_navigation_buttons(next_button_disabled=not st.session_state.project_initialized)
    add_download_buttons("Setup")
