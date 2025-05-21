import streamlit as st
import base64
import json
import pandas as pd
import polars as pl
import os
import io

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
    current_model = st.session_state.get('gemini_model', "gemini-2.5-flash-preview-04-17") # Use updated default
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
        'gemini_model': "gemini-2.5-flash-preview-04-17", # Updated default model (using latest flash)
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
