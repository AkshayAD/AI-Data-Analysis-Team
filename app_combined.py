import streamlit as st
import pandas as pd # Keep pandas for st.dataframe compatibility if needed, but prefer Polars
import polars as pl # Import polars
import os
import json
import io # Added for Excel download
import markdown # Added for HTML report generation
import re # For parsing LLM responses
import openpyxl # For Excel export
# import subprocess # Import commented out - for potential future automated install
from src.utils import configure_genai, get_gemini_response, process_uploaded_file, generate_data_profile_summary, extract_text_from_docx, extract_text_from_pdf
from src.error_logger import log_exception, read_log, clear_log

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
    # --- Prompt Templates --- (Stored in session state to be editable)
    'manager_prompt_template': """
You are an AI Data Analysis Manager acting as a consultant. Your primary role is to translate business problems into structured analytical projects.

**Context:**
Project Name: {project_name}
Problem Statement: {problem_statement}
Data Context: {data_context}
Available Data Files: {file_info}

**Your Task:**
1.  Clarify Business Objectives: If the problem statement is unclear, ask specific questions to understand the core business goals and desired outcomes.
2.  Develop a Structured Analysis Plan: Create a clear, step-by-step plan (numbered list) outlining the analytical approach. This should include:
    *   Data Understanding & Preparation: Initial checks, potential cleaning needs, feature engineering ideas.
    *   Key Analytical Questions & Hypotheses: Specific, testable questions and hypotheses directly linked to the business objectives.
    *   Proposed Methodologies: Suggest appropriate statistical or analytical techniques for each step.
    *   Risk Assessment: Identify potential challenges (data limitations, methodology constraints).
    *   Expected Deliverables: Outline the anticipated outputs (e.g., summary statistics, visualizations, model results, insights).
3.  Maintain Professionalism: Use clear, concise language suitable for a consulting engagement. State any assumptions made.
""",
    'analyst_prompt_template': """
You are an AI Data Analyst acting as a consultant. Your role is to perform rigorous, objective data exploration and summarization.

**Context:**
Problem Statement: {problem_statement}
Manager's Analysis Plan:
{manager_plan}
Data Profile Summary (from automated profiling):
{data_profiles_summary}

**Your Task:**
1.  Provide a Comprehensive Data Assessment: Based *only* on the provided data profiles and context, write a detailed summary covering:
    *   Key Characteristics: Describe data types, distributions (if available in profile), unique values, etc.
    *   Data Quality Issues: Highlight missing values, potential outliers (based on min/max if numeric), inconsistencies, or other red flags identified in the profile.
    *   Relevance to Plan: Assess how well the available data seems suited to address the Manager's plan and the problem statement. Identify potential gaps or limitations.
    *   Initial Observations: Note any immediate patterns or points of interest suggested by the profile *without making unsupported assumptions*.
2.  Be Objective and Precise: Stick to the facts presented in the data profile. Clearly state any limitations of the profile itself. Use precise language.
3.  Document Clearly: Structure your summary logically.
""",
    'associate_prompt_template': """
You are an AI Senior Data Associate acting as a consultant. Your role is to guide the analytical execution by refining the plan and defining specific tasks based on initial data understanding.

**Context:**
Problem Statement: {problem_statement}
Manager's Analysis Plan:
{manager_plan}
Analyst's Data Summary:
{analyst_summary}

**Your Task:**
1.  Refine Initial Analysis Steps: Based on the Analyst's summary, refine the first few steps of the Manager's plan.
2.  Formulate Testable Hypotheses: Define 2-3 specific, measurable hypotheses relevant to the business problem that can be tested with the available data.
3.  Identify Key Checks: Highlight critical data quality checks or edge cases to investigate based on the Analyst's findings.
4.  Outline Next Analysis Tasks: Provide the Analyst with 2-3 *concrete, actionable* next tasks. Specify the exact analysis, target file(s)/columns, and expected output (e.g., 'Calculate correlation for numeric columns in file X.csv using Polars', 'Generate frequency counts and bar chart for column Y in file Z.csv using Polars and Plotly').
5.  Develop Narrative: Briefly outline the initial storyline or angle for exploration based on the hypotheses and tasks.
6.  Be Strategic and Detailed: Ensure guidance is practical, statistically sound, and clearly linked to the overall project goals.
""",
    'analyst_task_prompt_template': """
You are an AI Data Analyst acting as a consultant. Execute the requested analysis task rigorously and objectively using the specified tools.

**Context:**
Project Name: {project_name}
Problem Statement: {problem_statement}
Previous Analysis Tasks Completed Summary:
{previous_results_summary}
Current Analysis Task: {task_to_execute}
Data Source File(s): {file_names}
Available Columns (relevant file): {available_columns}
Data Sample (first 5 rows of relevant file as JSON):
{data_sample}

**Your Task:** Execute the analysis task using **Polars** for data manipulation and **Plotly Express (px)** for visualization. Provide the following in separate, clearly marked sections:

1.  **Approach:** Briefly explain the steps you will take.
2.  **Python Code:** Provide the complete, executable Python code using Polars (imported as `pl`) and Plotly Express (imported as `px`). Assume the relevant data is loaded into a Polars DataFrame named `df`. For visualizations, generate the Plotly figure object and assign it to a variable named `fig` (e.g., `fig = px.scatter(df, ...)`). **Only output the code required for this specific task.**
3.  **Results:** Describe the key results obtained from executing the code (e.g., calculated statistics, patterns observed). If a visualization was created (assigned to `fig` in the code), describe what it shows clearly.
4.  **Key Insights:** State 1-2 objective insights derived *directly* from these results, linking them back to the analysis task if possible. Avoid speculation.

**Important:** If you encounter issues (e.g., data type problems, missing columns needed for the task), clearly state the issue in the Results section instead of attempting to proceed with invalid code. Use professional and precise language. Respond ONLY with the 4 sections requested.
""",
    'associate_review_prompt_template': """
You are an AI Senior Data Associate acting as a consultant. Critically review the analysis performed by the Analyst.

**Context:**
Problem Statement: {problem_statement}
Original Analysis Guidance Provided to Analyst:
{associate_guidance}
Analysis Results Provided by Analyst:
{analysis_results_summary}

**Your Task:** Provide a structured review covering:
1.  Adherence to Guidance: Did the Analyst correctly execute the tasks outlined in the guidance? Were the specified methods/tools used?
2.  Quality of Results: Are the results clearly presented? Are the insights logical and directly supported by the results?
3.  Progress Towards Goal: How well do these results contribute to answering the overall problem statement or testing the defined hypotheses?
4.  Potential Issues/Limitations: Identify any limitations in the analysis performed, potential biases, or areas needing further investigation.
5.  Recommended Next Steps: Suggest the next logical analysis tasks or refinements based on these findings. Be specific.

**Maintain Objectivity:** Base your review strictly on the provided context. Use clear, constructive language.
""",
    'manager_report_prompt_template': """
You are an AI Data Analysis Manager acting as a consultant. Synthesize the project findings into a professional report for a business audience.

**Context:**
Project Name: {project_name}
Problem Statement: {problem_statement}
Original Analysis Plan:
{manager_plan}
Data Summary Provided by Analyst:
{analyst_summary}
Summary of Analysis Results & Insights:
{analysis_results_summary}

**Your Task:** Generate a coherent final report in Markdown format. Structure the report as follows:

1.  Executive Summary: Briefly state the problem, the approach taken, and the main conclusions/recommendations (2-3 key sentences).
2.  Key Findings & Business Impact: Use bullet points to highlight the most important insights derived from the analysis. Explain the potential business implications of each finding.
3.  Analysis Overview: Briefly describe the data used and the main analytical steps performed (referencing the plan).
4.  Detailed Findings: Elaborate on the key findings, referencing specific analyses or results where appropriate. Include visualizations if they were central to the findings (describe them).
5.  Limitations & Assumptions: Clearly state any limitations of the data or analysis and any key assumptions made.
6.  Recommendations & Next Steps: Provide actionable recommendations based on the findings. Suggest concrete next steps if further analysis is warranted.

**Focus on Clarity and Impact:** Use clear, non-technical language where possible. Ensure the narrative flows logically and emphasizes the business relevance of the findings.
""",
    'reviewer_prompt_template': """
You are an AI Project Director acting as a reviewer. Your role is to provide strategic oversight and quality control on the analysis project.

**Context:**
Project Name: {project_name}
Problem Statement: {problem_statement}
Current Project Stage: {current_stage}
Relevant Project Artifacts:
User's Specific Review Request: {specific_request}

**Your Task:**
Review the provided project artifacts and context based on the user's specific request (if any) or provide a general strategic review if no specific request is given. Focus on:
1.  Strategic Alignment: Does the current work align with the overall project goals and problem statement?
2.  Methodology Soundness: Is the approach logical and appropriate? Are there any potential flaws?
3.  Risk Identification: Are there any potential risks or roadblocks?
4.  Clarity & Communication: Is the information presented clearly? Is it suitable for the intended audience?
5.  Business Impact: Does the analysis effectively address potential value or impact?

Provide constructive feedback and actionable recommendations. Use clear, professional language.
"""
}

# Apply defaults
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Helper Functions ---

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
    for key, value in defaults.items():
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

def parse_associate_tasks(guidance_text):
    """
    Attempts to parse actionable tasks from the Associate's guidance.
    (This is a simple parser and might need adjustment based on LLM output format)
    """
    tasks = []
    if not guidance_text:
        return tasks

    # Look for lines starting with a number and a period (common for lists)
    # or potentially specific keywords if the prompt enforces them.
    # This regex looks for list items or lines mentioning 'task' or 'analysis' explicitly.
    potential_tasks = re.findall(r"^\s*\d+\.\s*(.*)|(?:Next Analysis Tasks:|Task:|Analysis:)\s*(.*)", guidance_text, re.MULTILINE | re.IGNORECASE)

    for match in potential_tasks:
        task_text = next((item for item in match if item), None) # Get the non-empty capture group
        if task_text and len(task_text.strip()) > 10: # Basic check for substance
             # Clean up potential leading/trailing whitespace or list markers
            cleaned_task = re.sub(r"^\s*[\d\.\-\*]+\s*", "", task_text).strip()
            if cleaned_task:
                 tasks.append(cleaned_task)

    # Fallback if no numbered list found, look for bullet points near "Next Tasks"
    if not tasks:
         if "next analysis tasks" in guidance_text.lower():
              lines = guidance_text.split('\n')
              task_section = False
              for line in lines:
                   line_stripped = line.strip()
                   if "next analysis tasks" in line_stripped.lower():
                        task_section = True
                        continue
                   if task_section and (line_stripped.startswith('*') or line_stripped.startswith('-')):
                        cleaned_task = line_stripped[1:].strip()
                        if cleaned_task:
                             tasks.append(cleaned_task)
                   elif task_section and not line_stripped: # Stop if blank line after tasks
                       break
                   elif task_section and not (line_stripped.startswith('*') or line_stripped.startswith('-')): # Stop if non-bullet line
                       break


    if not tasks:
        st.warning("Could not automatically parse specific tasks from Associate guidance. Please manually define the task below.")
        return ["Manually define task based on guidance above."] # Provide a default prompt

    return tasks

def parse_analyst_task_response(response_text):
    """
    Parses the Analyst's response into Approach, Code, Results, and Insights.
    Assumes the LLM followed the requested format with clear headers.
    """
    if not response_text:
        return {"approach": "Error: No response from Analyst.", "code": "", "results_text": "", "insights": ""}

    # Normalize headers (lowercase, remove markdown bolding)
    response_text_lower = response_text.lower()
    response_text_lower = response_text_lower.replace("**", "") # Remove bold markdown

    parts = {
        "approach": None,
        "code": None,
        "results_text": None,
        "insights": None
    }

    # Find indices of headers (more flexible matching)
    indices = {}
    patterns = {
        "approach": r"^\s*1\.\s*approach:",
        "code": r"^\s*2\.\s*(python\s)?code:",
        "results_text": r"^\s*3\.\s*results:",
        "insights": r"^\s*4\.\s*key\sinsights:"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, response_text_lower, re.MULTILINE | re.IGNORECASE)
        if match:
            indices[key] = match.start()
        else:
             # Fallback: Try finding without the number if pattern fails
             fallback_match = re.search(rf"^\s*{key.replace('_text','')}:", response_text_lower, re.MULTILINE | re.IGNORECASE)
             if fallback_match:
                 indices[key] = fallback_match.start()


    # Filter out headers not found
    found_indices = {k: v for k, v in indices.items() if v is not None} # Check for None explicitly

    if not found_indices:
        st.error("Could not parse Analyst response. Headers (e.g., '1. Approach:') not found.")
        # Return the whole response in code section for debugging
        return {"approach": "Error: Could not parse sections.", "code": response_text, "results_text": "", "insights": ""}

    # Sort found headers by their position
    sorted_keys = sorted(found_indices, key=found_indices.get)

    # Extract content between headers
    for i, key in enumerate(sorted_keys):
        start_index = found_indices[key]
        # Find the start of the content (after the header line)
        content_start_match = re.search(r"\n", response_text_lower[start_index:])
        if content_start_match:
            content_start = start_index + content_start_match.end()
        else: # If header is the last line
            content_start = start_index + len(re.match(patterns[key], response_text_lower[start_index:], re.IGNORECASE).group(0)) # Estimate header length

        # Find the end index (start of the next header, or end of string)
        if i + 1 < len(sorted_keys):
            end_index = found_indices[sorted_keys[i+1]]
        else:
            end_index = len(response_text)

        # Extract the original case content
        parts[key] = response_text[content_start:end_index].strip()

        # Specific cleanup for code block
        if key == 'code':
            parts[key] = re.sub(r"```python\n?|```", "", parts[key]).strip() # Remove markdown code fences

    # Handle missing parts gracefully
    for key in parts:
        if parts[key] is None:
            parts[key] = f"Could not parse '{key}' section."

    return parts

def format_conversation_markdown(history):
    """Formats conversation history into a Markdown string."""
    md = "# Conversation History\n\n"
    for msg in history:
        role = msg.get('role', 'unknown').capitalize()
        content = msg.get('content', '')
        md += f"**{role}**: \n"
        # Indent content for readability
        content_lines = str(content).split('\n') # Ensure content is string
        for line in content_lines:
             md += f"> {line}\n"
        md += "\n---\n\n"
    return md

def format_results_markdown(results):
    """Formats analysis results into a Markdown string."""
    if not results:
        return "No analysis results yet."

    md = "# Analysis Results\n\n"
    for i, result in enumerate(results):
        md += f"## Task {i+1}: {result.get('task', 'N/A')}\n\n"
        md += f"**Files Used:** {', '.join(result.get('files', ['N/A']))}\n\n"
        md += f"**Approach:**\n```\n{result.get('approach', 'N/A')}\n```\n\n"
        md += f"**Python Code:**\n```python\n{result.get('code', '# N/A')}\n```\n\n"
        md += f"**Results:**\n```\n{result.get('results_text', 'N/A')}\n```\n\n"
        md += f"**Key Insights:**\n```\n{result.get('insights', 'N/A')}\n```\n\n"
        md += "---\n\n"
    return md

def create_excel_download(history, results):
    """Creates an Excel file in memory for download."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Conversation History Sheet
        if history:
            # Ensure consistent keys even if some messages are malformed
            hist_df = pd.DataFrame([{'role': msg.get('role','N/A'), 'content': str(msg.get('content','N/A'))} for msg in history])
            hist_df.to_excel(writer, sheet_name='Conversation History', index=False)
        else:
            pd.DataFrame([{"status": "No conversation history yet."}]).to_excel(writer, sheet_name='Conversation History', index=False)

        # Analysis Results Sheet
        if results:
            # Ensure all keys exist for consistent columns
            all_keys = set(['task', 'files', 'approach', 'code', 'results_text', 'insights']) # Define expected keys
            for r in results:
                all_keys.update(r.keys()) # Add any unexpected keys found

            # Convert list of files to string for Excel
            results_prepared = []
            for r in results:
                res_copy = r.copy()
                res_copy['files'] = ', '.join(res_copy.get('files', [])) # Join list to string
                results_prepared.append(res_copy)

            results_df = pd.DataFrame([{k: res.get(k, None) for k in all_keys} for res in results_prepared])
            results_df.to_excel(writer, sheet_name='Analysis Results', index=False)
        else:
            pd.DataFrame([{"status": "No analysis results yet."}]).to_excel(writer, sheet_name='Analysis Results', index=False)

    output.seek(0)
    return output

# --- Utility to add Download Buttons ---
def add_download_buttons(step_name):
    """Adds standard download buttons for conversation and results."""
    st.markdown("---")
    st.subheader("Download Artifacts")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Conversation & Project Data")
        # Conversation Markdown
        conv_md = format_conversation_markdown(st.session_state.conversation_history)
        st.download_button(
            label="Download Conversation (Markdown)",
            data=conv_md,
            file_name=f"{st.session_state.project_name}_step_{step_name}_conversation.md",
            mime="text/markdown",
            key=f"download_conv_md_{step_name}"
        )
        # Conversation/Results Excel
        try: # Add error handling for Excel generation
            excel_data = create_excel_download(st.session_state.conversation_history, st.session_state.analysis_results)
            st.download_button(
                label="Download Project Data (Excel)",
                data=excel_data,
                file_name=f"{st.session_state.project_name}_step_{step_name}_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"download_excel_{step_name}"
            )
        except Exception as e:
            st.error(f"Error generating Excel file: {e}")


    with col2:
        # Analysis Results (only show if results exist)
        if st.session_state.analysis_results:
            st.markdown("##### Analysis Results")
            # Results Markdown
            results_md = format_results_markdown(st.session_state.analysis_results)
            st.download_button(
                label="Download Results (Markdown)",
                data=results_md,
                file_name=f"{st.session_state.project_name}_step_{step_name}_results.md",
                mime="text/markdown",
                key=f"download_results_md_{step_name}"
            )
            # Excel download already handled in col1

# --- Main Application Logic ---
def main():
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
        # else: # Project running, key removed?
        #     st.warning("Gemini API Key missing. AI features disabled.")


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
            # Placeholder for future implementation:
            # if st.button("Install Libraries Now (Experimental)"):
            #     with st.spinner("Attempting to install libraries..."):
            #         # Add logic using subprocess to run pip install
            #         st.info("Automated install logic goes here.")


        # Model Selection
        model_options = [
            "gemini-2.5-flash-preview-04-17",
            "gemini-2.5-pro-preview-03-25",
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
             # **IMPORTANT NOTE FOR USER:** Check the 'Analyst Task Prompt' above. Ensure it uses `{file_names}` (plural) and not `{file-name}`.


    # --- Main Content Area ---
    if not st.session_state.gemini_api_key and not st.session_state.project_initialized:
         st.error("Please enter your Gemini API Key in the sidebar to begin.")
         st.stop() # Halt if no key and no project started

    if not st.session_state.project_initialized:
        # Step 0: Project Setup
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
    else:
        # --- Project Initialized - Display Steps ---
        active_step = st.session_state.current_step

        # Function to check API key before AI calls
        def check_api_key():
            if not st.session_state.gemini_api_key:
                st.error("Gemini API Key missing. Please enter it in the sidebar settings.")
                return False
            # Optional: Add a quick test call here if desired
            return True

        if active_step == 0:
            st.title("✅ 1. Project Setup")
            st.success("Project Initialized!")
            st.write(f"**Project Name:** {st.session_state.project_name}")
            st.write(f"**Problem Statement:** {st.session_state.problem_statement}")
            if st.session_state.data_context: st.write(f"**Data Context:** {st.session_state.data_context}")

            st.subheader("Uploaded Data Summary")
            if st.session_state.dataframes:
                for name, df in st.session_state.dataframes.items(): st.write(f"- Tabular: {name} ({df.height} rows, {df.width} cols)") # Use Polars attributes
            if st.session_state.data_texts:
                for name in st.session_state.data_texts.keys(): st.write(f"- Text Document: {name}")

            # Use columns for buttons
            col1, col2 = st.columns([1,4]) # Adjust ratio as needed
            with col1:
                if st.button("Next: Manager Planning"):
                    st.session_state.current_step = 1
                    st.rerun()
            add_download_buttons("Setup") # Add downloads here too

        elif active_step == 1:
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

                # --- Reviewer Section (Integrated from app.py) ---
                with st.expander("⭐ Request Review from Project Director"):
                    reviewer_request = st.text_area("Specific Review Instructions (Optional):", key="review_request_manager_plan")
                    if st.button("Get Review", key="review_button_manager_plan"):
                        with st.spinner("Project Director is reviewing the plan..."):
                            # Gather context for reviewer
                            project_artifacts = f"Current Analysis Plan:\n{st.session_state.manager_plan}"
                            review_context_prompt = st.session_state.reviewer_prompt_template.format(
                                project_name=st.session_state.project_name,
                                problem_statement=st.session_state.problem_statement,
                                current_stage="Manager Planning",
                                project_artifacts=project_artifacts,
                                specific_request=reviewer_request if reviewer_request else "Provide a general strategic review of the analysis plan."
                            )

                            # Call LLM as reviewer
                            review_response = get_gemini_response(review_context_prompt, persona="reviewer", model=st.session_state.gemini_model)

                            if review_response and not review_response.startswith("Error:"):
                                st.session_state.reviewer_response = review_response
                                st.session_state.reviewer_specific_request = reviewer_request # Store request for display clarity
                                add_to_conversation("reviewer", f"Review Request: {reviewer_request}\n\nReview Response:\n{review_response}")
                                st.rerun() # Rerun to display the review
                            elif review_response:
                                st.error(f"Reviewer Error: {review_response}")
                            else:
                                st.error("Failed to get response from Reviewer.")

                # Display reviewer response if available
                if st.session_state.reviewer_response:
                    st.markdown("#### ⭐ Project Director's Review")
                    if st.session_state.reviewer_specific_request:
                        st.caption(f"Review focus: {st.session_state.reviewer_specific_request}")
                    st.markdown(st.session_state.reviewer_response)
                    # Clear after displaying once to avoid showing stale reviews on rerun
                    st.session_state.reviewer_response = None
                    st.session_state.reviewer_specific_request = None
                # --- End Reviewer Section ---


                col1, col2 = st.columns([1,4])
                with col1:
                    if st.button("Next: Data Understanding"):
                        st.session_state.current_step = 2
                        st.rerun()

            add_download_buttons("ManagerPlanning")

        elif active_step == 2:
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
                with st.expander("View Data Details", expanded=False):
                    for file_name, df in st.session_state.dataframes.items():
                        st.subheader(f"File: {file_name}")
                        st.dataframe(df.head(10)) # Display head as Pandas for better Streamlit rendering
                        profile = st.session_state.data_profiles.get(file_name)
                        if profile:
                            st.write(f"Dimensions: {profile.get('shape', 'N/A')}")
                            # Display missing summary if it exists and is a polars DataFrame
                            missing_summary = profile.get('missing_summary')
                            if isinstance(missing_summary, pl.DataFrame) and not missing_summary.is_empty():
                                st.write("Missing Value Counts:")
                                st.dataframe(missing_summary.to_pandas()) # Display as Pandas
                            elif isinstance(missing_summary, pd.DataFrame) and not missing_summary.empty: # Handle pandas case if profile generated it
                                 st.write("Missing Value Counts:")
                                 st.dataframe(missing_summary)
                            elif 'missing_summary' not in profile:
                                 st.write("Missing value info not generated in profile.")
                            else:
                                st.write("No missing values detected in profile.")

                        else:
                             st.write("Basic profile not available.")
                        st.markdown("---")

                    for file_name, text_content in st.session_state.data_texts.items():
                         st.subheader(f"Text Document: {file_name}")
                         text_snippet = text_content[:1000] + "..." if len(text_content) > 1000 else text_content
                         st.text_area("Content Snippet", text_snippet, height=150, disabled=True, key=f"text_snippet_{file_name}")
                         st.markdown("---")


                st.markdown("### Data Summary & Assessment")
                st.markdown(st.session_state.analyst_summary)

                # --- Reviewer Section (Integrated from app.py) ---
                with st.expander("⭐ Request Review from Project Director"):
                    reviewer_request = st.text_area("Specific Review Instructions (Optional):", key="review_request_analyst_summary")
                    if st.button("Get Review", key="review_button_analyst_summary"):
                        with st.spinner("Project Director is reviewing the data summary..."):
                            # Gather context for reviewer
                            project_artifacts = f"Analyst's Data Summary:\n{st.session_state.analyst_summary}"
                            review_context_prompt = st.session_state.reviewer_prompt_template.format(
                                project_name=st.session_state.project_name,
                                problem_statement=st.session_state.problem_statement,
                                current_stage="Data Understanding",
                                project_artifacts=project_artifacts,
                                specific_request=reviewer_request if reviewer_request else "Provide a general strategic review of the data summary."
                            )

                            # Call LLM as reviewer
                            review_response = get_gemini_response(review_context_prompt, persona="reviewer", model=st.session_state.gemini_model)

                            if review_response and not review_response.startswith("Error:"):
                                st.session_state.reviewer_response = review_response
                                st.session_state.reviewer_specific_request = reviewer_request # Store request for display clarity
                                add_to_conversation("reviewer", f"Review Request: {reviewer_request}\n\nReview Response:\n{review_response}")
                                st.rerun() # Rerun to display the review
                            elif review_response:
                                st.error(f"Reviewer Error: {review_response}")
                            else:
                                st.error("Failed to get response from Reviewer.")

                # Display reviewer response if available
                if st.session_state.reviewer_response:
                    st.markdown("#### ⭐ Project Director's Review")
                    if st.session_state.reviewer_specific_request:
                        st.caption(f"Review focus: {st.session_state.reviewer_specific_request}")
                    st.markdown(st.session_state.reviewer_response)
                    # Clear after displaying once
                    st.session_state.reviewer_response = None
                    st.session_state.reviewer_specific_request = None
                # --- End Reviewer Section ---

                col1, col2 = st.columns([1,4])
                with col1:
                    if st.button("Next: Analysis Guidance"):
                        st.session_state.current_step = 3
                        st.rerun()

            add_download_buttons("DataUnderstanding")

        elif active_step == 3:
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

                # --- Reviewer Section (Integrated from app.py) ---
                with st.expander("⭐ Request Review from Project Director"):
                    reviewer_request = st.text_area("Specific Review Instructions (Optional):", key="review_request_associate_guidance")
                    if st.button("Get Review", key="review_button_associate_guidance"):
                        with st.spinner("Project Director is reviewing the guidance..."):
                            # Gather context for reviewer
                            project_artifacts = f"Associate's Guidance:\n{st.session_state.associate_guidance}"
                            review_context_prompt = st.session_state.reviewer_prompt_template.format(
                                project_name=st.session_state.project_name,
                                problem_statement=st.session_state.problem_statement,
                                current_stage="Analysis Guidance",
                                project_artifacts=project_artifacts,
                                specific_request=reviewer_request if reviewer_request else "Provide a general strategic review of the analysis guidance."
                            )

                            # Call LLM as reviewer
                            review_response = get_gemini_response(review_context_prompt, persona="reviewer", model=st.session_state.gemini_model)

                            if review_response and not review_response.startswith("Error:"):
                                st.session_state.reviewer_response = review_response
                                st.session_state.reviewer_specific_request = reviewer_request # Store request for display clarity
                                add_to_conversation("reviewer", f"Review Request: {reviewer_request}\n\nReview Response:\n{review_response}")
                                st.rerun() # Rerun to display the review
                            elif review_response:
                                st.error(f"Reviewer Error: {review_response}")
                            else:
                                st.error("Failed to get response from Reviewer.")

                # Display reviewer response if available
                if st.session_state.reviewer_response:
                    st.markdown("#### ⭐ Project Director's Review")
                    if st.session_state.reviewer_specific_request:
                        st.caption(f"Review focus: {st.session_state.reviewer_specific_request}")
                    st.markdown(st.session_state.reviewer_response)
                    # Clear after displaying once
                    st.session_state.reviewer_response = None
                    st.session_state.reviewer_specific_request = None
                # --- End Reviewer Section ---

                col1, col2 = st.columns([1,4])
                with col1:
                    if st.button("Next: Analysis Execution"):
                        st.session_state.current_step = 4 # Index 4 is Step 5
                        st.rerun()

            add_download_buttons("AnalysisGuidance")

        elif active_step == 4:
            st.title("⚙️ 5. AI Analyst - Analysis Execution")
            if not check_api_key(): st.stop()

            if not st.session_state.associate_guidance:
                st.warning("Associate Guidance not available. Please complete Step 4 first.")
                if st.button("Go back to Analysis Guidance"): st.session_state.current_step = 3; st.rerun()
                st.stop()

            st.markdown("### Task Execution")
            st.markdown("Based on the Associate's guidance, select or define a task for the Analyst to execute.")

            # Display guidance for context
            with st.expander("Show Associate Guidance"):
                 st.markdown(st.session_state.associate_guidance)

            # Suggest tasks based on parsing Associate guidance
            suggested_tasks = parse_associate_tasks(st.session_state.associate_guidance)

            # Select or define task
            # Check if 'selected_task_execution' exists, otherwise set default
            if 'selected_task_execution' not in st.session_state:
                st.session_state.selected_task_execution = suggested_tasks[0] if suggested_tasks else "Manually define task below"

            st.session_state.selected_task_execution = st.selectbox(
                 "Select suggested task or define manually:",
                 options=suggested_tasks + ["Manually define task below"],
                 index=(suggested_tasks + ["Manually define task below"]).index(st.session_state.selected_task_execution), # Try to keep selection
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
                     st.markdown(f"```\n{last_result.get('approach', 'N/A')}\n```")
                     st.markdown("**Key Insights:**")
                     st.markdown(f"```\n{last_result.get('insights', 'N/A')}\n```")
                 with col_code:
                     st.markdown("**Python Code (Polars + Plotly):**")
                     st.code(last_result.get('code', '# No code generated'), language='python')

                 st.markdown("**Results (Description from AI):**")
                 # Use markdown for potentially formatted results from AI
                 st.markdown(last_result.get('results_text', 'N/A'))
                 # Add note about execution - IMPORTANT
                 st.info("ℹ️ **Note:** Code is generated by the AI based on the task and data sample. It is **not executed** within this application. Visualizations are described by the AI, not rendered live. You would need to run this code in your own Python environment.")


                 # Expander for all previous results
                 with st.expander("View All Task Results"):
                      # Iterate in reverse to show newest first
                      for i, result in enumerate(reversed(st.session_state.analysis_results)):
                           task_num = len(st.session_state.analysis_results) - i
                           st.markdown(f"---")
                           st.markdown(f"##### Task {task_num}: {result.get('task', 'N/A')}")
                           st.markdown(f"*Files Used:* {', '.join(result.get('files', []))}")
                           with st.container(): # Use container for better visual separation
                                c1, c2 = st.columns(2)
                                with c1:
                                     st.markdown("**Approach:**")
                                     st.text(result.get('approach', 'N/A'))
                                     st.markdown("**Insights:**")
                                     st.text(result.get('insights', 'N/A'))
                                with c2:
                                    st.markdown("**Code:**")
                                    st.code(result.get('code', '# N/A'), language='python')
                           st.markdown("**Results Description:**")
                           st.markdown(result.get('results_text', 'N/A')) # Use markdown here too


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


        elif active_step == 5:
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


if __name__ == "__main__":
    # Guard against running utils functions at global scope if they depend on streamlit widgets
    # For example, if utils uses st.secrets or other session state elements not yet initialized
    try:
        main()
    except Exception as e:
        # Generic error catch for unexpected issues during app execution
        st.error(f"An unexpected error occurred in the main application: {e}")
        st.exception(e) # Show traceback in streamlit for debugging
        log_exception(e)
