import streamlit as st
import markdown # Added for HTML report generation

from prompts import MANAGER_REPORT_PROMPT_TEMPLATE # Import specific prompts
from src.utils import configure_genai, get_gemini_response
from src.ui_helpers import (
    add_to_conversation,
    check_api_key,
    add_download_buttons,
    format_results_markdown,
    display_navigation_buttons
)

def display_final_report_step():
    """Displays the Final Report step."""
    st.title("📝 6. AI Manager - Final Report")

    # --- Prerequisite Checks ---
    if not check_api_key():
        st.warning("Please enter your Gemini API Key in the sidebar to continue.", icon="🔑")
        st.stop()
    if not st.session_state.manager_plan or not st.session_state.analyst_summary or not st.session_state.analysis_results:
        missing = []
        if not st.session_state.manager_plan: missing.append("Manager Plan (Step 2)")
        if not st.session_state.analyst_summary: missing.append("Analyst Summary (Step 3)")
        if not st.session_state.analysis_results: missing.append("Analysis Results (Step 5)")
        st.warning(f"The following prerequisites are missing: {', '.join(missing)}.", icon="⚠️")
        st.info("Please complete all previous steps to generate the final report.")
        display_navigation_buttons(next_button_disabled=True)
        st.stop()

    # --- Report Generation ---
    if st.session_state.final_report is None:
        st.info("This step synthesizes all project findings into a final report using the AI Manager.")
        if st.button("Generate Final Report", type="primary"):
            with st.spinner("AI Manager is drafting the final report..."):
                try:
                    results_summary = format_results_markdown(st.session_state.analysis_results)
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
                        st.success("Final report generated successfully!", icon="✅")
                        st.rerun()
                    else:
                        st.error(f"Failed to generate report: {report_response}", icon="❌")
                        add_to_conversation("system", f"Error generating final report: {report_response}")

                except KeyError as e:
                    st.error(f"Prompt Formatting Error: Missing key {e} in Manager Report Prompt template.", icon="⚠️")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}", icon="🔥")

    # --- Display Report and Downloads ---
    if st.session_state.final_report:
        st.markdown("### Final Report Draft")
        st.markdown(st.session_state.final_report, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Download Report")
        col1, col2 = st.columns(2)
        with col1:
            try:
                st.download_button(
                     label="Download as Markdown (.md)",
                     data=st.session_state.final_report,
                     file_name=f"{st.session_state.project_name}_Final_Report.md",
                     mime="text/markdown",
                     key="download_report_md",
                     use_container_width=True
                )
            except Exception as e:
                st.error(f"Error creating Markdown download: {e}")
        with col2:
            try:
                html_report = markdown.markdown(st.session_state.final_report)
                st.download_button(
                   label="Download as HTML (.html)",
                   data=html_report,
                   file_name=f"{st.session_state.project_name}_Final_Report.html",
                   mime="text/html",
                   key="download_report_html",
                   use_container_width=True
                   )
            except Exception as e:
                st.warning(f"Could not generate HTML download: {e}")

    # --- Navigation ---
    display_navigation_buttons() # No "Next" button on the final step
    add_download_buttons("FinalReport")
