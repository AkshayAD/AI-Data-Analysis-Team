import streamlit as st


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
