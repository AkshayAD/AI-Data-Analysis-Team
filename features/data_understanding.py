import streamlit as st
import pandas as pd
import polars as pl

from src.utils import (
    configure_genai,
    get_gemini_response,
    generate_data_profile_summary,
    escape_curly_braces,
)
from prompts import ANALYST_PROMPT_TEMPLATE, REVIEWER_PROMPT_TEMPLATE # Import specific prompts
from src.ui_helpers import add_to_conversation, check_api_key, add_download_buttons, display_navigation_buttons # Import necessary helpers

def display_data_understanding_step():
    """Displays the Data Understanding step."""
    st.title("📊 3. AI Analyst - Data Understanding")

    # --- Prerequisite Checks ---
    if not check_api_key():
        st.warning("Please enter your Gemini API Key in the sidebar to continue.", icon="🔑")
        st.stop()
    if not st.session_state.manager_plan:
        st.warning("Manager Plan not available. Please complete Step 2 first.", icon="👨‍💼")
        display_navigation_buttons(next_button_disabled=True)
        st.stop()

    # --- Summary Generation ---
    if st.session_state.analyst_summary is None:
        with st.spinner("AI Analyst is examining data profiles..."):
            # Generate a combined summary of all data profiles
            all_profiles_summary = ""
            for file_name, profile in st.session_state.data_profiles.items():
                try:
                    profile_summary = generate_data_profile_summary(profile)
                    all_profiles_summary += f"\n## Profile: {file_name}\n{profile_summary}\n"
                except Exception as e:
                    all_profiles_summary += f"\n## Profile: {file_name}\nError generating summary: {e}\n"
                    st.warning(f"Could not generate profile summary for {file_name}: {e}")

            for file_name, text in st.session_state.data_texts.items():
                text_snippet = text[:200] + "..." if len(text) > 200 else text
                all_profiles_summary += f"\n## Text Document: {file_name}\nSnippet: {text_snippet}\n"

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
                    st.error(f"Failed to get data summary: {analyst_response}", icon="❌")
                    add_to_conversation("system", f"Error getting Analyst summary: {analyst_response}")
            except KeyError as e:
                st.error(f"Prompt Formatting Error: Missing key {e} in Analyst Summary Prompt template.", icon="⚠️")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}", icon="🔥")

    # --- Display Content ---
    if st.session_state.analyst_summary:
        st.markdown("### Data Summary & Assessment")
        st.markdown(st.session_state.analyst_summary, unsafe_allow_html=True)

        with st.expander("View Raw Data Details", expanded=False):
            if st.session_state.get('dataframes'):
                st.markdown("#### Tabular Data Previews")
                for file_name, df in st.session_state.dataframes.items():
                    st.subheader(f"File: `{file_name}`")
                    st.dataframe(df.head(10).to_pandas()) # Use to_pandas for better rendering

                    profile = st.session_state.data_profiles.get(file_name)
                    if profile and profile.get("file_type") == "tabular":
                        # Display key profile info concisely
                        st.write(f"**Dimensions:** {profile.get('shape', ('N/A', 'N/A'))[0]} rows, {profile.get('shape', ('N/A', 'N/A'))[1]} columns")
                        missing_summary = profile.get('missing_summary')
                        if isinstance(missing_summary, pl.DataFrame):
                            st.write(f"**Missing Values Summary:**")
                            st.dataframe(missing_summary.to_pandas())
                        describe_df = profile.get('numeric_summary')
                        if isinstance(describe_df, pl.DataFrame):
                             st.write(f"**Descriptive Statistics:**")
                             st.dataframe(describe_df.to_pandas())
                    st.markdown("---")

            if st.session_state.get('data_texts'):
                 st.markdown("#### Text Document Snippets")
                 for file_name, text_content in st.session_state.data_texts.items():
                      st.subheader(f"Document: `{file_name}`")
                      text_snippet = text_content[:1000] + "..." if len(text_content) > 1000 else text_content
                      st.text_area("Content Snippet", text_snippet, height=150, disabled=True, key=f"text_snippet_{file_name}")
                      st.markdown("---")

        with st.expander("💬 Consult with another AI Persona"):
            persona_options = ["Manager", "Associate", "Reviewer"]
            selected_consult_persona = st.selectbox("Select Persona:", options=persona_options, key="consult_persona_select_data")
            consultation_request = st.text_area(f"Your question for the {selected_consult_persona}:", key="consult_request_data", placeholder=f"Ask the {selected_consult_persona} about the data summary...")

            if st.button(f"Consult {selected_consult_persona}", key="consult_button_data"):
                if consultation_request:
                    st.info(f"Consultation with {selected_consult_persona} is a planned feature.")
                else:
                    st.warning("Please enter a question for the consultation.", icon="⚠️")

    # --- Navigation ---
    display_navigation_buttons(next_button_disabled=(st.session_state.analyst_summary is None))
    add_download_buttons("DataUnderstanding")
