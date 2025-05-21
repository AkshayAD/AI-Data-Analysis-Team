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
from src.ui_helpers import add_to_conversation, check_api_key, add_download_buttons # Import necessary helpers

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
