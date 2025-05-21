import streamlit as st

from src.utils import (
    configure_genai,
    get_gemini_response,
    escape_curly_braces,
)
from prompts import ASSOCIATE_PROMPT_TEMPLATE, REVIEWER_PROMPT_TEMPLATE # Import specific prompts
from src.ui_helpers import add_to_conversation, check_api_key, add_download_buttons # Import necessary helpers

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
