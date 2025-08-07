import streamlit as st
import os
import json

from src.utils import (
    configure_genai,
    get_gemini_response,
    generate_data_profile_summary,
    escape_curly_braces,
)
from prompts import MANAGER_PROMPT_TEMPLATE, REVIEWER_PROMPT_TEMPLATE # Import specific prompts
from src.ui_helpers import add_to_conversation, check_api_key, add_download_buttons, display_navigation_buttons # Import necessary helpers

def display_manager_planning_step():
    """Displays the Manager Planning step."""
    st.title("👨‍💼 2. AI Manager - Analysis Planning")

    if not check_api_key():
        st.warning("Please enter your Gemini API Key in the sidebar to continue.", icon="🔑")
        st.stop()
    if not st.session_state.project_initialized:
        st.warning("Please initialize a project in Step 1 before planning.", icon="🚀")
        if st.button("Go to Project Setup"):
            st.session_state.current_step = 0
            st.rerun()
        st.stop()


    # Generate plan if it doesn't exist in the session state
    if st.session_state.manager_plan is None:
        # This block runs only once when the plan is needed and not yet created
        with st.spinner("AI Manager is generating the analysis plan..."):
            file_info = ""
            # Create a summary of available data files for the Manager
            for file_name, profile in st.session_state.data_profiles.items():
                file_info += f"\nFile: {file_name}\n"
                if profile:
                    file_info += f"- Columns: {profile.get('columns', 'N/A')}\n"
                    file_info += f"- Shape: {profile.get('shape', 'N/A')}\n"
                else:
                    file_info += "- Profile: Not available.\n"
            for file_name, text in st.session_state.data_texts.items():
                text_snippet = text[:100] + "..." if len(text) > 100 else text
                file_info += f"\nFile: {file_name}\n- Type: Text Document\n- Snippet: {text_snippet}\n"

            try:
                # Format the prompt using the template from session state
                prompt = st.session_state.manager_prompt_template.format(
                    project_name=st.session_state.project_name,
                    problem_statement=st.session_state.problem_statement,
                    data_context=st.session_state.data_context,
                    file_info=file_info if file_info else "No data files loaded."
                )
                # Get the response from the AI
                manager_response = get_gemini_response(prompt, persona="manager", model=st.session_state.gemini_model)
                if manager_response and not manager_response.startswith("Error:"):
                    st.session_state.manager_plan = manager_response
                    add_to_conversation("manager", f"Generated Analysis Plan:\n{manager_response}")
                    st.rerun() # Rerun to display the newly generated plan immediately
                else:
                    st.error(f"Failed to get plan from Manager: {manager_response}", icon="❌")
                    add_to_conversation("system", f"Error getting Manager plan: {manager_response}")
            except KeyError as e:
                st.error(f"Prompt Formatting Error: Missing key {e} in Manager Prompt template. Check settings.", icon="⚠️")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}", icon="🔥")


    # Display plan and interaction options once the plan exists
    if st.session_state.manager_plan:
        st.markdown("### Analysis Plan")
        st.markdown(st.session_state.manager_plan, unsafe_allow_html=True)

        with st.expander("Provide Feedback to Refine Plan"):
             feedback = st.text_area("Your feedback on the plan:", key="manager_feedback_input", placeholder="e.g., 'Please focus more on customer segmentation.'")
             if st.button("Send Feedback", key="manager_feedback_btn"):
                  if feedback:
                       add_to_conversation("user", f"Feedback on Manager Plan: {feedback}")
                       with st.spinner("Manager is revising plan based on your feedback..."):
                            revision_prompt = f"""
                            **Original Plan:**
                            {st.session_state.manager_plan}

                            **User Feedback:**
                            {feedback}

                            **Your Task (as AI Manager):**
                            Revise the original analysis plan based ONLY on the user feedback. Maintain the structure and format. Output only the revised plan.
                            """
                            try:
                                revised_plan = get_gemini_response(revision_prompt, persona="manager", model=st.session_state.gemini_model)
                                if revised_plan and not revised_plan.startswith("Error:"):
                                     st.session_state.manager_plan = revised_plan
                                     add_to_conversation("manager", f"Revised Plan:\n{revised_plan}")
                                     st.success("Plan updated!", icon="✅")
                                     st.rerun()
                                else:
                                     st.error(f"Failed to revise plan: {revised_plan}", icon="❌")
                            except Exception as e:
                                st.error(f"An error occurred during plan revision: {e}", icon="🔥")
                  else:
                       st.warning("Please enter feedback before sending.", icon="⚠️")

        with st.expander("💬 Consult with another AI Persona"):
            persona_options = ["Analyst", "Associate", "Reviewer"]
            selected_consult_persona = st.selectbox("Select Persona:", options=persona_options, key="consult_persona_select_manager")
            consultation_request = st.text_area(f"Your question for the {selected_consult_persona}:", key="consult_request_manager", placeholder=f"Ask the {selected_consult_persona} about the plan...")

            if st.button(f"Consult {selected_consult_persona}", key="consult_button_manager"):
                if consultation_request:
                    # Logic for handling consultation is complex and would be implemented here
                    # For this refactoring, we assume the logic exists and focus on UI flow
                    st.info(f"Consultation with {selected_consult_persona} is a planned feature.")
                else:
                    st.warning("Please enter a question for the consultation.", icon="⚠️")


    # --- Navigation ---
    # Display navigation buttons at the end of the page content
    # The "Next" button is disabled until the manager's plan has been generated.
    display_navigation_buttons(next_button_disabled=(st.session_state.manager_plan is None))
    add_download_buttons("ManagerPlanning")
