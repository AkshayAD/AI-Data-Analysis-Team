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
from src.ui_helpers import add_to_conversation, check_api_key, add_download_buttons # Import necessary helpers

def display_manager_planning_step():
    """Displays the Manager Planning step."""
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

        # --- Consultation/Review Section ---
        with st.expander("💬 Consult with AI Persona"):
            st.markdown("Select an AI persona to consult with regarding the current plan.")
            
            persona_options = ["Manager", "Analyst", "Associate", "Reviewer"]
            selected_consult_persona = st.selectbox(
                "Select Persona:",
                options=persona_options,
                key="consult_persona_select_manager_planning"
            )

            consultation_request = st.text_area(f"Your message to the {selected_consult_persona}:", key="consult_request_manager_planning")

            if st.button(f"Send to {selected_consult_persona}", key="consult_button_manager_planning"):
                if consultation_request:
                    if not check_api_key(): st.stop()

                    # Determine which prompt template and persona name to use
                    persona_key = selected_consult_persona.lower()
                    prompt_template_key = f"{persona_key}_prompt_template"
                    
                    # Special case for Reviewer, use the specific reviewer template
                    if persona_key == "reviewer":
                         prompt_template_key = "reviewer_prompt_template"
                         # For reviewer, format the template with specific context
                         project_artifacts = f"Current Analysis Plan:\n{st.session_state.manager_plan}"
                         project_artifacts = escape_curly_braces(project_artifacts)
                         consult_prompt = st.session_state[prompt_template_key].format(
                             project_name=st.session_state.project_name,
                             problem_statement=st.session_state.problem_statement,
                             current_stage="Manager Planning",
                             project_artifacts=project_artifacts,
                             specific_request=consultation_request
                         )
                    else:
                         # For other personas, use a generic consultation format
                         # This might need refinement based on how each persona should respond to arbitrary questions
                         # For now, format using their main template if possible, or a generic wrapper
                         try:
                             # Attempt to use the persona's main template, providing relevant context
                             # This assumes the template can handle additional context like a specific question
                             # This might fail if the template expects specific keys not available here.
                             # A more robust approach might be a dedicated 'consultation_prompt_template' per persona.
                             # For now, let's try a generic wrapper + their main template content
                             generic_consult_wrapper = f"""
                             You are the AI {selected_consult_persona}. A user is consulting with you about the current analysis plan.

                             **Current Analysis Plan:**
                             {st.session_state.manager_plan}

                             **User's Question/Request:**
                             {consultation_request}

                             **Your Task:** Respond to the user's request based on your persona's expertise regarding the plan.
                             """
                             consult_prompt = generic_consult_wrapper # Use the wrapper for now

                         except KeyError as e:
                             st.error(f"Error preparing prompt for {selected_consult_persona}: Missing key {e}. Template might not support this type of consultation.")
                             add_to_conversation("system", f"Error preparing consultation prompt for {selected_consult_persona}: Missing key {e}")
                             st.stop() # Stop execution on prompt error
                         except Exception as e:
                             st.error(f"An unexpected error occurred preparing prompt for {selected_consult_persona}: {e}")
                             add_to_conversation("system", f"Error preparing consultation prompt for {selected_consult_persona}: {e}")
                             st.stop() # Stop execution on prompt error


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
            if st.button("Next: Data Understanding"):
                st.session_state.current_step = 2
                st.rerun()

        add_download_buttons("ManagerPlanning")
