import streamlit as st

from src.utils import (
    configure_genai,
    get_gemini_response,
    escape_curly_braces,
)
from prompts import ASSOCIATE_PROMPT_TEMPLATE, REVIEWER_PROMPT_TEMPLATE # Import specific prompts
from src.ui_helpers import add_to_conversation, check_api_key, add_download_buttons, display_navigation_buttons

def display_analysis_guidance_step():
    """Displays the Analysis Guidance step."""
    st.title("🔍 4. AI Associate - Analysis Guidance")

    # --- Prerequisite Checks ---
    if not check_api_key():
        st.warning("Please enter your Gemini API Key in the sidebar to continue.", icon="🔑")
        st.stop()
    if not st.session_state.analyst_summary:
        st.warning("Analyst Summary not available. Please complete Step 3 first.", icon="📊")
        display_navigation_buttons(next_button_disabled=True)
        st.stop()
    if not st.session_state.manager_plan:
        st.warning("Manager Plan not available. Please complete Step 2 first.", icon="👨‍💼")
        display_navigation_buttons(next_button_disabled=True)
        st.stop()

    # --- Guidance Generation ---
    if st.session_state.associate_guidance is None:
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
                    st.error(f"Failed to get guidance: {assoc_response}", icon="❌")
                    add_to_conversation("system", f"Error getting Associate guidance: {assoc_response}")
            except KeyError as e:
                st.error(f"Prompt Formatting Error: Missing key {e} in Associate Guidance Prompt template.", icon="⚠️")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}", icon="🔥")

    # --- Display Content ---
    if st.session_state.associate_guidance:
        st.markdown("### Analysis Guidance & Next Tasks")
        st.markdown(st.session_state.associate_guidance, unsafe_allow_html=True)

        with st.expander("💬 Consult with another AI Persona"):
            persona_options = ["Manager", "Analyst", "Reviewer"]
            selected_consult_persona = st.selectbox("Select Persona:", options=persona_options, key="consult_persona_select_guidance")
            consultation_request = st.text_area(f"Your question for the {selected_consult_persona}:", key="consult_request_guidance", placeholder=f"Ask the {selected_consult_persona} about the guidance...")

            if st.button(f"Consult {selected_consult_persona}", key="consult_button_guidance"):
                if consultation_request:
                    st.info(f"Consultation with {selected_consult_persona} is a planned feature.")
                else:
                    st.warning("Please enter a question for the consultation.", icon="⚠️")

    # --- Navigation ---
    display_navigation_buttons(next_button_disabled=(st.session_state.associate_guidance is None))
    add_download_buttons("AnalysisGuidance")
