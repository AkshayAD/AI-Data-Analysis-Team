import streamlit as st
import json
import re
import pandas as pd # For displaying dataframes in streamlit
import os # Import the os module to fix the undefined variable error

from src.utils import configure_genai, get_gemini_response
from prompts import ANALYST_TASK_PROMPT_TEMPLATE # Import specific prompts
from src.ui_helpers import add_to_conversation, check_api_key, add_download_buttons # Import necessary helpers
from src.processing_helpers import parse_associate_tasks, parse_analyst_task_response # Import necessary helpers

def display_analysis_execution_step():
    """Displays the Analysis Execution step."""
    st.title("⚙️ 5. AI Analyst - Analysis Execution")
    if not check_api_key(): st.stop()

    if not st.session_state.associate_guidance:
        st.warning("Associate Guidance not available. Please complete Step 4 first.")
        if st.button("Go back to Analysis Guidance"): st.session_state.current_step = 3; st.rerun()
        st.stop()

    st.markdown("### Task Execution")
    st.markdown("Based on the Associate's guidance, select or define a task for the Analyst to execute.")

    # Initialize expander state in session state if it doesn't exist
    if 'show_guidance_expander_state' not in st.session_state:
        st.session_state.show_guidance_expander_state = False # Default to collapsed

    # Display guidance for context, linked to session state
    # Removed 'key' as it's not supported in this Streamlit version
    with st.expander("Show Associate Guidance", expanded=st.session_state.show_guidance_expander_state):
         st.markdown(st.session_state.associate_guidance)
         # Removed explicit state update as it's not needed without 'key'

    # --- TEMPORARY: Display Raw Associate Guidance Markdown ---
    with st.expander("🔍 Raw Associate Guidance Markdown (Temporary for Debugging)"):
        st.text(st.session_state.associate_guidance)
    # --- END TEMPORARY SECTION ---


    # Suggest tasks based on parsing Associate guidance
    suggested_tasks = parse_associate_tasks(st.session_state.associate_guidance)

    # Select or define task
    # Check if 'selected_task_execution' exists, otherwise set default
    if 'selected_task_execution' not in st.session_state:
        st.session_state.selected_task_execution = suggested_tasks[0] if suggested_tasks else "Manually define task below"

    st.session_state.selected_task_execution = st.selectbox(
         "Select suggested task or define manually:",
         options=suggested_tasks + ["Manually define task below"],
         # Try to keep selection, but handle cases where the stored value is not in the current list
         index=(suggested_tasks + ["Manually define task below"]).index(st.session_state.selected_task_execution)
               if st.session_state.selected_task_execution in (suggested_tasks + ["Manually define task below"])
               else (suggested_tasks + ["Manually define task below"]).index("Manually define task below"), # Default to manual if stored value is invalid
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
                        # --- TEMPORARY: Display Raw LLM Output ---
                        with st.expander("🔍 Raw LLM Output (Temporary for Debugging)"):
                            st.text(analyst_response)
                        # --- END TEMPORARY SECTION ---

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
             # Display Approach as regular markdown
             st.markdown(last_result.get('approach', '123.Could not parse "Approach" section.'))
             st.markdown("**Key Insights:**")
             # Display Insights as regular markdown
             st.markdown(last_result.get('insights', 'Could not parse "Key Insights" section.'))
         with col_code:
             st.markdown("**Python Code (Polars + Plotly):**")
             # Use st.text_area for editable code
             st.session_state.last_generated_code = last_result.get('code', '# Could not parse "Python Code" section.') # Store code in session state for editing
             edited_code = st.text_area(
                 "Generated Code:",
                 value=st.session_state.last_generated_code,
                 height=400, # Increase height slightly for better visibility
                 key="editable_code_area",
                 help="Edit the code if needed before running it locally. The text area is scrollable for longer code." # Add help text about scrolling
             )
             # Update the stored code if edited
             st.session_state.last_generated_code = edited_code


         st.markdown("**Results (Description from AI):**")
         # Use st.text to preserve formatting of text-based tables from code output
         st.text(last_result.get('results_text', 'Could not parse "Results" section.'))

         # --- Section for External Execution Output ---
         st.markdown("---")
         st.subheader("External Execution Output")
         st.info("Run the generated code in your local Python environment. If the code produces text output, paste it below. If it generates a Plotly figure object named `fig`, the code should include `fig.write_html('plot_output.html')` to save it. Upload the saved plot file below.")

         # Text area for pasting output
         pasted_output = st.text_area(
             "Paste Code Output Here:",
             height=200,
             key="pasted_output_area",
             help="Paste the text output from running the generated code locally."
         )

         # Button to get insights from pasted output
         if st.button("Get Insights from Pasted Output", key="get_insights_from_output_btn"):
             if pasted_output:
                 if not check_api_key(): st.stop()
                 with st.spinner("AI Analyst is generating insights from the output..."):
                     # Prepare prompt for Analyst to get insights from output
                     output_insights_prompt = st.session_state.analyst_task_prompt_template.format(
                         project_name=st.session_state.project_name,
                         problem_statement=st.session_state.problem_statement,
                         previous_results_summary="Context: Analyzing output from a previous task.", # Provide context
                         task_to_execute=f"Analyze the following output based on the original task: {last_result.get('task', 'N/A')}",
                         file_names=", ".join(last_result.get('files', [])), # Use files from the task
                         available_columns="N/A (Analyzing output, not raw data)",
                         data_sample=f"Original Code:\n```python\n{edited_code}\n```\n\nPasted Output:\n```\n{pasted_output}\n```" # Include code and output as sample
                     )
                     # Modify prompt slightly to focus on interpreting output
                     output_insights_prompt += "\n\nBased on the 'Pasted Output' provided, interpret the results and provide key insights related to the original task. Focus on explaining what the output means in the context of the analysis."


                     insights_response = get_gemini_response(output_insights_prompt, persona="analyst", model=st.session_state.gemini_model)

                     if insights_response and not insights_response.startswith("Error:"):
                         # Store the new insights (maybe append or replace a specific insights field?)
                         # For now, let's just display it below the button
                         st.session_state.output_insights = insights_response
                         add_to_conversation("analyst", f"Insights from Pasted Output:\n{insights_response}")
                         st.success("Insights generated!")
                         st.rerun() # Rerun to display insights
                     else:
                         st.error(f"Failed to get insights from output: {insights_response}")
                         add_to_conversation("system", f"Error getting insights from output: {insights_response}")
             else:
                 st.warning("Please paste the code output first.")

         # Display generated insights from output if available
         if st.session_state.get('output_insights'):
             st.markdown("#### Insights from Output")
             st.markdown(st.session_state.output_insights)
             # Clear after displaying
             # del st.session_state.output_insights # Or keep it? Let's keep for now.


         # File uploader for plots
         uploaded_plot_file = st.file_uploader(
             "Upload Saved Plot File (HTML, PNG, JPG):",
             type=["html", "png", "jpg", "jpeg"],
             key="plot_uploader"
         )

         # Display uploaded plot
         if uploaded_plot_file is not None:
             file_extension = os.path.splitext(uploaded_plot_file.name)[1].lower()
             if file_extension == ".html":
                 # Read HTML content and display
                 html_content = uploaded_plot_file.getvalue().decode("utf-8")
                 st.components.v1.html(html_content, height=500, scrolling=True)
             elif file_extension in [".png", ".jpg", ".jpeg"]:
                 # Display image
                 st.image(uploaded_plot_file)
             else:
                 st.warning(f"Unsupported plot file type for display: {file_extension}")

             # Button to get insights from uploaded plot (based on description)
             if st.button("Get Insights from Uploaded Plot", key="get_insights_from_plot_btn"):
                 if not check_api_key(): st.stop()
                 with st.spinner("AI Analyst is generating insights from the plot description..."):
                     # Prepare prompt for Analyst to get insights from plot description
                     # Send original task, code, and AI's original results_text (plot description)
                     plot_insights_prompt = st.session_state.analyst_task_prompt_template.format(
                         project_name=st.session_state.project_name,
                         problem_statement=st.session_state.problem_statement,
                         previous_results_summary="Context: Analyzing a generated plot based on its description.", # Provide context
                         task_to_execute=f"Analyze the plot described below based on the original task: {last_result.get('task', 'N/A')}",
                         file_names=", ".join(last_result.get('files', [])), # Use files from the task
                         available_columns="N/A (Analyzing plot description)",
                         data_sample=f"Original Code:\n```python\n{edited_code}\n```\n\nAI's Description of Plot:\n```\n{last_result.get('results_text', 'N/A')}\n```" # Include code and AI's description
                     )
                     # Modify prompt slightly to focus on interpreting the described plot
                     plot_insights_prompt += "\n\nBased on the 'AI's Description of Plot' provided, interpret the visualization and provide key insights related to the original task. Focus on explaining what the described plot suggests about the data."

                     plot_insights_response = get_gemini_response(plot_insights_prompt, persona="analyst", model=st.session_state.gemini_model)

                     if plot_insights_response and not plot_insights_response.startswith("Error:"):
                         # Store the new insights (maybe append or replace a specific insights field?)
                         # For now, let's just display it below the button
                         st.session_state.plot_insights = plot_insights_response
                         add_to_conversation("analyst", f"Insights from Uploaded Plot Description:\n{plot_insights_response}")
                         st.success("Insights generated!")
                         st.rerun() # Rerun to display insights
                     else:
                         st.error(f"Failed to get insights from plot description: {plot_insights_response}")
                         add_to_conversation("system", f"Error getting insights from plot description: {plot_insights_response}")


         # Display generated insights from plot description if available
         if st.session_state.get('plot_insights'):
             st.markdown("#### Insights from Plot Description")
             st.markdown(st.session_state.plot_insights)
             # Clear after displaying
             # del st.session_state.plot_insights # Or keep it? Let's keep for now.


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
                            # Use text_area for previous results too for consistency
                            st.text_area(f"Task {task_num} Code:", value=result.get('code', '# N/A'), height=200, key=f"code_area_{task_num}", disabled=True)
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
