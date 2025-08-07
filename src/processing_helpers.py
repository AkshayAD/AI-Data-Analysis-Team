import streamlit as st # Needed for st.warning/st.error
import re
import json # Although not used directly, analyst response parsing might involve JSON in future

def parse_associate_tasks(guidance_text):
    """
    Attempts to parse actionable tasks from the Associate's guidance.
    Extracts blocks starting with **Task N:** and stops at "5. Develop Narrative:".
    """
    tasks = []
    if not guidance_text:
        return tasks

    # Split the guidance text into lines, keeping line endings to preserve structure within blocks
    lines = guidance_text.strip().splitlines(keepends=True)

    # Regex to identify the start of a new Task block.
    # This pattern looks for lines starting with optional whitespace, followed by optional bolding (**),
    # then "Task", a space, a number, optional period, optional bolding, and a colon.
    task_header_pattern = re.compile(r"^\s*\**\s*Task\s*\d+\.?\s*\**:", re.IGNORECASE)

    current_task_block = []
    formatted_tasks = []

    for line in lines:
        # Check for the stop line first
        if "5. Develop Narrative:" in line:
            # If we were processing a task block, add its current content and stop
            if current_task_block: # Ensure block is not empty before appending
                formatted_tasks.append("".join(current_task_block).strip())
            break # Stop processing lines

        line_stripped = line.strip()
        header_match = task_header_pattern.match(line_stripped)

        if header_match:
            # If we were processing a previous task block, add it to the list
            if current_task_block: # Ensure block is not empty before appending
                formatted_tasks.append("".join(current_task_block).strip())

            # Start a new task block with the current line
            current_task_block = [line]
        # If it's not a task header but we are inside a task block, append the line
        elif current_task_block:
            current_task_block.append(line)

    # Add the last processed task block if it exists after the loop finishes
    if current_task_block:
         formatted_tasks.append("".join(current_task_block).strip())

    # Clean up empty strings that might result from parsing
    formatted_tasks = [task for task in formatted_tasks if task]

    # If no specific Task items were parsed, return the whole guidance as a single option
    # This might happen if the guidance doesn't use "Task N:" format
    if not formatted_tasks and guidance_text.strip():
         # Check if the guidance contains any lines that look like potential tasks (e.g., starting with *, -, or numbers)
         # If so, maybe return the whole guidance as one block, otherwise return default.
         # For now, let's just return the whole guidance if no specific tasks were found.
         formatted_tasks = [guidance_text.strip()]


    # If still empty, provide the default prompt
    if not formatted_tasks:
        try:
            st.warning("Could not automatically parse specific tasks from Associate guidance. Please manually define the task below.")
        except Exception:
            print("Warning: Could not automatically parse specific tasks from Associate guidance.")
        return ["Manually define task based on guidance above."] # Provide a default prompt

    # Ensure the "Manually define task below" option is always available
    if "Manually define task below" not in formatted_tasks:
        formatted_tasks.append("Manually define task below")

    # Remove duplicates while preserving order as much as possible (simple approach)
    seen = set()
    unique_formatted_tasks = []
    for task in formatted_tasks:
        if task not in seen:
            seen.add(task)
            unique_formatted_tasks.append(task)

    formatted_tasks = unique_formatted_tasks

    # Move "Manually define task below" to the end if it exists
    if "Manually define task below" in formatted_tasks:
        formatted_tasks.remove("Manually define task below")
        formatted_tasks.append("Manually define task below")

    # The final check for empty formatted_tasks was redundant and has been removed.
    # If the list was empty, it would have been handled by the check before adding "Manually define task below".

    return formatted_tasks

def parse_analyst_task_response(response_text):
    """
    Parses the Analyst's response into Approach, Code, Results, and Insights.
    Uses more robust header matching and content extraction.
    """
    if not response_text:
        print("Parsing Error: No response text provided.")
        return {"approach": "Error: No response from Analyst.", "code": "", "results_text": "", "insights": ""}

    # Define headers and their potential variations (case-insensitive, flexible spacing/numbering, optional markdown bolding)
    # Make bolding optional and handle variations in header text
    headers = {
        "approach": r"^\s*\d*\.?\s*\**approach\**:", # Optional number, period, optional bolding
        "code": r"^\s*\d*\.?\s*\**python?\s*code\**:", # Optional number, period, optional bolding, optional 'python'
        "results_text": r"^\s*\d*\.?\s*\**results\**:", # Optional number, period, optional bolding
        "insights": r"^\s*\d*\.?\s*\**key?\s*insights\**:", # Optional number, period, optional bolding, optional 'key'
    }

    parts = {
        "approach": "Could not parse 'approach' section.",
        "code": "Could not parse 'code' section.",
        "results_text": "Could not parse 'results_text' section.",
        "insights": "Could not parse 'insights' section."
    }

    # Find the start index of each section using the more flexible patterns
    section_matches = []
    for key, pattern in headers.items():
        for match in re.finditer(pattern, response_text, re.MULTILINE | re.IGNORECASE):
            section_matches.append({"key": key, "start": match.start(), "end": match.end()})

    # Sort found sections by their start index
    sorted_sections = sorted(section_matches, key=lambda x: x["start"])

    print(f"Found {len(sorted_sections)} potential section headers:")
    for match in sorted_sections:
        print(f"- Key: {match['key']}, Start: {match['start']}, End: {match['end']}")


    # Extract content for each section
    for i, section_match in enumerate(sorted_sections):
        key = section_match["key"]
        content_start_index = section_match["end"] # Content starts immediately after the header match

        # The content for the current section goes from content_start_index
        # up to the start index of the next *any* subsequent recognized header,
        # or the end of the text if this is the last recognized header.
        content_end_index = len(response_text)
        if i + 1 < len(sorted_sections):
            content_end_index = sorted_sections[i+1]["start"] # Start of the next found header

        content = response_text[content_start_index:content_end_index].strip()

        # Remove leading/trailing markdown bolding from content
        content = re.sub(r"^\s*\*\*", "", content).strip()
        content = re.sub(r"\*\*\s*$", "", content).strip()


        # Specific cleanup for code block
        if key == 'code':
            # Remove markdown code fences and language specifier
            content = re.sub(r"```python\n?|```", "", content, flags=re.IGNORECASE).strip()

        parts[key] = content
        print(f"Extracted content for '{key}':\n---\n{content[:200]}...\n---\n") # Print snippet of extracted content


    # Ensure all expected keys are present, even if parsing failed
    expected_keys = ["approach", "code", "results_text", "insights"]
    for key in expected_keys:
        if key not in parts:
             parts[key] = f"Could not parse '{key}' section."


    return parts
