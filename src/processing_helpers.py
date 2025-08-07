import streamlit as st # Needed for st.warning/st.error
import re
import json # Although not used directly, analyst response parsing might involve JSON in future

def parse_associate_tasks(guidance_text):
    """
    Attempts to parse actionable tasks from the Associate's guidance using regex splitting.
    Extracts blocks starting with "Task N:".
    """
    if not guidance_text:
        return []

    # Regex to identify the start of a new Task block.
    # This pattern looks for lines starting with optional whitespace, optional bolding (**),
    # "Task", a space, a number, an optional period, optional bolding, and a colon.
    # It's designed to be used as a delimiter for splitting.
    task_header_pattern = re.compile(r"(^\s*\**\s*Task\s*\d+\.?\s*\**:\s*)", re.IGNORECASE | re.MULTILINE)

    # Stop processing at a specific narrative marker to avoid including irrelevant text.
    stop_marker = "5. Develop Narrative:"
    if stop_marker in guidance_text:
        guidance_text = guidance_text.split(stop_marker)[0]

    # Split the guidance text by the task headers. The `re.split` will include the
    # delimiters in the results array because of the capturing group in the pattern.
    parts = task_header_pattern.split(guidance_text)

    # The first part is the text before any tasks, so we discard it.
    # Then, we group the delimiter (header) with its content that follows.
    # e.g., parts = ["", "Task 1:", "Content 1", "Task 2:", "Content 2"]
    # We want ["Task 1: Content 1", "Task 2: Content 2"]
    if len(parts) > 1:
        # Combine header with its corresponding content
        tasks = [header + content for header, content in zip(parts[1::2], parts[2::2])]
        # Strip whitespace from each task block
        formatted_tasks = [task.strip() for task in tasks if task.strip()]
    else:
        formatted_tasks = []


    # If no specific "Task N:" items were parsed, it could be that the format is
    # different. As a fallback, we can treat the entire guidance as one actionable block.
    if not formatted_tasks and guidance_text.strip():
        formatted_tasks = [guidance_text.strip()]

    # If the list is still empty, show a warning and provide a default option.
    if not formatted_tasks:
        try:
            st.warning("Could not automatically parse specific tasks from Associate guidance.")
        except Exception:
            # Fallback for non-Streamlit environments
            print("Warning: Could not automatically parse specific tasks from Associate guidance.")
        # Provide a default prompt to guide the user
        return ["Manually define task based on guidance above."]

    # Always provide a manual option for the user.
    formatted_tasks.append("Manually define task below")

    return formatted_tasks

def parse_analyst_task_response(response_text):
    """
    Parses the Analyst's response into Approach, Code, Results, and Insights.
    Uses robust header matching and content extraction. This version is more resilient
    to variations in LLM output, such as missing sections or different ordering.
    """
    if not response_text:
        return {
            "approach": "Error: No response from Analyst.",
            "code": "# No code provided.",
            "results_text": "# No results provided.",
            "insights": "# No insights provided."
        }

    # Define headers with flexible regex to catch variations (case-insensitive, optional elements).
    headers = {
        "approach": r"^\s*\d*\.?\s*\**approach\**:",
        "code": r"^\s*\d*\.?\s*\**python?\s*code\**:",
        "results_text": r"^\s*\d*\.?\s*\**results\**:",
        "insights": r"^\s*\d*\.?\s*\**key?\s*insights\**:",
    }

    # Default parts dictionary to ensure all keys are present in the final output.
    parts = {
        "approach": "Could not parse 'Approach' section.",
        "code": "# Could not parse 'Python Code' section.",
        "results_text": "Could not parse 'Results' section.",
        "insights": "Could not parse 'Key Insights' section."
    }

    # Find all occurrences of headers in the text to determine section boundaries.
    section_matches = []
    for key, pattern in headers.items():
        # re.finditer allows finding all non-overlapping matches.
        for match in re.finditer(pattern, response_text, re.MULTILINE | re.IGNORECASE):
            section_matches.append({"key": key, "start": match.start(), "end": match.end()})

    # Sort sections by their starting position in the text. This handles cases where
    # the LLM provides sections in a non-standard order.
    sorted_sections = sorted(section_matches, key=lambda x: x["start"])

    # Extract content for each identified section.
    for i, section_match in enumerate(sorted_sections):
        key = section_match["key"]
        # Content begins immediately after the matched header.
        content_start_index = section_match["end"]

        # Content ends at the beginning of the next section, or at the end of the text
        # if this is the last section.
        content_end_index = len(response_text)
        if i + 1 < len(sorted_sections):
            content_end_index = sorted_sections[i+1]["start"]

        content = response_text[content_start_index:content_end_index].strip()

        # Clean up the extracted content.
        # For the code block, remove markdown fences (e.g., ```python ... ```).
        if key == 'code':
            content = re.sub(r"^```python\n?|^```\n?", "", content, flags=re.IGNORECASE)
            content = re.sub(r"\n?```$", "", content, flags=re.IGNORECASE)
            content = content.strip()
        # For other sections, you might remove other unwanted artifacts if necessary.

        parts[key] = content

    return parts
