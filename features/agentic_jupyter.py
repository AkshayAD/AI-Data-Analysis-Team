import streamlit as st
import streamlit.components.v1 as components

def display_agentic_jupyter_step():
    """
    Displays the agentic jupyter step.
    """
    st.title("Agentic Jupyter")

    st.write("This is an embedded `marimo` notebook.")

    components.iframe(
        "http://localhost:2718",
        width=None,
        height=600,
        scrolling=False
    )
