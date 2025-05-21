import asyncio
import streamlit as st
import plotly.express as px

from src.code_execution.executor import VirtualEnvManager, CodeExecutor
from src.security.security_manager import SecurityManager


class AnalysisDashboard:
    """UI components for interactive code execution and visualization."""

    def __init__(self):
        self.env_manager = VirtualEnvManager()
        self.executor = CodeExecutor(self.env_manager)
        self.security = SecurityManager()

    def render_code_section(self):
        with st.container():
            st.markdown("### 📝 Code Editor")

            # Package Management
            with st.expander("📦 Manage Dependencies"):
                self._render_package_manager()

            # Code Editor
            self._render_code_editor()

            # Execution Controls
            self._render_execution_controls()

    def _render_package_manager(self):
        col1, col2 = st.columns([3, 1])
        with col1:
            pkg = st.text_input("Package Name", key="pkg_name", placeholder="e.g., pandas==2.0.0")
        with col2:
            if st.button("Install", key="install_pkg_btn") and pkg:
                result = self.env_manager.install_package(pkg)
                st.text(result.stdout)
                if result.stderr:
                    st.text(result.stderr)

    def _render_code_editor(self):
        st.text_area("Python Code", height=300, key="code_editor")

    def _render_execution_controls(self):
        if st.button("Run Code", key="run_code_btn"):
            code = st.session_state.get("code_editor", "")
            if not code:
                st.warning("Please enter code to run.")
                return
            if not self.security.validate_code(code):
                st.error("Security check failed for the provided code.")
                return
            result = self.executor.execute(code)
            st.text_area("stdout", result.stdout.decode(), height=150)
            if result.stderr:
                st.text_area("stderr", result.stderr.decode(), height=150)
