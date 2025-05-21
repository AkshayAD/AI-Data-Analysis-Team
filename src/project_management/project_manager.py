import json
from pathlib import Path
from datetime import datetime
import streamlit as st


class ProjectManager:
    """Manage saving and loading of project states."""

    def __init__(self):
        self.base_dir = Path.home() / ".ai-data-analysis" / "projects"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_state(self, name: str = "default") -> Path:
        """Persist current project state to disk."""
        state = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user": st.session_state.get("user", "anonymous"),
            "analysis_state": st.session_state.to_dict(),
        }
        path = self.base_dir / f"{name}.json"
        path.write_text(json.dumps(state, indent=2))
        return path

    def load_state(self, name: str = "default"):
        """Load a saved project state into ``st.session_state``."""
        path = self.base_dir / f"{name}.json"
        if not path.exists():
            return None
        state = json.loads(path.read_text())
        for key, value in state.get("analysis_state", {}).items():
            st.session_state[key] = value
        return state
