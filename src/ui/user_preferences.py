import json
from pathlib import Path


class UserPreferences:
    """Manages user customization settings."""

    def __init__(self):
        self.base_path = Path.home() / ".ai-data-analysis" / "preferences.json"
        self.theme = "light"
        self.shortcuts = {}
        self.load()

    def load(self):
        if self.base_path.exists():
            data = json.loads(self.base_path.read_text())
            self.theme = data.get("theme", "light")
            self.shortcuts = data.get("shortcuts", {})

    def save(self):
        self.base_path.parent.mkdir(parents=True, exist_ok=True)
        self.base_path.write_text(
            json.dumps({"theme": self.theme, "shortcuts": self.shortcuts}, indent=2)
        )
