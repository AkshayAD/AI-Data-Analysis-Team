class SecurityManager:
    """Basic security checks for code execution."""

    def validate_code(self, code: str) -> bool:
        """Validate code for security concerns."""
        forbidden_imports = ["os.system", "subprocess"]
        return all(imp not in code for imp in forbidden_imports)
