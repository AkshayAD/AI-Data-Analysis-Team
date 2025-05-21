from pathlib import Path
import venv
import subprocess
import asyncio


class VirtualEnvManager:
    """Handles creation of isolated Python environments."""

    def __init__(self):
        self.base_dir = Path.home() / ".ai-data-analysis"
        self.envs_dir = self.base_dir / "environments"
        self.packages_dir = self.base_dir / "packages"
        self.envs_dir.mkdir(parents=True, exist_ok=True)
        self.packages_dir.mkdir(parents=True, exist_ok=True)

    def create_environment(self, name: str = "default") -> Path:
        """Create a virtual environment with pip enabled."""
        env_path = self.envs_dir / name
        if not env_path.exists():
            venv.create(env_path, with_pip=True)
        return env_path

    def install_package(self, package: str, env_name: str = "default") -> subprocess.CompletedProcess:
        """Install a package into the specified environment."""
        env_path = self.create_environment(env_name)
        pip_executable = env_path / "bin" / "pip"
        result = subprocess.run(
            [str(pip_executable), "install", package],
            capture_output=True,
            text=True,
        )
        return result


class CodeExecutor:
    """Execute arbitrary Python code inside a managed virtualenv."""

    def __init__(self, env_manager: VirtualEnvManager):
        self.env_manager = env_manager
        self.timeout = 30

    async def execute_code(self, code: str, env_name: str = "default") -> subprocess.CompletedProcess:
        """Execute the provided code asynchronously.

        Parameters
        ----------
        code: str
            Python code to run.
        env_name: str
            Name of the virtual environment in which to run the code.
        """
        env_path = self.env_manager.create_environment(env_name)
        python_executable = env_path / "bin" / "python"
        process = await asyncio.create_subprocess_exec(
            str(python_executable),
            "-c",
            code,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.timeout)
        except asyncio.TimeoutError:
            process.kill()
            stdout, stderr = await process.communicate()
            stderr += b"\nExecution timed out"
        return subprocess.CompletedProcess(
            args=[str(python_executable), "-c", code],
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
        )

    def execute(self, code: str, env_name: str = "default") -> subprocess.CompletedProcess:
        """Convenience wrapper to run ``execute_code`` synchronously."""
        return asyncio.run(self.execute_code(code, env_name))
