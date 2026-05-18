"""
Configuration for RepoMap module.
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import List


@dataclass
class RepoMapConfig:
    """Configuration for RepoMap generation."""

    # API settings
    api_key: str = ""
    api_base: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"

    # Token limits
    max_map_tokens: int = 4096  # Max tokens for repo map
    max_file_size: int = 100_000  # Skip files larger than 100KB

    # File filtering
    include_patterns: List[str] = field(default_factory=lambda: [
        "*.py", "*.pyx", "*.pyi",
        "*.js", "*.ts", "*.jsx", "*.tsx",
        "*.go", "*.rs", "*.java", "*.kt",
        "*.cpp", "*.c", "*.h", "*.hpp",
        "*.cs", "*.rb", "*.php",
    ])

    exclude_patterns: List[str] = field(default_factory=lambda: [
        "__pycache__", ".git", ".venv", "venv", ".env",
        "node_modules", ".pytest_cache", ".mypy_cache",
        ".tox", "build", "dist", ".eggs", "*.egg-info",
        ".tox", ".nox", ".coverage", "htmlcov",
        ".hypothesis", ".serverless", ".webpack",
        ".next", ".nuxt", ".cache", ".parcel-cache",
        "*.pyc", "*.pyo", "*.pyd", ".DS_Store",
        "package-lock.json", "yarn.lock", "poetry.lock",
        "Pipfile.lock", "requirements.txt", "*.whl",
        ".idea", ".vscode", ".vs", "*.swp", "*.swo",
    ])

    # Depth limit for directory traversal
    max_depth: int = 10

    # Cache settings
    cache_dir: str = ".repomap_cache"
    cache_enabled: bool = True

    @classmethod
    def from_env(cls) -> "RepoMapConfig":
        """Create config from environment variables."""
        import os
        return cls(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            api_base=os.getenv("DEEPSEEK_API_BASE", cls.api_base),
            model=os.getenv("DEEPSEEK_MODEL", cls.model),
        )
