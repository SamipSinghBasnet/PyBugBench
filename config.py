from pathlib import Path

REPOS_DIR = Path("repos")

TARGET_REPOS = [
    "airflow",
    "django",
    "fastapi",
    "flask",
    "matplotlib",
    "numpy",
    "pandas",
    "pytest",
    "requests",
    "scikit-learn",
]

BUGFIX_KEYWORDS = [
    "fix",
    "bug",
    "issue",
    "error",
    "patch",
    "resolve",
    "repair",
]

PYTHON_EXTENSIONS = [".py"]

# None = use full commit history (recommended)
MAX_COMMITS_PER_REPO = None
CODEBERT_MODEL_NAME = "microsoft/codebert-base"
CODEBERT_MAX_FILES = 50000
CODEBERT_BATCH_SIZE = 8
CODEBERT_MAX_TOKENS = 256
