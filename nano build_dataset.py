import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
from tqdm import tqdm
from git import Repo, GitCommandError

from radon.complexity import cc_visit
from radon.metrics import mi_visit

from config import (
    REPOS_DIR,
    TARGET_REPOS,
    BUGFIX_KEYWORDS,
    PYTHON_EXTENSIONS,
    MAX_COMMITS_PER_REPO,
)

# -------------------------------------------------------------------
# Radon helpers: compute complexity + maintainability for each file
# -------------------------------------------------------------------

def compute_radon_metrics(source_code: str):
    """
    Compute average cyclomatic complexity and maintainability index
    for a given Python source file.
    """
    avg_cc = 0.0
    mi_score = 0.0

    # Cyclomatic complexity
    try:
        blocks = cc_visit(source_code)
        if blocks:
            avg_cc = sum(b.complexity for b in blocks) / len(blocks)
        else:
            avg_cc = 0.0
    except Exception:
        avg_cc = 0.0

    # Maintainability index
    try:
        mi_score = float(mi_visit(source_code))
    except Exception:
        mi_score = 0.0

    return avg_cc, mi_score


# -------------------------------------------------------------------
# Git helpers
# -------------------------------------------------------------------

def get_default_branch(repo: Repo) -> str:
    """
    Try to detect the main branch name ('main' or 'master').
    Falls back to current HEAD if needed.
    """
    try:
        branch = repo.git.rev_parse("--abbrev-ref", "HEAD")
    except GitCommandError:
        branch = "main"

    if branch == "HEAD":
        # Detached head, try common names
        for cand in ("main", "master"):
            try:
                if cand in repo.heads:
                    branch = cand
                    break
            except Exception:
                continue
    return branch


def find_bugfix_commits(repo: Repo, commits):
    """
    Identify bug-fix commits using commit message keywords and
    build a mapping of (commit_sha, file_path) that were buggy
    using a simplified SZZ approach.

    We mark the parent revision of each bug-fix change as buggy.
    """
    bugfix_commits = []
    bug_keywords_lower = [k.lower() for k in BUGFIX_KEYWORDS]

    print("Identifying bug-fix commits and buggy revisions (simplified SZZ)...")

    # 1) Find bug-fix commits by message
    for c in tqdm(commits, desc="bugfix scan", leave=False):
        msg_lower = c.message.lower()
        if any(k in msg_lower for k in bug_keywords_lower):
            bugfix_commits.append(c)

    buggy_pairs = set()  # (commit_sha, file_path)

    # 2) For each bug-fix commit, mark the parent side as buggy
    for c in tqdm(bugfix_commits, desc="building buggy pairs", leave=False):
        if not c.parents:
            continue
        parent = c.parents[0]
        try:
            diff = parent.diff(c, create_patch=False)
        except GitCommandError:
            continue

        for d in diff:
            path = d.a_path or d.b_path
            if not path:
                continue
            if not any(path.endswith(ext) for ext in PYTHON_EXTENSIONS):
                continue
            buggy_pairs.add((parent.hexsha, path))

    print(f"Total bug-fix commits: {len(bugfix_commits)}")
    print(f"Total buggy (commit, file) pairs: {len(buggy_pairs)}")
    return buggy_pairs


def process_repo(repo_name: str):
    """
    Process a single repository:
    - load git history
    - identify bug-fix commits
    - for each commit & Python file, compute LOC + Radon metrics
    - label (commit, file) as buggy or clean
    """
    repo_path = Path(REPOS_DIR) / repo_name
    if not repo_path.exists():
        print(f"[ERROR] Repo path not found: {repo_path}")
        return []

    print(f"\n=== Processing repo: {repo_name} ===")
    print(f"Path: {repo_path}")

    repo = Repo(repo_path)
    branch = get_default_branch(repo)
    print(f"Using branch: {branch}")

    # Collect commits (optionally limit for speed)
    if MAX_COMMITS_PER_REPO is not None and MAX_COMMITS_PER_REPO > 0:
        commits = list(repo.iter_commits(branch, max_count=MAX_COMMITS_PER_REPO))
    else:
        commits = list(repo.iter_commits(branch))

    print(f"Total commits considered: {len(commits)}")

    # Find buggy revisions via simplified SZZ
    buggy_pairs = find_bugfix_commits(repo, commits)

    rows = []

    # Iterate over commits for feature extraction
    print("Collecting features for each file revision...")
    for c in tqdm(commits, desc=f"{repo_name} - feature extraction"):
        commit_sha = c.hexsha

        # Walk the tree of this commit and consider all Python files
        try:
            tree = c.tree
        except GitCommandError:
            continue

        for blob in tree.traverse():
            if blob.type != "blob":
                continue

            file_path = blob.path
            if not any(file_path.endswith(ext) for ext in PYTHON_EXTENSIONS):
                continue

            try:
                source = blob.data_stream.read().decode("utf-8", errors="ignore")
            except Exception:
                continue

            loc = source.count("\n") + 1
            avg_cc, mi_score = compute_radon_metrics(source)
            buggy = int((commit_sha, file_path) in buggy_pairs)

            rows.append(
                {
                    "repo": repo_name,
                    "commit": commit_sha,
                    "file": file_path,
                    "loc": loc,
                    "avg_complexity": avg_cc,
                    "maintainability": mi_score,
                    "buggy": buggy,
                }
            )

    print(f"Total rows collected for {repo_name}: {len(rows)}")
    return rows


def main():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    all_dfs = []

    for repo_name in TARGET_REPOS:
        rows = process_repo(repo_name)
        if not rows:
            continue

        df_repo = pd.DataFrame(rows)
        csv_path = data_dir / f"{repo_name}_bug_data.csv"
        df_repo.to_csv(csv_path, index=False)
        print(f"[OK] Saved {len(df_repo)} rows → {csv_path}")
        all_dfs.append(df_repo)

    if not all_dfs:
        print("[ERROR] No data collected from any repo.")
        sys.exit(1)

    merged = pd.concat(all_dfs, ignore_index=True)
    merged_path = data_dir / "python_bug_data.csv"
    merged.to_csv(merged_path, index=False)
    print("\nMerging all datasets...")
    print(f"\n✅ Final merged dataset saved to: {merged_path}")
    print(f"Total rows: {len(merged)}")


if __name__ == "__main__":
    main()
