#!/usr/bin/env python3
"""
bug_label_audit.py -- Step 1 feasibility audit for moving RiskLens from
"change-proneness" to "bug (defect) proneness".

For each of Requests, HTTPX, Flask, Click and Rich it answers:

  1. How many closed issues carry a bug label?
  2. How many of those can be linked to a fixing commit?
  3. How many Python source files that existed at the cutoff received
     a bug-fix change after the cutoff?

Two bug-fix definitions are reported:
  ISSUE   commits linked to bug-labelled issues.
  KEYWORD commits with fix/bugfix/regression/crash in the subject,
          excluding likely docs, typo, CI, release and similar changes.

Outputs:
  bug_audit_out/bug_label_audit_summary.csv
  bug_audit_out/bug_labels_<repo>.csv
  bug_audit_out/cache/ (cached GitHub responses)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

REPOS = {
    "requests": "psf/requests",
    "httpx": "encode/httpx",
    "flask": "pallets/flask",
    "click": "pallets/click",
    "rich": "Textualize/rich",
}

API = "https://api.github.com"

FIX_REF_RE = re.compile(
    r"\b(?:fix(?:e[sd])?|close[sd]?|resolve[sd]?)\b[:\s]*"
    r"(?:[\w.-]+/[\w.-]+)?#(\d+)",
    re.I,
)

KW_RE = re.compile(
    r"\b(fix(?:e[sd])?|bug ?fix|regression|crash)\b", re.I
)

KW_EXCL_RE = re.compile(
    r"\b(typos?|docs?|documentation|readme|changelog|ci|"
    r"lint(?:ing)?|format(?:ting)?|style|tests?|release|"
    r"bump|pre-commit|dependabot|deps?|spelling|mypy|typing)\b",
    re.I,
)

DEFAULT_EXCLUDE = (
    r"(^|/)(tests?|docs?|examples?|benchmarks?|scripts?|tools?|\.github)(/|$)"
    r"|(^|/)(test_[^/]*|conftest|setup|noxfile|fabfile)\.py$"
)


# ---------------------------------------------------------------------
# GitHub client
# ---------------------------------------------------------------------

def _next_link(link_header: str):
    for part in link_header.split(","):
        match = re.match(r'\s*<([^>]+)>;\s*rel="next"', part)
        if match:
            return match.group(1)
    return None


class GitHub:
    def __init__(self, token: str):
        self.token = token

    def _request(self, url: str):
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "risklens-bug-label-audit",
            },
        )

        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    body = json.loads(resp.read().decode("utf-8"))

                    remaining = resp.headers.get("X-RateLimit-Remaining")
                    reset = resp.headers.get("X-RateLimit-Reset")

                    if remaining is not None and int(remaining) < 3 and reset:
                        wait = max(
                            0, int(reset) - int(time.time())
                        ) + 5
                        print(
                            f"  Rate limit nearly exhausted; "
                            f"sleeping {wait}s.",
                            file=sys.stderr,
                            flush=True,
                        )
                        time.sleep(wait)

                    return body, resp.headers.get("Link", "")

            except urllib.error.HTTPError as exc:
                if exc.code in (403, 429):
                    retry = exc.headers.get("Retry-After")
                    reset = exc.headers.get("X-RateLimit-Reset")

                    if retry:
                        wait = int(retry) + 2
                    elif (
                        reset
                        and exc.headers.get("X-RateLimit-Remaining") == "0"
                    ):
                        wait = max(
                            0, int(reset) - int(time.time())
                        ) + 5
                    else:
                        wait = 30 * (attempt + 1)

                    print(
                        f"  HTTP {exc.code}; retrying after {wait}s.",
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(wait)
                    continue

                if exc.code in (404, 410):
                    return None, ""

                if exc.code >= 500:
                    wait = 5 * (attempt + 1)
                    print(
                        f"  GitHub server error {exc.code}; "
                        f"retrying after {wait}s.",
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(wait)
                    continue

                raise

            except urllib.error.URLError as exc:
                wait = 5 * (attempt + 1)
                print(
                    f"  Network error: {exc}; retrying after {wait}s.",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(wait)

        raise RuntimeError(f"Giving up on GitHub API request: {url}")

    def paginate(self, url: str):
        """Fetch every page and print progress while doing so."""
        out = []
        page = 0

        while url:
            page += 1
            print(
                f"  Fetching GitHub API page {page}...",
                flush=True,
            )

            body, link = self._request(url)

            if body is None:
                break

            if not isinstance(body, list):
                raise RuntimeError(
                    "Expected a list from GitHub API. "
                    f"Received: {str(body)[:300]}"
                )

            out.extend(body)

            print(
                f"  Page {page} received {len(body)} items "
                f"({len(out)} total so far).",
                flush=True,
            )

            url = _next_link(link)

        return out


def fetch_closed_issues(gh: GitHub, slug: str):
    raw = gh.paginate(
        f"{API}/repos/{slug}/issues"
        "?state=closed&per_page=100&sort=created&direction=asc"
    )

    out = []

    for issue in raw:
        # GitHub's issues endpoint also returns pull requests.
        if "pull_request" in issue:
            continue

        out.append({
            "number": issue["number"],
            "labels": [
                label["name"]
                for label in issue.get("labels", [])
            ],
            "state_reason": issue.get("state_reason"),
        })

    return out


def fetch_closing_commits(gh: GitHub, slug: str, number: int):
    events = gh.paginate(
        f"{API}/repos/{slug}/issues/{number}/timeline?per_page=100"
    )

    return sorted({
        event["commit_id"]
        for event in events
        if event.get("event") == "closed" and event.get("commit_id")
    })


# ---------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------

class GitError(RuntimeError):
    pass


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        [
            "git", "-C", str(repo),
            "-c", "core.quotepath=false",
            *args,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={
            **os.environ,
            "GIT_TERMINAL_PROMPT": "0",
        },
    )

    if result.returncode != 0:
        raise GitError(
            f"git {' '.join(args)} failed: "
            f"{result.stderr.strip()[:200]}"
        )

    return result.stdout


def load_commits(repo: Path):
    """Load non-merge commits and the files they changed."""
    output = git(
        repo,
        "log",
        "--no-merges",
        "--name-only",
        "--format=%x1e%H%x1f%ct%x1f%s%x1f%b%x1d",
    )

    commits = {}

    for chunk in output.split("\x1e"):
        if not chunk.strip():
            continue

        meta, _, files_blob = chunk.partition("\x1d")
        parts = meta.split("\x1f", 3)

        if len(parts) < 4:
            continue

        sha, timestamp, subject, body = parts

        commits[sha.strip()] = {
            "ts": int(timestamp),
            "subject": subject,
            "body": body,
            "files": [
                line for line in files_blob.split("\n")
                if line.strip()
            ],
        }

    return commits


# ---------------------------------------------------------------------
# Audit one repository
# ---------------------------------------------------------------------

def audit_repo(
    name,
    slug,
    repo,
    gh,
    args,
    exclude_re,
    bug_re,
    cache_dir,
):
    print(f"\n=== {name} ({slug}) ===", flush=True)

    if not (repo / ".git").exists():
        print(f"  No Git clone at {repo}; skipping.")
        return None

    if git(repo, "rev-parse", "--is-shallow-repository").strip() == "true":
        print(
            "  WARNING: shallow clone; history is incomplete. "
            "Run git fetch --unshallow."
        )

    def is_source(path: str) -> bool:
        return path.endswith(".py") and not exclude_re.search(path)

    n_closed = 0
    n_bug = 0
    n_bug_linked = 0
    closing = {}
    bug_nums = set()

    # GitHub issue analysis
    if gh is not None:
        issues_cache = cache_dir / f"{name}_closed_issues.json"

        if issues_cache.exists():
            print("  Loading cached closed issues.")
            issues = json.loads(
                issues_cache.read_text(encoding="utf-8")
            )
        else:
            print("  Fetching closed issues...", flush=True)
            issues = fetch_closed_issues(gh, slug)
            issues_cache.write_text(
                json.dumps(issues), encoding="utf-8"
            )

        n_closed = len(issues)

        label_counts = Counter(
            label
            for issue in issues
            for label in issue["labels"]
        )

        print(
            "  Most common labels (* indicates a bug-matching label):"
        )

        for label, count in label_counts.most_common(12):
            marker = "*" if bug_re.search(label) else " "
            print(f"    {marker} {label}: {count}")

        bug_issues = [
            issue
            for issue in issues
            if any(
                bug_re.search(label)
                for label in issue["labels"]
            )
            and issue["state_reason"] != "not_planned"
        ]

        n_bug = len(bug_issues)
        bug_nums = {
            issue["number"] for issue in bug_issues
        }

        timeline_cache = cache_dir / f"{name}_closing_commits.json"

        if timeline_cache.exists():
            closing = {
                int(key): value
                for key, value in json.loads(
                    timeline_cache.read_text(encoding="utf-8")
                ).items()
            }

        todo = [
            issue["number"]
            for issue in bug_issues
            if issue["number"] not in closing
        ]
        todo = todo[:args.max_timeline_calls]

        if todo:
            print(
                f"  Fetching timelines for {len(todo)} bug issues...",
                flush=True,
            )

        for index, number in enumerate(todo, 1):
            closing[number] = fetch_closing_commits(
                gh, slug, number
            )

            if index % 10 == 0 or index == len(todo):
                timeline_cache.write_text(
                    json.dumps(closing), encoding="utf-8"
                )
                print(
                    f"    Timeline progress: {index}/{len(todo)}",
                    flush=True,
                )

        if todo:
            timeline_cache.write_text(
                json.dumps(closing), encoding="utf-8"
            )

        n_bug_linked = sum(
            1 for number in bug_nums if closing.get(number)
        )

    # Git history analysis
    print("  Reading Git commit history...", flush=True)
    commits = load_commits(repo)

    first_parent = git(
        repo, "rev-list", "--first-parent", "HEAD"
    ).split()
    first_parent.reverse()

    if len(first_parent) < 10:
        print("  Too little history; skipping.")
        return None

    cutoff_index = int(
        args.cutoff_frac * (len(first_parent) - 1)
    )
    cutoff = first_parent[cutoff_index]

    cutoff_date = git(
        repo, "show", "-s", "--format=%cs", cutoff
    ).strip()

    post_set = set(
        git(repo, "rev-list", f"{cutoff}..HEAD").split()
    )
    pre_set = set(
        git(repo, "rev-list", cutoff).split()
    )

    files_at_cutoff = [
        path
        for path in git(
            repo, "ls-tree", "-r", "--name-only", cutoff
        ).split("\n")
        if path and is_source(path)
    ]

    cutoff_set = set(files_at_cutoff)
    extra_files = {}

    def files_for(sha: str):
        if sha in commits:
            return commits[sha]["files"]

        if sha not in extra_files:
            try:
                output = git(
                    repo, "diff", "--name-only", f"{sha}^1", sha
                )
                extra_files[sha] = [
                    line for line in output.split("\n")
                    if line.strip()
                ]
            except GitError:
                extra_files[sha] = None

        return extra_files[sha]

    # Build fix commit sets
    issue_fix = {
        sha
        for shas in closing.values()
        for sha in shas
    }

    if bug_nums:
        for sha, commit in commits.items():
            text_to_search = (
                commit["subject"] + "\n" + commit["body"]
            )

            for match in FIX_REF_RE.finditer(text_to_search):
                if int(match.group(1)) in bug_nums:
                    issue_fix.add(sha)
                    break

    keyword_fix = {
        sha
        for sha, commit in commits.items()
        if KW_RE.search(commit["subject"])
        and not KW_EXCL_RE.search(commit["subject"])
    }

    def tally(fix_shas):
        pre = Counter()
        post = Counter()
        stats = Counter()

        for sha in fix_shas:
            files = files_for(sha)

            if files is None:
                stats["not_in_local_clone"] += 1
                continue

            source_files = [
                path for path in files if is_source(path)
            ]

            if not source_files:
                continue

            if len(files) > args.max_files_per_commit:
                stats["skipped_bulk_commit"] += 1
                continue

            if sha in post_set:
                bucket = post
            elif sha in pre_set:
                bucket = pre
            else:
                stats["not_reachable_from_HEAD"] += 1
                continue

            for path in source_files:
                bucket[path] += 1

        return pre, post, stats

    issue_pre, issue_post, issue_stats = tally(issue_fix)
    keyword_pre, keyword_post, keyword_stats = tally(keyword_fix)

    positive_issue = [
        path for path in files_at_cutoff
        if issue_post[path] > 0
    ]

    positive_keyword = [
        path for path in files_at_cutoff
        if keyword_post[path] > 0
    ]

    positive_either = [
        path for path in files_at_cutoff
        if issue_post[path] > 0 or keyword_post[path] > 0
    ]

    new_files_issue = sum(
        1 for path in issue_post if path not in cutoff_set
    )
    new_files_keyword = sum(
        1 for path in keyword_post if path not in cutoff_set
    )

    n_files = len(files_at_cutoff)

    def rate(positives):
        return (
            round(100 * len(positives) / n_files, 1)
            if n_files else 0.0
        )

    if gh is None:
        verdict = "n/a (GitHub skipped)"
    elif len(positive_issue) >= 30:
        verdict = "OK for issue-based labels"
    elif len(positive_issue) >= 15:
        verdict = "THIN - use cautiously / more repos"
    else:
        verdict = "TOO FEW issue-based positives"

    row = {
        "repo": name,
        "commits_no_merge": len(commits),
        "cutoff_date": cutoff_date,
        "src_files_at_cutoff": n_files,
        "closed_issues": n_closed,
        "bug_labelled_issues": n_bug,
        "bug_issues_with_closing_commit": n_bug_linked,
        "issue_fix_commits": len(issue_fix),
        "keyword_fix_commits": len(keyword_fix),
        "pos_issue_post_cutoff": len(positive_issue),
        "pos_issue_rate_pct": rate(positive_issue),
        "pos_keyword_post_cutoff": len(positive_keyword),
        "pos_keyword_rate_pct": rate(positive_keyword),
        "pos_either_post_cutoff": len(positive_either),
        "fixed_post_cutoff_but_new_or_renamed_issue": new_files_issue,
        "fixed_post_cutoff_but_new_or_renamed_keyword": new_files_keyword,
        "files_with_any_issue_fix_ever": len(
            set(issue_pre) | set(issue_post)
        ),
        "files_with_any_keyword_fix_ever": len(
            set(keyword_pre) | set(keyword_post)
        ),
        "issue_stats": dict(issue_stats),
        "keyword_stats": dict(keyword_stats),
        "verdict": verdict,
    }

    print(
        f"  Cutoff date: {cutoff_date}; "
        f"source files at cutoff: {n_files}"
    )
    print(
        f"  Bug-labelled closed issues: {n_bug} "
        f"(with closing commit: {n_bug_linked})"
    )
    print(
        f"  Positives after cutoff - "
        f"ISSUE: {len(positive_issue)} ({rate(positive_issue)}%) | "
        f"KEYWORD: {len(positive_keyword)} "
        f"({rate(positive_keyword)}%) | "
        f"either: {len(positive_either)}"
    )
    print(f"  Verdict: {verdict}")

    # Write one row for each source file existing at the cutoff.
    output_path = args.out_dir / f"bug_labels_{name}.csv"

    with output_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "repo",
            "path",
            "pre_issue_fix_commits",
            "pre_keyword_fix_commits",
            "post_issue_fix_commits",
            "post_keyword_fix_commits",
            "label_issue",
            "label_keyword",
        ])

        for path in files_at_cutoff:
            writer.writerow([
                name,
                path,
                issue_pre[path],
                keyword_pre[path],
                issue_post[path],
                keyword_post[path],
                int(issue_post[path] > 0),
                int(keyword_post[path] > 0),
            ])

    return row


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--repos-dir",
        required=True,
        type=Path,
        help="Folder containing clones: requests, httpx, flask, click, rich",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("bug_audit_out"),
    )
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated subset of repo names",
    )
    parser.add_argument(
        "--cutoff-frac",
        type=float,
        default=0.70,
        help="Cutoff position in first-parent history",
    )
    parser.add_argument(
        "--bug-label-regex",
        default="bug",
        help="Case-insensitive regex for bug labels",
    )
    parser.add_argument(
        "--exclude-regex",
        default=DEFAULT_EXCLUDE,
        help="Paths matching this are not counted as source files",
    )
    parser.add_argument(
        "--max-files-per-commit",
        type=int,
        default=30,
        help="Ignore fix commits touching more files than this",
    )
    parser.add_argument(
        "--max-timeline-calls",
        type=int,
        default=3000,
        help="Maximum timeline calls per repository per run",
    )
    parser.add_argument(
        "--skip-github",
        action="store_true",
        help="Run only local Git keyword analysis",
    )

    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN", "")

    if not args.skip_github and not token:
        print(
            "GITHUB_TOKEN is not set. In PowerShell, set it using "
            "$env:GITHUB_TOKEN='YOUR_TOKEN', or use --skip-github.",
            file=sys.stderr,
        )
        return 2

    gh = None if args.skip_github else GitHub(token)

    exclude_re = re.compile(args.exclude_regex)
    bug_re = re.compile(args.bug_label_regex, re.I)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    wanted = (
        [name.strip() for name in args.only.split(",") if name.strip()]
        or list(REPOS)
    )

    rows = []

    for name in wanted:
        if name not in REPOS:
            print(
                f"Unknown repo '{name}'. Choose from {', '.join(REPOS)}.",
                file=sys.stderr,
            )
            continue

        row = audit_repo(
            name,
            REPOS[name],
            args.repos_dir / name,
            gh,
            args,
            exclude_re,
            bug_re,
            cache_dir,
        )

        if row:
            rows.append(row)

    if not rows:
        print("Nothing audited.")
        return 1

    summary_path = args.out_dir / "bug_label_audit_summary.csv"

    with summary_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()

        for row in rows:
            writer.writerow({
                key: (
                    json.dumps(value)
                    if isinstance(value, dict)
                    else value
                )
                for key, value in row.items()
            })

    print(
        "\n=== SUMMARY "
        "(positives = files at cutoff fixed afterwards) ==="
    )
    print(
        f"{'repo':10} {'files':>6} {'bugIss':>7} "
        f"{'linked':>7} {'posISSUE':>9} {'posKW':>6}  verdict"
    )

    for row in rows:
        print(
            f"{row['repo']:10} "
            f"{row['src_files_at_cutoff']:>6} "
            f"{row['bug_labelled_issues']:>7} "
            f"{row['bug_issues_with_closing_commit']:>7} "
            f"{row['pos_issue_post_cutoff']:>9} "
            f"{row['pos_keyword_post_cutoff']:>6}  "
            f"{row['verdict']}"
        )

    total_files = sum(
        row["src_files_at_cutoff"] for row in rows
    )
    total_issue = sum(
        row["pos_issue_post_cutoff"] for row in rows
    )
    total_keyword = sum(
        row["pos_keyword_post_cutoff"] for row in rows
    )

    print(
        f"{'TOTAL':10} {total_files:>6} {'':>7} "
        f"{'':>7} {total_issue:>9} {total_keyword:>6}"
    )

    print(
        f"\nWrote {summary_path} and per-repository CSV files "
        f"to {args.out_dir}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
