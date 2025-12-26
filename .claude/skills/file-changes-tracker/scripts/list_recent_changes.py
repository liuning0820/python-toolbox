#!/usr/bin/env python3
"""
File Changes Tracker - Lists files changed in the last week, ordered by modification date.

This script uses git log to find files that have been modified in the last 7 days
and displays them sorted by the most recent modification date.
"""

import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path


def run_git_command(command, cwd=None):
    """Run a git command and return the output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=cwd,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running git command: {e}")
        print(f"Error output: {e.stderr}")
        return None


def get_files_changed_last_week(workspace_path=None):
    """
    Get files changed in the last week, ordered by modification date.

    Args:
        workspace_path: Path to the git repository. If None, uses current directory.

    Returns:
        List of tuples: (filepath, datetime, author, commit_hash)
    """
    if workspace_path is None:
        workspace_path = Path.cwd()

    # Check if this is a git repository
    if not (Path(workspace_path) / '.git').exists():
        print(f"Error: {workspace_path} is not a git repository")
        return []

    # Get git log for files changed in the last week
    # Format: --pretty=format:"%H|%an|%ae|%ad" --date=iso
    # This gives: commit_hash|author_name|author_email|date
    cmd = 'git log --name-only --since="1 week ago" --pretty=format:"%H|%an|%ae|%ad" --date=iso'

    output = run_git_command(cmd, cwd=workspace_path)
    if output is None:
        return []

    # Parse the output
    file_changes = []
    current_commit_info = None

    for line in output.split('\n'):
        line = line.strip()
        if not line:
            continue

        # Check if this is a commit header line (contains | separator)
        if '|' in line and len(line.split('|')) == 4:
            # Parse commit info: hash|author|email|date
            parts = line.split('|')
            if len(parts) == 4:
                commit_hash, author, email, date_str = parts
                try:
                    # Parse the ISO date
                    # Remove timezone info if present (e.g., +0000)
                    date_str = date_str.split('+')[0].strip()
                    # Convert space to T for ISO format
                    date_str = date_str.replace(' ', 'T')
                    commit_date = datetime.fromisoformat(date_str)
                    current_commit_info = (commit_hash, author, commit_date)
                except ValueError as e:
                    print(f"Warning: Could not parse date '{date_str}': {e}")
                    current_commit_info = None
        elif current_commit_info and line:
            # This is a file path
            commit_hash, author, commit_date = current_commit_info
            file_path = line

            # Check if we already have this file (keep the most recent change)
            existing_entry = None
            for i, (existing_file, existing_date, existing_author, existing_hash) in enumerate(file_changes):
                if existing_file == file_path:
                    existing_entry = i
                    break

            if existing_entry is not None:
                # Update if this commit is more recent
                if commit_date > file_changes[existing_entry][1]:
                    file_changes[existing_entry] = (file_path, commit_date, author, commit_hash)
            else:
                # Add new file entry
                file_changes.append((file_path, commit_date, author, commit_hash))

    # Sort by date (most recent first)
    file_changes.sort(key=lambda x: x[1], reverse=True)

    return file_changes


def format_file_list(file_changes):
    """Format the file changes list for display."""
    if not file_changes:
        return "No files have been changed in the last week."

    output = []
    output.append("Files changed in the last week (ordered by most recent modification):")
    output.append("=" * 80)

    for file_path, mod_date, author, commit_hash in file_changes:
        # Format the date
        date_str = mod_date.strftime("%Y-%m-%d %H:%M")

        # Truncate long file paths
        if len(file_path) > 60:
            file_path = "..." + file_path[-57:]

        output.append(f"{file_path:<60} {date_str}  {author:<15} {commit_hash[:7]}")

    output.append(f"\nTotal: {len(file_changes)} files changed")

    return "\n".join(output)


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="List files changed in the last week, ordered by modification date"
    )
    parser.add_argument(
        "--path",
        help="Path to the git repository (default: current directory)",
        default=None
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format"
    )

    args = parser.parse_args()

    workspace_path = args.path
    if workspace_path:
        workspace_path = Path(workspace_path).resolve()

    file_changes = get_files_changed_last_week(workspace_path)

    if args.json:
        # Output as JSON
        import json
        json_output = []
        for file_path, mod_date, author, commit_hash in file_changes:
            json_output.append({
                "file": file_path,
                "modified": mod_date.isoformat(),
                "author": author,
                "commit": commit_hash
            })
        print(json.dumps(json_output, indent=2))
    else:
        # Output as formatted text
        print(format_file_list(file_changes))


if __name__ == "__main__":
    main()