#!/usr/bin/env python3
"""
Code Review Script using Local Qwen3 Models via vLLM with OpenAI Compatible API
"""

import os
import sys
import argparse
import json
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Optional, Tuple
import openai
from dataclasses import dataclass
import difflib
import re
import tempfile
import shutil
from urllib.parse import urlparse
from datetime import datetime, timedelta

@dataclass
class ReviewConfig:
    """Configuration for code review"""
    api_base: str
    api_key: str
    model: str
    max_tokens: int = 4000
    temperature: float = 0.1
    instructions_file: str = "instructions.md"

class CodeReviewError(Exception):
    """Custom exception for code review errors"""
    pass

class GitHelper:
    """Helper class for Git operations"""
    
    @staticmethod
    def clone_repository(repo_url: str, target_dir: str, branch: str = None) -> str:
        """Clone a GitHub repository to target directory"""
        try:
            cmd = ["git", "clone"]
            if branch:
                cmd.extend(["-b", branch])
            cmd.extend([repo_url, target_dir])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            return target_dir
        except subprocess.CalledProcessError as e:
            raise CodeReviewError(f"Failed to clone repository {repo_url}: {e.stderr}")
    
    @staticmethod
    def get_commit_hash_by_days(days: int, repo_path: str = ".") -> str:
        """Get commit hash from N days ago"""
        try:
            # Get commit from N days ago
            date_str = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            result = subprocess.run(
                ["git", "rev-list", "-1", "--before", date_str, "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                cwd=repo_path
            )
            commit_hash = result.stdout.strip()
            if not commit_hash:
                # If no commit found for that date, get the first commit
                result = subprocess.run(
                    ["git", "rev-list", "--max-parents=0", "HEAD"],
                    capture_output=True,
                    text=True,
                    check=True,
                    cwd=repo_path
                )
                commit_hash = result.stdout.strip().split('\n')[0]
            return commit_hash
        except subprocess.CalledProcessError as e:
            raise CodeReviewError(f"Failed to get commit from {days} days ago: {e}")
    
    @staticmethod
    def get_commit_hash_by_count(commit_count: int, repo_path: str = ".") -> str:
        """Get commit hash N commits ago"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", f"HEAD~{commit_count}"],
                capture_output=True,
                text=True,
                check=True,
                cwd=repo_path
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise CodeReviewError(f"Failed to get commit {commit_count} commits ago: {e}")
    
    @staticmethod
    def get_changed_files_since_commit(base_commit: str, repo_path: str = ".") -> List[str]:
        """Get list of changed files since a specific commit"""
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", f"{base_commit}...HEAD"],
                capture_output=True,
                text=True,
                check=True,
                cwd=repo_path
            )
            return [f.strip() for f in result.stdout.split('\n') if f.strip()]
        except subprocess.CalledProcessError as e:
            raise CodeReviewError(f"Failed to get changed files since {base_commit}: {e}")
    
    @staticmethod
    def get_changed_files(base_branch: str = "main", repo_path: str = ".") -> List[str]:
        """Get list of changed files compared to base branch"""
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", f"{base_branch}...HEAD"],
                capture_output=True,
                text=True,
                check=True,
                cwd=repo_path
            )
            return [f.strip() for f in result.stdout.split('\n') if f.strip()]
        except subprocess.CalledProcessError as e:
            raise CodeReviewError(f"Failed to get changed files: {e}")
    
    @staticmethod
    def get_all_files(repo_path: str = ".") -> List[str]:
        """Get all tracked files in the repository"""
        try:
            result = subprocess.run(
                ["git", "ls-files"],
                capture_output=True,
                text=True,
                check=True,
                cwd=repo_path
            )
            return [f.strip() for f in result.stdout.split('\n') if f.strip()]
        except subprocess.CalledProcessError as e:
            raise CodeReviewError(f"Failed to get repository files: {e}")
    
    @staticmethod
    def get_file_diff_since_commit(filepath: str, base_commit: str, repo_path: str = ".") -> str:
        """Get diff for a specific file since a commit"""
        try:
            result = subprocess.run(
                ["git", "diff", f"{base_commit}...HEAD", "--", filepath],
                capture_output=True,
                text=True,
                check=True,
                cwd=repo_path
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            raise CodeReviewError(f"Failed to get diff for {filepath} since {base_commit}: {e}")
    
    @staticmethod
    def get_file_diff(filepath: str, base_branch: str = "main", repo_path: str = ".") -> str:
        """Get diff for a specific file"""
        try:
            result = subprocess.run(
                ["git", "diff", f"{base_branch}...HEAD", "--", filepath],
                capture_output=True,
                text=True,
                check=True,
                cwd=repo_path
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            raise CodeReviewError(f"Failed to get diff for {filepath}: {e}")
    
    @staticmethod
    def get_file_content(filepath: str, repo_path: str = ".") -> str:
        """Get current content of a file"""
        try:
            full_path = os.path.join(repo_path, filepath)
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise CodeReviewError(f"Failed to read file {filepath}: {e}")
    
    @staticmethod
    def get_recent_commits_info(repo_path: str = ".", limit: int = 10) -> List[Dict]:
        """Get information about recent commits"""
        try:
            result = subprocess.run(
                ["git", "log", f"--max-count={limit}", "--pretty=format:%H|%s|%an|%ad", "--date=short"],
                capture_output=True,
                text=True,
                check=True,
                cwd=repo_path
            )
            
            commits = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split('|')
                    if len(parts) >= 4:
                        commits.append({
                            'hash': parts[0],
                            'subject': parts[1],
                            'author': parts[2],
                            'date': parts[3]
                        })
            return commits
        except subprocess.CalledProcessError as e:
            raise CodeReviewError(f"Failed to get recent commits: {e}")
    
    @staticmethod
    def is_valid_github_url(url: str) -> bool:
        """Check if URL is a valid GitHub repository URL"""
        try:
            parsed = urlparse(url)
            return (
                parsed.netloc in ['github.com', 'www.github.com'] and
                len(parsed.path.strip('/').split('/')) >= 2
            )
        except:
            return False

class CodeReviewer:
    """Main code reviewer class"""
    
    def __init__(self, config: ReviewConfig):
        self.config = config
        self.client = openai.OpenAI(
            api_key=config.api_key,
            base_url=config.api_base
        )
        self.instructions = self._load_instructions()
    
    def _load_instructions(self) -> str:
        """Load review instructions from file"""
        try:
            with open(self.config.instructions_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise CodeReviewError(f"Failed to load instructions: {e}")
    
    def _get_file_language(self, filepath: str) -> str:
        """Determine programming language from file extension"""
        ext = Path(filepath).suffix.lower()
        language_map = {
            '.go': 'go',
            '.sql': 'sql',
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.cs': 'csharp',
            '.sh': 'bash',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.json': 'json',
            '.xml': 'xml',
            '.html': 'html',
            '.css': 'css',
            '.md': 'markdown',
        }
        return language_map.get(ext, 'text')
    
    def _should_review_file(self, filepath: str) -> bool:
        """Check if file should be reviewed"""
        # Skip certain file types and directories
        skip_patterns = [
            r'\.git/',
            r'node_modules/',
            r'vendor/',
            r'\.vscode/',
            r'\.idea/',
            r'__pycache__/',
            r'\.pyc$',
            r'\.exe$',
            r'\.dll$',
            r'\.so$',
            r'\.dylib$',
            r'\.jpg$',
            r'\.jpeg$',
            r'\.png$',
            r'\.gif$',
            r'\.pdf$',
            r'\.zip$',
            r'\.tar\.gz$',
            r'\.log$',
            r'package-lock\.json$',
            r'yarn\.lock$',
            r'Cargo\.lock$',
            r'go\.sum$',
        ]
        
        for pattern in skip_patterns:
            if re.search(pattern, filepath, re.IGNORECASE):
                return False
        
        return True
    
    
    
    def review_file(self, filepath: str, diff: str = None, full_content: str = None) -> Dict:
        """Review a single file"""
        if not self._should_review_file(filepath):
            return {
                'filepath': filepath,
                'skipped': True,
                'reason': 'File type not suitable for review'
            }
        
        language = self._get_file_language(filepath)
        
        # Prepare the prompt
        prompt_parts = [
            "You are conducting a code review. Follow the instructions provided below.",
            "",
            "# Instructions",
            self.instructions,
            "",
            f"# File to Review: {filepath}",
            f"# Language: {language}",
            ""
        ]
        
        # Add content based on what's available
        if diff and diff.strip():
            prompt_parts.extend([
                "# Git Diff (changes made):",
                "```diff",
                diff,
                "```",
                ""
            ])
        
        if full_content:
            prompt_parts.extend([
                "# Full File Content:",
                f"```{language}",
                full_content,
                "```",
                ""
            ])
        
        # Adjust review focus based on available content
        if diff and not full_content:
            # Diff-only review
            prompt_parts.extend([
                "Please provide a code review focusing on the changes shown in the diff above:",
                "1. Security implications of the changes",
                "2. Performance impact of the changes", 
                "3. Code quality and maintainability of the changes",
                "4. Best practices adherence in the changes",
                "5. Potential bugs or edge cases introduced by the changes",
                "",
                "Note: You are reviewing only the changes (diff), not the entire file.",
                "Format your response with clear categories (Critical, Major, Minor, Suggestions) and specific line references where applicable."
            ])
        elif full_content and not diff:
            # Full file review
            prompt_parts.extend([
                "Please provide a thorough code review of the entire file:",
                "1. Security vulnerabilities",
                "2. Performance issues", 
                "3. Code quality and maintainability",
                "4. Best practices adherence",
                "5. Potential bugs or edge cases",
                "",
                "Format your response with clear categories (Critical, Major, Minor, Suggestions) and specific line references where applicable."
            ])
        else:
            # Both diff and full content available
            prompt_parts.extend([
                "Please provide a thorough code review focusing on:",
                "1. Security vulnerabilities",
                "2. Performance issues", 
                "3. Code quality and maintainability",
                "4. Best practices adherence",
                "5. Potential bugs or edge cases",
                "",
                "Pay special attention to the changes shown in the diff, but consider the full file context.",
                "Format your response with clear categories (Critical, Major, Minor, Suggestions) and specific line references where applicable."
            ])
        
        prompt = "\n".join(prompt_parts)
        
        # ... rest of method remains same
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert code reviewer. Provide thorough, constructive feedback following the given instructions."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            review_content = response.choices[0].message.content
            
            return {
                'filepath': filepath,
                'language': language,
                'review': review_content,
                'skipped': False,
                'tokens_used': response.usage.total_tokens if response.usage else 0
            }
        
        except Exception as e:
            return {
                'filepath': filepath,
                'error': str(e),
                'skipped': True
            }
    
    def review_repository(self, repo_url: str = None, repo_path: str = ".", base_branch: str = "main",
                         files: List[str] = None, review_all: bool = False,
                         since_days: int = None, since_commits: int = None,
                         since_commit: str = None, diff_only: bool = False) -> Dict:
        """Review a repository (local or remote)"""
        temp_dir = None
        working_path = repo_path
        
        try:
            # If repo_url is provided, clone it to a temporary directory
            if repo_url:
                if not GitHelper.is_valid_github_url(repo_url):
                    raise CodeReviewError(f"Invalid GitHub repository URL: {repo_url}")
                
                temp_dir = tempfile.mkdtemp(prefix="code_review_")
                print(f"Cloning repository {repo_url}...")
                working_path = GitHelper.clone_repository(repo_url, temp_dir)
                print(f"Repository cloned to {working_path}")
            
            # Determine base commit for comparison
            base_commit = None
            comparison_info = ""
            
            if since_commit:
                base_commit = since_commit
                comparison_info = f"since commit {since_commit[:8]}"
            elif since_days:
                base_commit = GitHelper.get_commit_hash_by_days(since_days, working_path)
                comparison_info = f"since {since_days} days ago (commit {base_commit[:8]})"
            elif since_commits:
                base_commit = GitHelper.get_commit_hash_by_count(since_commits, working_path)
                comparison_info = f"since {since_commits} commits ago (commit {base_commit[:8]})"
            
            # Determine which files to review
            if files:
                files_to_review = files
            elif review_all:
                files_to_review = GitHelper.get_all_files(working_path)
            elif base_commit:
                files_to_review = GitHelper.get_changed_files_since_commit(base_commit, working_path)
            else:
                files_to_review = GitHelper.get_changed_files(base_branch, working_path)
            
            if not files_to_review:
                return {
                    'summary': f'No files to review {comparison_info}' if comparison_info else 'No files to review',
                    'reviews': []
                }
            
            reviews = []
            total_tokens = 0
            
            print(f"Reviewing {len(files_to_review)} files {comparison_info}...")
            
            # Show recent commits for context
            if base_commit:
                try:
                    recent_commits = GitHelper.get_recent_commits_info(working_path, 5)
                    print("\nRecent commits:")
                    for commit in recent_commits:
                        marker = "→" if commit['hash'].startswith(base_commit[:8]) else " "
                        print(f"  {marker} {commit['hash'][:8]} - {commit['subject']} ({commit['author']}, {commit['date']})")
                    print()
                except:
                    pass  # Don't fail if we can't get commit info
            
            for i, filepath in enumerate(files_to_review, 1):
                print(f"[{i}/{len(files_to_review)}] Reviewing {filepath}...")
                
                try:
                    # Get diff and full content
                    diff = None
                    full_content = None

                    if review_all or files:
                        if not diff_only:
                            full_content = GitHelper.get_file_content(filepath, working_path)
                    else:
                        if base_commit:
                            diff = GitHelper.get_file_diff_since_commit(filepath, base_commit, working_path)
                        else:
                            diff = GitHelper.get_file_diff(filepath, base_branch, working_path)

                        if not diff_only:
                            full_content = GitHelper.get_file_content(filepath, working_path)
                    
                    # Review the file
                    review_result = self.review_file(filepath, diff, full_content)
                    reviews.append(review_result)
                    
                    if 'tokens_used' in review_result:
                        total_tokens += review_result['tokens_used']
                
                except Exception as e:
                    reviews.append({
                        'filepath': filepath,
                        'error': str(e),
                        'skipped': True
                    })
            
            summary_parts = [f"Reviewed {len([r for r in reviews if not r.get('skipped', False)])} files successfully"]
            if comparison_info:
                summary_parts.append(comparison_info)
            
            return {
                'summary': ' '.join(summary_parts),
                'repository_url': repo_url,
                'comparison_info': comparison_info,
                'base_commit': base_commit,
                'total_files': len(files_to_review),
                'total_tokens': total_tokens,
                'reviews': reviews
            }
        
        finally:
            # Clean up temporary directory
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temporary directory: {temp_dir}")

# ... rest of code remains same (ReportGenerator class and other functions)

class ReportGenerator:
    """Generate review reports in different formats"""
    
    @staticmethod
    def generate_markdown_report(review_results: Dict, output_file: str = None) -> str:
        """Generate markdown report"""
        report_lines = [
            "# Code Review Report",
            "",
            f"**Summary:** {review_results['summary']}",
        ]
        
        if review_results.get('repository_url'):
            report_lines.append(f"**Repository:** {review_results['repository_url']}")
        
        if review_results.get('comparison_info'):
            report_lines.append(f"**Comparison:** {review_results['comparison_info']}")
        
        if review_results.get('base_commit'):
            report_lines.append(f"**Base Commit:** {review_results['base_commit']}")
        
        report_lines.extend([
            f"**Total Files:** {review_results.get('total_files', 0)}",
            f"**Total Tokens Used:** {review_results.get('total_tokens', 0)}",
            "",
            "---",
            ""
        ])
        
        for review in review_results['reviews']:
            if review.get('skipped'):
                report_lines.extend([
                    f"## {review['filepath']} (Skipped)",
                    f"**Reason:** {review.get('reason', review.get('error', 'Unknown'))}",
                    ""
                ])
            else:
                report_lines.extend([
                    f"## {review['filepath']}",
                    f"**Language:** {review.get('language', 'Unknown')}",
                    "",
                    review['review'],
                    "",
                    "---",
                    ""
                ])
        
        report_content = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"Report saved to {output_file}")
        
        return report_content
    
    @staticmethod
    def generate_json_report(review_results: Dict, output_file: str = None) -> str:
        """Generate JSON report"""
        report_content = json.dumps(review_results, indent=2, ensure_ascii=False)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"JSON report saved to {output_file}")
        
        return report_content

def load_config() -> ReviewConfig:
    """Load configuration from environment variables"""
    load_dotenv()
    api_base = os.getenv('QWEN_API_BASE')
    api_key = os.getenv('QWEN_API_KEY')
    model = os.getenv('QWEN_MODEL')
    
    if not api_key:
        raise CodeReviewError("QWEN_API_KEY environment variable is required")
    
    return ReviewConfig(
        api_base=api_base,
        api_key=api_key,
        model=model
    )


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='AI Code Review using Local Qwen3 Models')
    parser.add_argument('--repo-url', help='GitHub repository URL to clone and review')
    parser.add_argument('--repo-path', default='.', help='Local repository path (default: current directory)')
    parser.add_argument('--base-branch', default='main', help='Base branch for comparison (default: main)')
    parser.add_argument('--files', nargs='+', help='Specific files to review')
    parser.add_argument('--review-all', action='store_true', help='Review all files in repository (not just changes)')
    
    # New options for trunk-based development
    parser.add_argument('--since-days', type=int, help='Review changes since N days ago')
    parser.add_argument('--since-commits', type=int, help='Review changes since N commits ago')
    parser.add_argument('--since-commit', help='Review changes since specific commit hash')
    
    # New option to control diff vs full file review
    parser.add_argument('--diff-only', action='store_true', help='Review only the diff/changes (not full file content)')
    
    parser.add_argument('--output', help='Output file for report')
    parser.add_argument('--format', choices=['markdown', 'json'], default='markdown', help='Report format')
    parser.add_argument('--instructions', default='instructions.md', help='Instructions file path')
    parser.add_argument('--model', help='Override model name')
    parser.add_argument('--api-base', help='Override API base URL')
    
    args = parser.parse_args()
    

    try:
        # Validate mutually exclusive options
        time_options = [args.since_days, args.since_commits, args.since_commit]
        if sum(x is not None for x in time_options) > 1:
            raise CodeReviewError("Only one of --since-days, --since-commits, or --since-commit can be specified")
        
        # Load configuration
        config = load_config()
        
        # Override config with command line arguments
        if args.instructions:
            config.instructions_file = args.instructions
        if args.model:
            config.model = args.model
        if args.api_base:
            config.api_base = args.api_base
        
        # Initialize reviewer
        reviewer = CodeReviewer(config)
        
        # Perform review
        print("Starting code review...")
        results = reviewer.review_repository(
            repo_url=args.repo_url,
            repo_path=args.repo_path,
            base_branch=args.base_branch,
            files=args.files,
            review_all=args.review_all,
            since_days=args.since_days,
            since_commits=args.since_commits,
            since_commit=args.since_commit,
            diff_only=args.diff_only  # Pass the new option
        )

        
        # Generate report
        if args.format == 'json':
            report = ReportGenerator.generate_json_report(results, args.output)
        else:
            report = ReportGenerator.generate_markdown_report(results, args.output)
        
        # Print to console if no output file specified
        if not args.output:
            print("\n" + "="*80)
            print("CODE REVIEW REPORT")
            print("="*80)
            print(report)
        
        print(f"\nReview completed successfully!")
        print(f"Files reviewed: {results.get('total_files', 0)}")
        print(f"Tokens used: {results.get('total_tokens', 0)}")
    
    except CodeReviewError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nReview interrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
