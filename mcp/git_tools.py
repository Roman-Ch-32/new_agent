# mcp/git_tools.py
"""Git Tools — Инструменты для работы с Git репозиторием"""

import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any


class GitTools:
    """Инструменты для работы с Git"""

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self._check_git()

    def _check_git(self):
        """Проверяет что это git репозиторий"""
        if not (self.repo_path / '.git').exists():
            raise ValueError(f"Not a git repository: {self.repo_path}")

    def _run_git(self, *args: str, timeout: int = 60) -> str:
        """Выполняет git команду"""
        try:
            result = subprocess.run(
                ['git'] + list(args),
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            if result.returncode != 0:
                raise RuntimeError(f"Git error: {result.stderr.strip()}")
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            raise RuntimeError("Git command timed out")

    def get_current_branch(self) -> str:
        """Текущая ветка"""
        return self._run_git('rev-parse', '--abbrev-ref', 'HEAD')

    def get_status(self) -> Dict[str, Any]:
        """Статус репозитория"""
        status_output = self._run_git('status', '--porcelain')

        changed_files = []
        untracked_files = []

        for line in status_output.split('\n'):
            if not line.strip():
                continue
            status = line[:2]
            file_path = line[3:]

            if status.startswith('??'):
                untracked_files.append(file_path)
            else:
                changed_files.append({'status': status, 'path': file_path})

        return {
            'branch': self.get_current_branch(),
            'changed_files': changed_files,
            'untracked_files': untracked_files,
            'clean': len(changed_files) == 0 and len(untracked_files) == 0
        }

    def create_branch(self, branch_name: str, from_branch: str = None) -> Dict[str, Any]:
        """Создать новую ветку"""
        try:
            current = self.get_current_branch()

            if from_branch and from_branch != current:
                self._run_git('checkout', from_branch)
                self._run_git('pull', 'origin', from_branch)

            self._run_git('checkout', '-b', branch_name)

            return {
                'success': True,
                'branch': branch_name,
                'from': from_branch or current,
                'message': f'Created branch {branch_name} from {from_branch or current}'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def commit(self, message: str, files: List[str] = None) -> Dict[str, Any]:
        """Сделать коммит"""
        try:
            if files:
                for file in files:
                    self._run_git('add', file)
            else:
                self._run_git('add', '-A')

            status = self.get_status()
            if status['clean']:
                return {
                    'success': False,
                    'error': 'No changes to commit'
                }

            self._run_git('commit', '-m', message)

            commit_hash = self._run_git('rev-parse', 'HEAD')

            return {
                'success': True,
                'hash': commit_hash[:7],
                'full_hash': commit_hash,
                'message': message,
                'branch': self.get_current_branch()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def push(self, branch: str = None, force: bool = False) -> Dict[str, Any]:
        """Пуш в удалённый репозиторий"""
        try:
            branch = branch or self.get_current_branch()

            try:
                self._run_git('remote', 'get-url', 'origin')
            except:
                return {
                    'success': False,
                    'error': 'No remote origin configured'
                }

            if force:
                self._run_git('push', '-u', '--force', 'origin', branch)
            else:
                self._run_git('push', '-u', 'origin', branch)

            return {
                'success': True,
                'branch': branch,
                'message': f'Pushed to origin/{branch}'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def pull(self, branch: str = None) -> Dict[str, Any]:
        """Пулл из удалённого репозитория"""
        try:
            branch = branch or self.get_current_branch()
            self._run_git('pull', 'origin', branch)

            return {
                'success': True,
                'branch': branch,
                'message': f'Pulled from origin/{branch}'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def get_log(self, limit: int = 10) -> List[Dict[str, Any]]:
        """История коммитов"""
        try:
            output = self._run_git(
                'log', f'-{limit}',
                '--format=%H|%an|%ae|%ai|%s'
            )

            commits = []
            for line in output.split('\n'):
                if not line.strip():
                    continue
                parts = line.split('|', 4)
                if len(parts) >= 5:
                    commits.append({
                        'hash': parts[0][:7],
                        'full_hash': parts[0],
                        'author': parts[1],
                        'email': parts[2],
                        'date': parts[3],
                        'message': parts[4]
                    })

            return commits
        except Exception as e:
            return []

    def diff(self, branch1: str = None, branch2: str = None) -> Dict[str, Any]:
        """Разница между ветками"""
        try:
            branch1 = branch1 or 'HEAD'
            branch2 = branch2 or 'master'

            diff_output = self._run_git('diff', f'{branch2}..{branch1}')
            stat_output = self._run_git('diff', '--stat', f'{branch2}..{branch1}')

            return {
                'success': True,
                'diff': diff_output,
                'stat': stat_output,
                'from': branch2,
                'to': branch1
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def checkout(self, branch: str) -> Dict[str, Any]:
        """Переключиться на ветку"""
        try:
            status = self.get_status()
            if not status['clean']:
                return {
                    'success': False,
                    'error': 'Uncommitted changes. Commit or stash first.'
                }

            self._run_git('checkout', branch)

            return {
                'success': True,
                'branch': branch,
                'message': f'Checked out {branch}'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def merge(self, branch: str) -> Dict[str, Any]:
        """Слияние веток"""
        try:
            current = self.get_current_branch()
            self._run_git('merge', branch)

            return {
                'success': True,
                'merged': branch,
                'into': current,
                'message': f'Merged {branch} into {current}'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def stash(self, message: str = 'WIP') -> Dict[str, Any]:
        """Сохранить изменения в stash"""
        try:
            self._run_git('stash', 'push', '-m', message)

            return {
                'success': True,
                'message': f'Stashed: {message}'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def stash_pop(self) -> Dict[str, Any]:
        """Восстановить из stash"""
        try:
            self._run_git('stash', 'pop')

            return {
                'success': True,
                'message': 'Stash popped'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def get_branches(self, remote: bool = False) -> List[str]:
        """Список веток"""
        try:
            if remote:
                output = self._run_git('branch', '-r')
            else:
                output = self._run_git('branch')

            branches = []
            for line in output.split('\n'):
                branch = line.strip().lstrip('* ').replace('remotes/origin/', '')
                if branch:
                    branches.append(branch)

            return branches
        except Exception as e:
            return []

    def get_remote_url(self) -> Dict[str, Any]:
        """URL удалённого репозитория"""
        try:
            url = self._run_git('remote', 'get-url', 'origin')

            return {
                'success': True,
                'url': url
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def fetch(self) -> Dict[str, Any]:
        """Fetch из удалённого репозитория"""
        try:
            self._run_git('fetch', 'origin')

            return {
                'success': True,
                'message': 'Fetched from origin'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }