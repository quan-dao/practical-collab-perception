import subprocess
from pathlib import Path

from .version import __version__
import gitinfo


__all__ = [
    '__version__'
]


def get_git_commit_number():
    git_info = gitinfo.get_git_info()
    git_commit_number = git_info['commit'][:7]
    return git_commit_number


script_version = f"0.5.2+{get_git_commit_number()}"


if script_version not in __version__:
    __version__ = __version__ + '+py%s' % script_version
