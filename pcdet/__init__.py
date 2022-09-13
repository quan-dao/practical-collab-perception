import subprocess
from pathlib import Path

from .version import __version__
import gitinfo


__all__ = [
    '__version__'
]


def get_git_commit_number():
    # if not (Path(__file__).parent / '../.git').exists():
    #     return '0000000'
    #
    # cmd_out = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
    # git_commit_number = cmd_out.stdout.decode('utf-8')[:7]
    git_info = gitinfo.get_git_info()
    git_commit_number = git_info['commit'][:7]
    return git_commit_number


script_version = get_git_commit_number()


if script_version not in __version__:
    __version__ = __version__ + '+py%s' % script_version
