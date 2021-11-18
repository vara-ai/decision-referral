import subprocess


class RunGitCommandError(Exception):
    pass


def get_commit_sha():
    try:
        sha_string = _run_git_command("rev-parse --short HEAD")
    except RunGitCommandError:
        sha_string = ">no git"

    return sha_string


def _run_git_command(command):
    git_command = "git {}".format(command)
    try:
        output = subprocess.getoutput(git_command)
    except subprocess.CalledProcessError:
        raise RunGitCommandError('Git command "{}" threw an error'.format(git_command))

    return output.rstrip()
