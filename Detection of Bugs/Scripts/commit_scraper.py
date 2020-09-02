import csv
from datetime import datetime, timedelta
from time import sleep

from github import Github

repo_author = 'videolan'
repo_name = 'vlc-android'

# Enter below the username of the account that the GIT_ACCESS_KEY token belongs to
GIT_ACCOUNT_NAME = 'MonaAb'
GIT_ACCESS_KEY = 'd6cbdec96444f739813e5e34af147ff406d4b756'


# When the number of remaining calls is less than or equal to this number,
# the script will pause and wait until it is safe to continue executing.
# The information about when it paused and when it will continue will be printed out in the terminal
pause_number = 50

# available Extensions
LANGUAGES_EXTENSION = {'sh': 'shell', 'py': 'python', 'scala': 'scala', 'css': 'css', 'js': 'javascript',
                       'clj': 'clojure', 'java': 'java', 'cpp': 'cpp', 'm': 'objc', 'bat': 'shell', 'PL': 'perl',
                       'c': 'c', 'pl': 'perl', 'txt': 'text', 'flake8': 'flake', 'yml': 'yml',
                       'rst': 'reStructuredText', 'md': 'readme',
                       'gitignore': 'gitignore', 'in': 'in', 'json': 'json', 'LESSER': 'lesser'}
# initialized github with access token
# Using username and token will make you authenticated user.
# Use if 60 calls/h is too little for your use.
# Max calls using this method of authentication: 5000/h;
g = Github(GIT_ACCOUNT_NAME, GIT_ACCESS_KEY, per_page=100, timeout=100)


def check_remaining_calls():
    """
    This function checks the remaining number of calls and pauses if the number of remaining calls is less than pause_number.
    """

    current_time = datetime.now()

    # Add a number of seconds to reset time just to be safe
    reset_time = datetime.fromtimestamp(g.rate_limiting_resettime) + timedelta(seconds=10)

    restart_time = reset_time - current_time

    if g.get_rate_limit().core.remaining <= pause_number:
        print(
            f'Rate limit exceeded, pausing script for {restart_time}\nPaused at {current_time} \nContinuing at {reset_time}')

        sleep(restart_time.total_seconds())

    else:
        pass


def write_to_csv(data):
    """
    Allows to write data in CSV format
    :param data:
    :return:
    """
    with open(f'{repo_name}_files_by_commit_data.csv', 'w', newline='',encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ['language', 'project', 'file_name', 'sha', 'author', 'author_date', 'is_bug', 'insertion',
             'deletion'])
        writer.writerows(data)
    file.close()
    return


def get_files_by_commit_data():
    """
    Get the files changes by commit data for a mentioned user
    :return:
    """
    commit_data = []

    # Check number of remaining API calls before the first call is made
    check_remaining_calls()

    r = g.get_repo(f"{repo_author}/{repo_name}")
    commits = r.get_commits()
    total_commits = commits.totalCount
    print(f"Totally {total_commits} commits founds")
    i = 0
    for commit in commits:
        print(f'Working on pull request {i + 1} out of {total_commits}')

        i += 1
        # Check number of remaining API calls
        check_remaining_calls()
        for file in commit.files:
            filename = file.filename
            if '.' in filename:
                ext = filename.split('.')[-1]
                temp_data = [LANGUAGES_EXTENSION.get(ext, ext)]
            else:
                temp_data = [None]
            temp_data.append(repo_name)
            temp_data.append(filename)
            temp_data.append(commit.sha)
            temp_data.append(commit.commit.author.name)
            temp_data.append(commit.commit.author.date)
            temp_data.append(None)
            temp_data.append(file.additions)
            temp_data.append(file.deletions)
            commit_data.append(temp_data)

    write_to_csv(commit_data)
    print(f'Successfully Completed.')


if __name__ == '__main__':
    # initiate the work flow from here
    print('Processing....')
    try:
        get_files_by_commit_data()
    except Exception as e:
        print(f'Error occurred : {e}')
