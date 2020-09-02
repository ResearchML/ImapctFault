from pydriller import GitRepository
from pydriller import RepositoryMining
import os
gr = GitRepository("sosy")
for commit in RepositoryMining ('sosy',single ='c80dbbd22b34add5952388e885dc2e9baf80b9f9').traverse_commits():
    print(commit.hash)
    print(commit.msg)
    for modified_files in commit.modifications:
        buggy_commits = gr.get_commits_last_modified_lines(commit,modified_files)
        print("       {} ".format(buggy_commits)) # result: (abc, def)
