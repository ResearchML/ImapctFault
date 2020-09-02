# commit abc modified line 1 of file A
# commit def modified line 2 of file A
# commit ghi modified line 3 of file A
# commit lmn deleted lines 1 and 2 of file A
from pydriller import GitRepository

gr = GitRepository('facebook/rocksdb')
commit = gr.get_commits_last_modified_lines().getcommit('lmn')
buggy_commits = gr.get_commits_last_modified_lines(commit)
print(buggy_commits)
