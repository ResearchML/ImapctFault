import csv
import json

from pydriller import GitRepository, RepositoryMining
bugs = ["fixed ", "fixes ", " fixed", " fixes", "crash", " resolves",
"resolves ", "regression", "fall back", "assertion", "coverity", "reproducible",
"stack-wanted", "steps-wanted", "testcase", "failur", "fail", "npe ",
" npe", "except", "broken", "bug", "differential testing", "error", "addresssanitizer",
"hang ", " hang", "permaorange", "random orange", "intermittent", "test fix",
"steps to reproduce", "crash", "assertion", "failure", "leak", "stack trace", "heap overflow", "freez", "str:", "fix ", " fix", "problem ", " problem", " overflow", "overflow ", "avoid ", " avoid", " issue", "issue ", "workaround ", " workaround", "break ", " break", " stop", "stop "];
#projectList = ["conscrypt","javacpp","java-smt","jna","jpype","zstd-jni"]
projectList = ["vlc-android"]
for project in projectList:
    data_file = open('bug-detals-conscrypt.csv', mode='w', encoding='utf-8')
    data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    data_writer.writerow(['project', 'Bugfixing',"Name","Email", "Date","Bugs","Commit-message","","Total-BugInducing",'Bug-Inducing'])
    data_writer.writerow([project, '',"","", "","","","","",''])


    data_file2 = open('bug-detals-conscrypt-summery.csv', mode='w', encoding='utf-8')
    data_writer2 = csv.writer(data_file2, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    data_writer2.writerow(['project',"Total-Commits",'Bugfixing',"Total-BugInducing", "Total-Added","Total-Deleted","Added-BugFixing", "Deleted-Bugfixing"])

    gr = GitRepository(project)
    total_buggy_commits = 0
    total_inducing_commits = 0
    total_added = 0
    total_deleted = 0

    added_buggy = 0
    deleted_buggy = 0
    index = 0
    for commit in RepositoryMining(project).traverse_commits():
        print("{} : {}".format(index, commit.hash))
        index += 1
        flag = 0
        bug_msg = ""
        added = 0
        deleted = 0
        for modified in commit.modifications:
            added += modified.added
            deleted += modified.removed
        total_added += added
        total_deleted += deleted
        for bug in bugs:
            if bug in commit.msg.lower():
                flag = 1
                bug_msg = "{}{} , ".format(bug_msg, bug)
                added_buggy += added
                deleted_buggy += deleted
        if flag == 1:
            total_buggy_commits += 1
            buggy_commits = gr.get_commits_last_modified_lines(commit)
            inducing_commits = 0
            inducing_commits += len(buggy_commits)
            data_writer.writerow(['', commit.hash, commit.author.name, commit.author.email, commit.author_date, bug_msg, commit.msg, "", '{}'.format(inducing_commits), '{}'.format(buggy_commits)])
            print("{} : {} - {}".format(total_buggy_commits, commit.hash,buggy_commits))
            total_inducing_commits += inducing_commits
    data_writer2.writerow([project,'{}'.format(index),'{}'.format(total_buggy_commits),'{}'.format(total_inducing_commits), '{}'.format(total_added),'{}'.format(total_deleted),'{}'.format(added_buggy), '{}'.format(deleted_buggy)])
    data_file.close()
    data_file2.close()
