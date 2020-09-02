from pydriller import RepositoryMining

for commit in RepositoryMining('https://github.com/java-native-access/jna').traverse_commits():
    print('hash {} authored by {}'.format(commit.hash, commit.author.name))





