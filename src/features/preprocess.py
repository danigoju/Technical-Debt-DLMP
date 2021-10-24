import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

'''
Returns a preprocessed dataframe 

Params:
    - dataframe: git_commits (coming from GIT_COMMITS table from td_VD_2.db)
    - dataframe: szz_fault_inducing_commits (coming from SZZ_FAULT_INDUCING_COMMITS table from td_VD_2.db)
    - dataframe: refactoring_miner (coming from RERFACTORING_MINER table from td_VD_2.db)
    - dataframe: git_commits_changes (coming from GIT_COMMITS_CHANGES table from td_VD_2.db)

Returns:
    - A dataframe containing the refactors that have induced faults including information about the commit
      This dataframe is thought to be used to train models to predict whether a refactor is going to induce a fault or not
'''

def preprocess(git_commits, szz_fault_inducing_commits, refactoring_miner, git_commits_changes):

    # keep columns and rows of interest of each dataframe individually and change formats

    git_commits = git_commits[['PROJECT_ID','COMMIT_HASH','COMMIT_MESSAGE','COMMITTER_DATE','BRANCHES']]
    szz_fault_inducing_commits = szz_fault_inducing_commits[["FAULT_INDUCING_COMMIT_HASH"]].drop_duplicates(ignore_index=True)
    refactoring_miner = refactoring_miner[["COMMIT_HASH", "REFACTORING_TYPE"]].drop_duplicates(ignore_index=True)
    git_commits_changes = git_commits_changes[["COMMIT_HASH","COMMITTER_ID","LINES_ADDED","LINES_REMOVED", "DATE"]]
    git_commits_changes["LINES_ADDED"] = git_commits_changes["LINES_ADDED"].astype(int)
    git_commits_changes["LINES_REMOVED"] = git_commits_changes["LINES_REMOVED"].astype(int)
    git_commits_changes = git_commits_changes.groupby(['COMMIT_HASH', 'COMMITTER_ID']).sum().reset_index()

    # creation of a temporal refference column



    # merge of dataframes
    
    refactoring_fault_inducing = refactoring_miner.merge(szz_fault_inducing_commits, left_on="COMMIT_HASH", right_on="FAULT_INDUCING_COMMIT_HASH")
    refactoring_fault_inducing = refactoring_fault_inducing.drop("COMMIT_HASH", axis=1)

    fault_inducing_commits = refactoring_fault_inducing.merge(git_commits, left_on="FAULT_INDUCING_COMMIT_HASH", right_on="COMMIT_HASH")
    fault_inducing_commits = fault_inducing_commits.drop("COMMIT_HASH", axis=1)

    fault_inducing_commits = fault_inducing_commits.merge(git_commits_changes, left_on="FAULT_INDUCING_COMMIT_HASH", right_on="COMMIT_HASH")
    fault_inducing_commits = fault_inducing_commits.drop("COMMIT_HASH", axis=1)



    return fault_inducing_commits



def preprocess_old(git_commits, szz_fault_inducing_commits, refactoring_miner):
    git_commits = git_commits[['PROJECT_ID','COMMIT_HASH','COMMIT_MESSAGE','COMMITTER_DATE','BRANCHES']]
    szz_fault_inducing_commits = szz_fault_inducing_commits[["FAULT_FIXING_COMMIT_HASH", "FAULT_INDUCING_COMMIT_HASH"]]
    refactoring_miner = refactoring_miner[["COMMIT_HASH", "REFACTORING_TYPE"]].drop_duplicates(ignore_index=True)
    # merge of dataframes
    
    refactoring_fault_inducing = refactoring_miner.merge(szz_fault_inducing_commits, left_on="COMMIT_HASH", right_on="FAULT_INDUCING_COMMIT_HASH")
    refactoring_fault_inducing = refactoring_fault_inducing.drop("COMMIT_HASH", axis=1)

    fault_fixing_commits = refactoring_fault_inducing.merge(git_commits, left_on="FAULT_FIXING_COMMIT_HASH", right_on="COMMIT_HASH")
    fault_fixing_commits = fault_fixing_commits.drop("COMMIT_HASH", axis=1)
    return fault_fixing_commits




'''
    first_commit = {}
    for proj in git_commits.PROJECT_ID.unique().tolist():
        if proj in ['org.apache:batik','org.apache:cocoon','org.apache:felix','org.apache:santuario']:
            pass
        else:
            git_commits.loc[git_commits.PROJECT_ID == proj, 'COMMITTER_DATE'] = git_commits['COMMITTER_DATE'].apply(lambda x: str(x)[:-6])
        first_commit[proj] = pd.to_datetime(git_commits[git_commits.PROJECT_ID == proj]['COMMITTER_DATE'].min()).date()
        git_commits.loc[git_commits.PROJECT_ID == proj,'commit_day'] = pd.to_datetime(git_commits.loc[git_commits.PROJECT_ID == proj,'COMMITTER_DATE'], dayfirst=True).dt.date - first_commit[proj]
    git_commits['commit_day'] = git_commits['commit_day'].apply(lambda x: x.days)
'''