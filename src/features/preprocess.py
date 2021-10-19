import pandas as pd
import numpy as np

def preprocess(git_commits, szz_fault_inducing_commits, refactoring_miner, git_commits_changes):
	'''
	
	'''
	# keep columns and rows of interest of each dataframe individually
	git_commits = git_commits[['PROJECT_ID','COMMIT_HASH','COMMIT_MESSAGE','COMMITTER_DATE','BRANCHES']]
	szz_fault_inducing_commits = szz_fault_inducing_commits[["FAULT_FIXING_COMMIT_HASH", "FAULT_INDUCING_COMMIT_HASH"]]
	refactoring_miner = refactoring_miner[["COMMIT_HASH", "REFACTORING_TYPE"]].drop_duplicates(ignore_index=True)
	git_commits_changes = git_commits_changes[["COMMIT_HASH", "COMMITTER_ID","LINES_ADDED","LINES_REMOVED"]].drop_duplicates(ignore_index=True)

	# merge of dataframes
	
	refactoring_fault_inducing = refactoring_miner.merge(szz_fault_inducing_commits, left_on="COMMIT_HASH", right_on="FAULT_INDUCING_COMMIT_HASH")
	refactoring_fault_inducing = refactoring_fault_inducing.drop("COMMIT_HASH", axis=1)

	fault_fixing_commits = refactoring_fault_inducing.merge(git_commits, left_on="FAULT_FIXING_COMMIT_HASH", right_on="COMMIT_HASH")
	fault_fixing_commits = fault_fixing_commits.drop("COMMIT_HASH", axis=1)
    
	fault_fixing_commits = fault_fixing_commits.merge(git_commits_changes, left_on="FAULT_FIXING_COMMIT_HASH", right_on="COMMIT_HASH")
	fault_fixing_commits = fault_fixing_commits.drop("COMMIT_HASH", axis=1)


	return fault_fixing_commits

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