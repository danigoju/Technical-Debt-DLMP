import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt 


def get_project_refactor(refactoring_miner):
    projects_refactor = {}

    for i in refactoring_miner.PROJECT_ID.unique():
        aux = {}
        for j in refactoring_miner.REFACTORING_TYPE.unique():
            faults = len(refactoring_miner[(refactoring_miner.PROJECT_ID==i) & (refactoring_miner.REFACTORING_TYPE==j)]["COMMIT_HASH"].unique())
            aux[j] = faults
        projects_refactor[i] = aux
    
    return projects_refactor
    
    
def get_project_faults(refactoring_miner, szz_fault_inducing_commits, clean_df):    
    projects_faults = {}

    for i in szz_fault_inducing_commits.PROJECT_ID.unique():
        aux = {}
        for j in refactoring_miner.REFACTORING_TYPE.unique():
            faults = len(clean_df[(clean_df.PROJECT_ID==i) & (clean_df.REFACTORING_TYPE==j)]["FAULT_INDUCING_COMMIT_HASH"].unique())
            aux[j] = faults
        projects_faults[i] = aux
        
    return projects_faults
    
    
def plot_refactor_faults(projects_refactor, projects_faults, project_id):    
    refact = pd.Series(pd.DataFrame(projects_refactor)[project_id], name="Refactor")
    faults = pd.Series(pd.DataFrame(projects_faults)[project_id], name="Faults")

    pd.concat([refact, faults], axis=1).plot(kind='bar', figsize=(18,7), rot=75, width=0.65, fontsize=17, color=('cornflowerblue','gold'),edgecolor = 'silver')
    plt.title(project_id, fontsize=20)
    plt.legend(fontsize=20)
    plt.show()
