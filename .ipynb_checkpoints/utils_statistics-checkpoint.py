import pandas as pd
import numpy as np
import math
from scipy import stats
import logging
import scikit_posthocs as sp
from statsmodels.sandbox.stats.multicomp import multipletests

def mann_whitney_u_test(group1, group2, alternative='two-sided'):
    # citation for effect size interpretation
    # Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences. New York, NY: Routledge Academic.  
    # small  : 0.2 - 0.5
    # medium : 0.5 – 0.8
    # large  : >0.8
    
    # Mann-Whitney U Test
    stat, p_value = stats.mannwhitneyu(group1, group2, alternative=alternative)
    
    # sample sizes
    n1 = len(group1)
    n2 = len(group2)
    
    # cohen's d for Mann-Whitney U
    cohens_d = (stat / (n1 * n2)) - 0.5
    
    return {'pvalue': p_value, 'cohens_d': cohens_d}

