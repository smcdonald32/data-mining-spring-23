# This script will load in the raw data, analyze, process, and save out as another .csv file for the data mining task

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport


# Load in the raw data
df = pd.read_csv('data/data_raw.csv')

# ydata profiling report
profile = ProfileReport(df, title='Data Profiling Report', explorative=True)
profile.to_file('data/data_profiling.html')

# Data description/summary:
# 6362620 rows representing simulated P2P payment transactions over 30 days
# 6354407 valid transactions, 8213 fraudulent transactions (0.13% of txns are fraud; highly imbalanced dataset)
# 11 columns:
# 1. step: integer, 1-744 representing hour of transaction in 30 simulated days
# 2. type: categorical, 5 types of transactions
# 3. amount: float, amount of currency transacted
# 4. nameOrig: string, customer id who started the transaction
# 5. oldbalanceOrg: float, initial balance of originator before the transaction
# 6. newbalanceOrig: float, new balance of originator after the transaction posts (if declined, same as oldbalanceOrig)
# 7. nameDest: string, customer id who is the recipient of the transaction
# 8. oldbalanceDest: float, initial balance of recipient before the transaction
# 9. newbalanceDest: float, new balance of recipient after the transaction posts (if declined, same as oldbalanceDest)
# 10. isFraud: boolean, 1 if fraudulent transaction, 0 if valid transaction (target variable for classification)
# 11. isFlaggedFraud: boolean, 1 if transaction is flagged as fraudulent by P2P payment system, 0 if not
# Note: possibly discard this column as redundant since only 16 transactions are flagged as fraud out of 8213 fraud txns
# The rule for flagging is reportedly amount > 200000

# How many transactions have amount > 200000?
print('Number of transactions with amount > 200000: ', len(df[df['amount'] > 200000]))
# isFlaggedFraud only has 16 entries of the 1673570 entries with amount > 200000
# inconsistent with the description of isFlaggedFraud, so we will discard this column
df = df.drop(columns=['isFlaggedFraud'])
