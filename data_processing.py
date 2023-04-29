# This script will load in the raw data, analyze, process, and save out as another .csv file for the data mining task

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
import seaborn as sns

#### Data Exploration ####

# Load in the raw data
df = pd.read_csv("data/data_raw.csv")

# ydata profiling report (commmented out because it takes a long time to run)
# profile = ProfileReport(df, title='Data Profiling Report', explorative=True)
# profile.to_file('data/data_profiling.html')

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

# How many transactions have amount > 200000?
print("Number of transactions with amount > 200000: ", len(df[df["amount"] > 200000]))
# isFlaggedFraud only has 16 entries of the 1673570 entries with amount > 200000
# inconsistent with the description of isFlaggedFraud where it is supposed to be 1 if amount > 200000

# What are the transaction types (and respective counts) among the fraudulent examples?
print("Fraudulent transaction types: ", df[df["isFraud"] == 1]["type"].unique())
print(
    "Fraudulent transaction type counts: ",
    df[df["isFraud"] == 1]["type"].value_counts(),
)
# How many legitimate transactions are of type 'TRANSFER' or 'CASH_OUT'?
print(
    "Number of legitimate transactions of type TRANSFER: ",
    len(df[(df["isFraud"] == 0) & (df["type"] == "TRANSFER")]),
)
print(
    "Number of legitimate transactions of type CASH_OUT: ",
    len(df[(df["isFraud"] == 0) & (df["type"] == "CASH_OUT")]),
)

# So, in this P2P payment system, fraudulent transactions are only of type 'CASH_OUT' or 'TRANSFER'
# and they are roughly evenly split between the two types, so maybe there is a chain of activity that is happening
# among fraudsters in this system of sending money to then cash out
# Can we find a sequence of TRANSFER and CASH_OUT transactions that are fraudulent?
# i.e. a fraudulent TRANSFER transaction followed by a fraudulent CASH_OUT transaction where the recipient
# of the TRANSFER transaction is the originator of the CASH_OUT transaction?
df_fraud = df[df["isFraud"] == 1]
df_fraud_transfer = df_fraud[df_fraud["type"] == "TRANSFER"]
df_fraud_cashout = df_fraud[df_fraud["type"] == "CASH_OUT"]
df_fraud_transfer_cashout = pd.merge(
    df_fraud_transfer, df_fraud_cashout, left_on="nameDest", right_on="nameOrig"
)
# How many fraudulent TRANSFER transactions are followed by a fraudulent CASH_OUT transaction?
print(
    "Number of fraudulent TRANSFER transactions followed by a fraudulent CASH_OUT transaction: ",
    len(df_fraud_transfer_cashout),
)
# len of df_fraud_transfer_cashout is zero, indicating that there is no sequential chain of activity
# This is evidently due to a limitation of the algorithm used to generate this dataset

# From the data report, we see there are some originators who have made multiple transactions
# How many distinct originators have made a certain number of transactions?
nameOrig_counts = df["nameOrig"].value_counts()
nameOrig_counts_counts = nameOrig_counts.value_counts()
print("Number of originators who have made 1 transaction: ", nameOrig_counts_counts[1])
print("Number of originators who have made 2 transactions: ", nameOrig_counts_counts[2])
print("Number of originators who have made 3 transactions: ", nameOrig_counts_counts[3])
print(
    "Number of originators who have made more than 3 transactions: ",
    sum(nameOrig_counts_counts[4:]),
)
# Have any of the multiple transaction originators made fraudulent transactions? Subset transactions by originators
# who have made more than 1 transaction
df_multiple = df[df["nameOrig"].isin(nameOrig_counts[nameOrig_counts > 1].index)]
# How many of these transactions are fraudulent?
print(
    "Number of fraudulent transactions made by originators who have made more than 1 transaction: ",
    len(df_multiple[df_multiple["isFraud"] == 1]),
)  # not very many, 28/8213 (0.34%)
# Logically, the majority of fraudulent transactions are made by originators who have made only 1 transaction
# for example, a fraudster who creates a fake account to make a fraudulent transaction
# Perhaps there should be extra scrutiny on originators who have not made any transactions before

# The data report shows that there are many transactions where oldbalanceDest = 0 and newbalanceDest = 0, but amount is
# non-zero. This could be an indicator of fraud, so we will compare this scenario between fraudulent and
# legitimate transactions
df_fraud = df[df["isFraud"] == 1]
df_legit = df[df["isFraud"] == 0]
print(
    "Fraction of fraudulent transactions with oldbalanceDest = 0 and newbalanceDest = 0 but amount is non-zero: ",
    len(
        df_fraud.loc[
            (df_fraud.oldbalanceDest == 0)
            & (df_fraud.newbalanceDest == 0)
            & (df_fraud.amount != 0)
        ]
    )
    / len(df_fraud),
)
print(
    "Fraction of legitimate transactions with oldbalanceDest = 0 and newbalanceDest = 0 but amount is non-zero: ",
    len(
        df_legit.loc[
            (df_legit.oldbalanceDest == 0)
            & (df_legit.newbalanceDest == 0)
            & (df_legit.amount != 0)
        ]
    )
    / len(df_legit),
)
# uptick in fraudulent transactions with this scenario, so we will do some imputation to highlight this trend

####Data cleaning and Feature Engineering ####

# As shown earlier, isFlaggedFraud is not consistent and only is positive in 16 cases, so we will discard this column
df = df.drop(columns=["isFlaggedFraud"])
# Considering the previous two sections failed to demonstrate a chain of fraudulent activity using IDs,
# nameOrig and nameDest are not useful features for predicting fraud in the data so they will be discarded
df = df.drop(columns=["nameOrig", "nameDest"])
# Correcting the spelling of column headers for consistency
df = df.rename(
    columns={
        "oldbalanceOrg": "oldBalanceOrig",
        "newbalanceOrig": "newBalanceOrig",
        "oldbalanceDest": "oldBalanceDest",
        "newbalanceDest": "newBalanceDest",
    }
)
# Limiting the data records to only those with type = 'TRANSFER' or 'CASH_OUT' since these are the only
# types of transactions where fraud occurs
df = df[(df["type"] == "TRANSFER") | (df["type"] == "CASH_OUT")]
# Encoding the type column as a binary variable, 0 for 'TRANSFER' and 1 for 'CASH_OUT'
df["type"] = df["type"].map({"TRANSFER": 0, "CASH_OUT": 1})
# Performing imputation to highlight the uptick in fraudulent transactions with
# oldbalanceDest = 0 and newbalanceDest = 0 but amount is non-zero, with a value of -1 for both
df.loc[
    (df.oldBalanceDest == 0) & (df.newBalanceDest == 0) & (df.amount != 0),
    ["oldBalanceDest", "newBalanceDest"],
] = -1
# Imputing the cases where oldBalanceOrig = 0 and newBalanceOrig = 0 but amount is non-zero, with a null value
df.loc[
    (df.oldBalanceOrig == 0) & (df.newBalanceOrig == 0) & (df.amount != 0),
    ["oldBalanceOrig", "newBalanceOrig"],
] = np.nan
# With these assignments, we will engineer features measuring the transaction error comparing the amount and the
# old and new balances
# This may be useful for detecting fraud in an ML model
# These columns will be nonzero if there is a discrepancy between the amount and the old and new balances
df["errorBalanceOrig"] = df.newBalanceOrig + df.amount - df.oldBalanceOrig
df["errorBalanceDest"] = df.oldBalanceDest + df.amount - df.newBalanceDest

# 3d scatter plot showing a separation between fraudulent and legitimate transactions with the errorBalance features
zOffset = 0.02
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_title("Balance error features separate fraudulent and legitimate transactions")
ax.scatter(
    df[df["isFraud"] == 0]["errorBalanceDest"],
    df[df["isFraud"] == 0]["step"],
    -np.log10(df[df["isFraud"] == 0]["errorBalanceOrig"] + zOffset),
    c="b",
    marker=".",
    s=1,
    label="legitimate",
)
ax.scatter(
    df[df["isFraud"] == 1]["errorBalanceDest"],
    df[df["isFraud"] == 1]["step"],
    -np.log10(df[df["isFraud"] == 1]["errorBalanceOrig"] + zOffset),
    c="r",
    marker=".",
    s=1,
    label="fraudulent",
)
ax.set_xlabel("errorBalanceDest")
ax.set_ylabel("step[hour]")
ax.set_zlabel("-log$_{10}$(errorBalanceOrig)")
plt.legend(loc="upper left")
plt.savefig("plots/3d_scatter.png")

# Heatmaps showing difference between fraud/legit in feature correlations
corrFraud = df[df["isFraud"] == 1].drop(columns=["step", "isFraud"]).corr()
corrLegit = df[df["isFraud"] == 0].drop(columns=["step", "isFraud"]).corr()
mask = np.zeros_like(corrFraud)
mask[np.triu_indices_from(corrFraud)] = True
grid_kws = {"width_ratios": (0.9, 0.9, 0.05), "wspace": 0.2}
fig, (ax1, ax2, cbar_ax) = plt.subplots(1, 3, gridspec_kw=grid_kws, figsize=(14, 9))
sns.heatmap(
    corrFraud,
    mask=mask,
    ax=ax1,
    vmin=-1,
    vmax=1,
    cmap="coolwarm",
    square=False,
    linewidths=0.5,
    cbar=False,
)
ax1.set_title("Fraudulent\ntransactions")
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
ax1.set_yticklabels(ax1.get_yticklabels())
sns.heatmap(
    corrLegit,
    mask=mask,
    ax=ax2,
    vmin=-1,
    vmax=1,
    cmap="coolwarm",
    square=False,
    yticklabels=False,
    linewidths=0.5,
    cbar_ax=cbar_ax,
    cbar_kws={"orientation": "vertical", "ticks": [-1, -0.5, 0, 0.5, 1]},
)
ax2.set_title("Legitimate\ntransactions")
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
cbar_ax.set_yticklabels(cbar_ax.get_yticklabels(), size=14)
plt.savefig("plots/corr_heatmap.png")

# Save out the cleaned data
print("Examples in cleaned data: ", len(df))
print("Saving cleaned data to data/data_cleaned.csv")
df.to_csv("data/data_cleaned.csv", index=False)
