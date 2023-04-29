import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from xgboost import plot_importance

# load the data
df = pd.read_csv("data/data_cleaned.csv")

# heavy class imbalance ratio
print(df["isFraud"].value_counts(normalize=True))
# Considering the scale of the data, rather than over/under sampling and distorting information in the data,
# I will use stratified split to preserve the class ratio in the train/test split
# Classification will be done using ensemble methods that are robust to class imbalance
# Evaluation of the model will be done using average precision score (summarizes precision-recall curve)
# instead of AUROC due to the class imbalance
# split the data
randomState = 42
np.random.seed(randomState)
x_train, x_test, y_train, y_test = train_test_split(
    df.drop("isFraud", axis=1),
    df["isFraud"],
    test_size=0.2,
    random_state=randomState,
    stratify=df["isFraud"],
)

# parameter grid search for xgboost
weights = (df["isFraud"] == 0).sum() / (
    df["isFraud"] == 1
).sum()  # Correcting for class imbalance
param_grid = {
    "max_depth": [2, 3, 4],  # max depth of tree
    "learning_rate": [0.1, 0.01],  # eta, step size shrinkage
    "gamma": [0, 1.0],  # min loss reduction to create new tree split
    "reg_lambda": [0, 1.0, 10.0],  # L2 regularization
}
xgb = XGBClassifier(random_state=randomState, n_jobs=-1, scale_pos_weight=weights)
# Using a 3-fold cross-validation to find the best parameter setting for gradient boosting
grid_search = GridSearchCV(
    estimator=xgb, param_grid=param_grid, scoring="average_precision", cv=3, verbose=1
)
grid_search.fit(x_train, y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)
# best parameters: gamma=0, learning_rate=0.1, max_depth=2, reg_lambda=10.0
# Average precision score: 0.9972
best_params = grid_search.best_params_

# train the model with the best parameters
xgb = XGBClassifier(
    random_state=randomState, n_jobs=-1, scale_pos_weight=weights, **best_params
)
xgb.fit(x_train, y_train)
# Evaluate the model with the holdout set
y_pred = xgb.predict_proba(x_test)[:, 1]
average_precision = average_precision_score(y_test, y_pred)
print(f"Average precision-recall score for XGBoost: {average_precision:0.4f}")
# Average precision-recall score for XGBoost: 0.9980
# Plot the precision-recall curve
precision, recall, _ = precision_recall_curve(y_test, y_pred)
plt.step(recall, precision, color="b", alpha=0.2, where="post")
plt.fill_between(recall, precision, step="post", alpha=0.2, color="b")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(f"XGBoost Precision-Recall curve: AP={average_precision:.4f}")
plt.savefig("plots/xgb_precision_recall.png")

# Variable Importance Plot
fig = plt.figure(figsize=(14, 9))
ax = fig.add_subplot(111)
ax = plot_importance(
    xgb,
    height=1,
    ax=ax,
    color="g",
    grid=False,
    show_values=False,
    importance_type="cover",
)
for axis in ["top", "bottom", "left", "right"]:
    ax.spines[axis].set_linewidth(2)
ax.set_xlabel("importance score", size=16)
ax.set_ylabel("features", size=16)
ax.set_yticklabels(ax.get_yticklabels(), size=12)
ax.set_title("Ordering of features by importance", size=20)
plt.savefig("plots/xgb_feature_importance.png")
