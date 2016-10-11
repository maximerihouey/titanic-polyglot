"""
Created on Tue Sep 20 16:31:28 2016

Basis for Scikit-learn scripts using titanic dataset
Examples of basic feature extraction and engineering
Expected accuracy: 0.784689

@author: maxime.rihouey
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

np.random.seed(1)

#
# functions for feature extraction
#
def extracting_title(x):
    title = x.split(",")[1].split(".")[0][1:]
    if title in ['Mr', 'Miss', 'Mrs', 'Master', 'Dr', 'Rev', 'Col']:
        return title
    return "Special_title"


def extract_ticket_number(x):
    ticket_fields = x.split(" ")
    try:
        number = int(ticket_fields[-1])
        return number
    except:
        return 0

#
# Retrieving data
#
train_df = pd.read_csv("data/train.csv", header=0, index_col=False)
test_df = pd.read_csv("data/test.csv", header=0, index_col=False)

#
# Building full dataset for feature engineering purposes
#
considered_features = [
    "PassengerId", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"
]
train_df, train_survived = train_df[considered_features], train_df["Survived"]
test_df = test_df[considered_features]

total_df = pd.concat([train_df, test_df], axis=0)
total_df = total_df.reset_index()

#
# Feature engineering
#
# New features
total_df['title'] = total_df['Name'].apply(extracting_title).astype("category")
total_df['ticket_number'] = total_df["Ticket"].map(extract_ticket_number)

# Continuus features
continuus_features = ["Pclass", "Age", "Fare", "SibSp", "Parch"]
for continuus_feature in continuus_features:
    if sum(total_df[continuus_feature].isnull()) > 0:
        # using the median for nans
        total_df.loc[total_df[continuus_feature].isnull(), continuus_feature] = np.nanmedian(total_df[continuus_feature])

categorical_features = ["Sex", "title", "Embarked"]
categorical_features_dummies = []
for categorical_feature in categorical_features:
    if sum(total_df[categorical_feature].isnull()) > 0:
        # using the most frequent categpry for nans
        total_df.loc[total_df[categorical_feature].isnull(), categorical_feature] = total_df[categorical_feature].value_counts().idxmax()

    # using dummies and appending the new created columns to the list of categorical_features_dummies
    dummies = pd.get_dummies(total_df[categorical_feature], prefix=categorical_feature)
    categorical_features_dummies += list(dummies.columns)
    total_df = pd.concat([total_df, dummies.astype(bool)], axis=1)

# Combined features
# family size
total_df['family_size'] = total_df['SibSp'] + total_df['Parch']

# Discretized features
# age discretization
total_df['age_cat'] = pd.qcut(total_df["Age"], 3).cat.codes

#
# Splitting the two datasets
#
train_cutoff = train_df.shape[0]
indices = np.arange(total_df.shape[0])
train_df, test_df = total_df.take(indices[:train_cutoff]), total_df.take(indices[train_cutoff:])

# features
train_features = ["family_size", "age_cat"] + continuus_features + categorical_features_dummies

# Establishing train and test datasets
X_train = train_df[train_features]
y_train = train_survived
X_test = test_df[train_features]

# Predictions
classifier = RandomForestClassifier(n_estimators=150, max_features=3, max_depth=3).fit(X_train, y_train)
prediction = classifier.predict(X_test)

# Exporting the results
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": prediction
    })
submission.to_csv("submission.csv", index=False)
