"""
Created on Tue Sep 20 16:31:28 2016

Basis for Scikit-learn scripts using titanic dataset
Examples of basic feature extraction and engineering
Expected accuracy: ~ 0.76

@author: maxime.rihouey
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import Normalizer

from pipeline_transformers import PandasSelector, PandasFillNas, PandasToDict

np.random.seed(1)

#
# Retrieving data
#
train_df = pd.read_csv("data/train.csv", header=0, index_col=False)
y_train = train_df["Survived"]
test_df = pd.read_csv("data/test.csv", header=0, index_col=False)

#
# Setting up pipeline
#
processing_pipeline = make_pipeline(
    
    PandasSelector(columns=["PassengerId", "Survived", "Pclass", "Name", "Cabin", "Ticket"], inverse=True),
    
    make_union(
        make_pipeline(
            # Numerical data
            PandasSelector(dtype="O", inverse=True),
            Imputer(strategy='median'),
        ),
        make_pipeline(
            # Categorical data & dates
            PandasSelector(dtype="O"),
#            Imputer(strategy="most_frequent"),
            PandasFillNas(-1),
            PandasToDict(),
            DictVectorizer(sparse=False)
        ),
    ),
)

#
# Fitting pipeline & predicting
#
prediction_pipeline = make_pipeline(processing_pipeline, RandomForestClassifier(n_estimators=300))

prediction_pipeline.fit(train_df, y_train)
prediction = prediction_pipeline.predict(test_df)

#
# Exporting the results
#
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": prediction
    })
submission.to_csv("submission.csv", index=False)

