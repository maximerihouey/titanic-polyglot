"""
Basic Spark classification for titanic dataset using mllib library (RDD)
Expected score: GBT = 0.787081339713, RF = 0.784688995215
"""

from pyspark import SparkContext, SparkConf

# Setting SparkContext
conf = SparkConf().setAppName("titanic_RDD").setMaster("local")
sc = SparkContext(conf=conf)

from pyspark.mllib.linalg import DenseVector
from StringIO import StringIO
import csv
from pyspark.mllib.stat import Statistics
from pyspark.mllib.feature import LabeledPoint
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
import pandas as pd
import operator

########## TRAIN DATA

######################### Values for nans
def raw_parsing(line):
    values = list(csv.reader(StringIO(line)))[0]
    return values

train_textFile = sc.textFile('data/train.csv')
header_train = train_textFile.take(1)[0]
train_raw = train_textFile.filter(lambda line: line != header_train).map(raw_parsing)

all_ages = train_raw.map(lambda x: x[5])
mean_age = all_ages.filter(lambda x: x != '').map(lambda x: float(x)).stats().mean()
all_fares = train_raw.map(lambda x: x[9])
mean_fare = all_fares.map(lambda x: float(x)).stats().mean()
most_frequent_embarked = sorted(train_raw.map(lambda x: x[11]).filter(lambda x: x!='').countByKey().items(), key=operator.itemgetter(1))[-1][0]

######################## Reparsing data and filling nans
def parsePoint(is_train):
    def fonction(line):

        # Parsing csv line
        values = list(csv.reader(StringIO(line)))[0]
        if is_train:
            PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked = values
        else:
            PassengerId, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked = values

        # Sex
        sex_code = 0 if Sex == "male" else 1

        # Age
        safe_age = float(Age) if Age.isdigit() else mean_age
        try:
            safe_fare = float(Fare)
        except:
            safe_fare = mean_fare

        # Embarked
        safe_embarked = Embarked if Embarked else most_frequent_embarked
        code_embarked = 0 if safe_embarked == 'S' else 1 if safe_embarked == 'C' else 2 # == 'Q'

        # Title
        title = Name.split(",")[1].split(".")[0]
        code_title = 0
        if title == "Mr":
            code_title = 1
        elif title == "Miss":
            code_title = 2
        elif title == "Mrs":
            code_title = 3
        elif title == "Master":
            code_title = 4

        # Pclass code
        code_pclass = int(Pclass) - 1

        # child
        child_flag = 1 if safe_age <= 6 else 0

        features_vector = DenseVector([
                sex_code, safe_age, code_pclass, int(Parch), int(SibSp), safe_fare, code_embarked, code_title, child_flag, int(Parch)
            ])

        if is_train:
            return LabeledPoint(float(Survived), features_vector)
        else:
            return features_vector

    return fonction

# Feature engineering on train
header_train = train_textFile.take(1)[0]
train_data = train_textFile.filter(lambda line: line != header_train).map(parsePoint(True))
train_labels = train_data.map(lambda x: x.label)
train_features = train_data.map(lambda x: x.features)

# Statistics
colStats_train = Statistics.colStats(train_features)

################################ Model fitting
models = {
        "RF": RandomForest.trainClassifier(
            train_data, numClasses=2, categoricalFeaturesInfo={0:2, 2:3, 6: 3, 7: 5, 8:2}, numTrees=150,
            featureSubsetStrategy="auto", impurity='gini', maxDepth=6
        ),
        "GBT": GradientBoostedTrees.trainClassifier(
            train_data, categoricalFeaturesInfo={0:2, 6: 3, 7: 5, 8:2}, numIterations=50
        )
    }
model = models["GBT"]

############################## Predictions on Train
predictions = model.predict(train_features)
labelsAndPredictions = train_labels.zip(predictions)

testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(train_data.count())
print('Train Error = ' + str(testErr))


########## TEST DATA
test_textFile = sc.textFile('data/test.csv')
header_test = test_textFile.take(1)[0]
test_data = test_textFile.filter(lambda line: line != header_test).map(parsePoint(False))

predictions_test = model.predict(test_data)


########## Exporting predictions
pd.DataFrame(
    zip(range(892,1310), predictions_test.map(int).collect()),
    columns=['PassengerId','Survived']
).to_csv('submission.csv', index=False)
