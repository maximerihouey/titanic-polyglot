"""
Basic Spark classification for titanic dataset using mllib library (RDD)
Expected score: GBT = 0.755981, RF = 0.789474
"""

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext

# Setting SparkContext
conf = SparkConf().setAppName("titanic_DF").setMaster("local")
sc = SparkContext(conf=conf)
sqlc = SQLContext(sc)

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorIndexer, VectorAssembler, StandardScaler, QuantileDiscretizer
from pyspark.sql.functions import UserDefinedFunction
from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel, GBTClassifier, GBTClassificationModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
import pyspark.sql.functions as spark_fct
import pandas as pd

customSchema_fields = [
        StructField("PassengerId", IntegerType(), True),
        StructField("Survived", DoubleType(), True),
        StructField("Pclass", IntegerType(), True),
        StructField("Name", StringType(), True),
        StructField("Sex", StringType(), True),
        StructField("Age", DoubleType(), True),
        StructField("SibSp", IntegerType(), True),
        StructField("Parch", IntegerType(), True),
        StructField("Ticket", StringType(), True),
        StructField("Fare", DoubleType(), True),
        StructField("Cabin", StringType(), True),
        StructField("Embarked", StringType(), True)
    ]

train_df = sqlc.read.load(
    path="data/train.csv",
    format="com.databricks.spark.csv",
    header=True,
    schema=StructType(customSchema_fields)
)

test_df = sqlc.read.load(
    path="data/test.csv",
    format="com.databricks.spark.csv",
    header=True,
    schema=StructType(customSchema_fields[:1] + customSchema_fields[2:])
)

################################################### Handling NAs
def blank_as_null(x):
    return spark_fct.when(x != "", x).otherwise(None)

def blank_as_value(x, value):
    return spark_fct.when(x != "", x).otherwise(value)

######################### Embarked
train_df = train_df.withColumn("Embarked", blank_as_value(train_df.Embarked, "S"))
test_df = test_df.withColumn("Embarked", blank_as_value(test_df.Embarked, "S"))

######################### Age
def null_as_value(x, value):
    return spark_fct.when(~x.isNull(), x).otherwise(value)

#### Fill age NAs with mean
mean_age = train_df.select("Age").unionAll(test_df.select("Age")).select(spark_fct.mean("Age")).head()[0]
train_df = train_df.withColumn("Age", null_as_value(train_df.Age, mean_age))
test_df = test_df.withColumn("Age", null_as_value(test_df.Age, mean_age))

######################### Fare
mean_fare = train_df.select("Fare").unionAll(test_df.select("Fare")).select(spark_fct.mean("Fare")).head()[0]
test_df = test_df.withColumn("Fare", null_as_value(test_df.Fare, mean_fare))

################################################### Encoding Strings
sex_stringIndexer = StringIndexer(inputCol="Sex", outputCol="sex_code").fit(train_df.select("Sex").unionAll(test_df.select("Sex")))
embarked_stringIndexer = StringIndexer(inputCol="Embarked", outputCol="embarked_code").fit(train_df.select("Embarked").unionAll(test_df.select("Embarked")))
embarked_encoder = OneHotEncoder(inputCol="embarked_code", outputCol="embarked_coded")
age_discretizer = QuantileDiscretizer(numBuckets=3, inputCol="Age", outputCol="age_discretized").fit(train_df.select("Age").unionAll(test_df.select("Age")))
fare_discretizer = QuantileDiscretizer(numBuckets=3, inputCol="Fare", outputCol="fare_discretized").fit(train_df.select("Fare").unionAll(test_df.select("Fare")))

features_column = ["sex_code", "embarked_coded", "Pclass", "SibSp", "Age", "age_discretized", "Parch", "fare_discretized"]
VectorAssembler = VectorAssembler(inputCols=features_column, outputCol="features")

############ Classifiers
rfC = RandomForestClassifier(labelCol="Survived", featuresCol="features", numTrees=300, maxDepth=5)
gbtC = GBTClassifier(labelCol="Survived", featuresCol="features", maxIter=50)

pipeline = Pipeline().setStages([
        sex_stringIndexer,
        age_discretizer,
        fare_discretizer,
        embarked_stringIndexer, embarked_encoder,
        VectorAssembler,
        rfC
    ]).fit(train_df)

##### Applying pipeline
train_piped = pipeline.transform(train_df)
test_piped = pipeline.transform(test_df)

############################################### Feature importances
print("\n----------- Feature importances")
rfCmodel = pipeline.stages[6]
for feature_name, feature_importance in sorted(zip(features_column, rfCmodel.featureImportances), key=lambda x: -x[1]):
    print("%20s: %s" % (feature_name, feature_importance))

############################################## Exporting
df_predictions = test_piped.select("prediction").toPandas().reset_index()
df_predictions['index'] = df_predictions['index'] + 892
df_predictions.columns = ['PassengerId', 'Survived']

df_predictions.to_csv('submission.csv', index=False)
