# %%
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Initialize Spark Session
spark = SparkSession.builder.appName("TeslaStockPrediction").getOrCreate()

# Load CSV Dataset
df = spark.read.csv("data/raw/NIFTY 100_minute.csv", header=True, inferSchema=True)

# Inspect Data
df.show(5)
print(f"Shape: ({df.count()}, {len(df.columns)})")
df.describe().show()
df.printSchema()


# %%

# Drop redundant column
df = df.drop("Adj Close")

# Check for Null Values
df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]).show()

# BUG: Date column actually has Datetime
# Extract Day, Month, Year from 'Date' (assumes MM/DD/YYYY format)
split_col = F.split(F.col("Date"), "/")
df = (
    df.withColumn("month", split_col.getItem(0).cast("int"))
    .withColumn("day", split_col.getItem(1).cast("int"))
    .withColumn("year", split_col.getItem(2).cast("int"))
)

# Add 'is_quarter_end'
df = df.withColumn("is_quarter_end", F.when(F.col("month") % 3 == 0, 1).otherwise(0))

# Derived Features
df = df.withColumn("open-close", F.col("Open") - F.col("Close")).withColumn(
    "low-high", F.col("Low") - F.col("High")
)

# Create 'target' (1 if next day's Close is higher, else 0) using Window functions
# Note: Ensure ordering by row or date index before applying lead
windowSpec = Window.orderBy(F.monotonically_increasing_id())
df = df.withColumn("next_close", F.lead("Close", 1).over(windowSpec))
df = df.withColumn(
    "target", F.when(F.col("next_close") > F.col("Close"), 1).otherwise(0)
)

# Drop rows where target is null (the last row due to lead shift)
df = df.dropna(subset=["target"])

# %%

# PySpark ML requires input features to be combined into a single `Vector` column before scaling and modeling.

from pyspark.ml.feature import StandardScaler, VectorAssembler

feature_cols = ["open-close", "low-high", "is_quarter_end"]

# Assemble features into a single vector column
assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
assembled_df = assembler.transform(df)

# Scale features
scaler = StandardScaler(
    inputCol="raw_features", outputCol="features", withStd=True, withMean=True
)
scaler_model = scaler.fit(assembled_df)
scaled_df = scaler_model.transform(assembled_df)

### 4. Data Splitting & Model Evaluation

from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Train / Validation Split (90/10)
train_df, valid_df = scaled_df.randomSplit([0.9, 0.1], seed=2022)

# Models to evaluate
models = {
    "LogisticRegression": LogisticRegression(featuresCol="features", labelCol="target"),
    "RandomForest": RandomForestClassifier(featuresCol="features", labelCol="target"),
}

evaluator = BinaryClassificationEvaluator(
    labelCol="target", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
)

# Train and evaluate models
for name, model in models.items():
    trained_model = model.fit(train_df)

    train_preds = trained_model.transform(train_df)
    valid_preds = trained_model.transform(valid_df)

    train_auc = evaluator.evaluate(train_preds)
    valid_auc = evaluator.evaluate(valid_preds)

    print(f"=== {name} ===")
    print(f"Training ROC-AUC  : {train_auc:.4f}")
    print(f"Validation ROC-AUC: {valid_auc:.4f}\n")
# %%
