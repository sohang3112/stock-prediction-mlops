"""
Baseline PySpark ML training script for NIFTY minute data.

Usage examples:
  python train_pyspark_baseline.py --input data/raw --model-out model_rf

This script will:
 - load CSV(s) from the `--input` path (header expected)
 - parse datetime, create lag and rolling features
 - create label = next-minute `close`
 - train a RandomForestRegressor baseline
 - evaluate (RMSE, MAE, R2)
 - save the trained pipeline model
 - show a small example of predicting a next-minute close for a single incoming row
"""

# Spark UI opens at http://localhost:4040/jobs/

# (pyspark) WARN WindowExec: No Partition Defined for Window operation! Moving all data to a single partition, this can cause serious performance degradation.
# (pyspark) Not enough space to cache rdd_100_0 in memory! (computed 318.1 MiB so far)

import mlflow
import mlflow.spark
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F


def create_spark(app_name="nifty_baseline"):
    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.default.parallelism", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_data(spark):
    df = (
        spark.read.option("header", True)
        .option("inferSchema", False)
        .csv("data/raw/NIFTY 100_minute.csv")
    )

    # Parse datetime to timestamp
    df = df.withColumn("ts", F.to_timestamp(F.col("date"), "yyyy-MM-dd HH:mm:ss"))

    # Cast numeric columns
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df = df.withColumn(col, F.col(col).cast("double"))

    # Keep only rows with non-null ts and close
    df = df.filter(F.col("ts").isNotNull() & F.col("close").isNotNull())

    # Sort by timestamp
    df = df.sort(F.col("ts"))
    return df


def featurize(df):
    # Window ordered by timestamp.
    win = Window.partitionBy(F.lit("all")).orderBy("ts")

    # Lags (previous minute values)
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df = df.withColumn(f"lag_{c}", F.lag(F.col(c), 1).over(win))

    # Stationary Features: Differences between current and previous values
    for c in ["open", "high", "low", "close", "volume"]:
        if f"lag_{c}" in df.columns:
            df = df.withColumn(f"diff_{c}", F.col(c) - F.col(f"lag_{c}"))

    # Rolling features: 5- and 10- minute averages of close
    for window_size in (5, 10):
        ma_col = f"ma_close_{window_size}"
        df = df.withColumn(
            ma_col,
            F.avg(F.col("close")).over(
                Window.partitionBy(F.lit("all"))
                .orderBy("ts")
                .rowsBetween(-window_size + 1, 0)
            ),
        )
        # Stationary feature: Difference from moving average
        df = df.withColumn(f"diff_ma_{window_size}", F.col("close") - F.col(ma_col))

    # Simple returns based on previous close (already stationary)
    df = df.withColumn(
        "ret_prev", (F.col("close") - F.col("lag_close")) / F.col("lag_close")
    )

    # Label: next minute price CHANGE (lead(close) - current close)
    # We also keep lead(close) as 'label_original' for evaluation
    df = df.withColumn("label_original", F.lead(F.col("close"), 1).over(win))
    df = df.withColumn("label", F.col("label_original") - F.col("close"))

    # Drop rows with nulls (arising from lags, rolling windows, or label)
    df = df.na.drop(
        subset=["lag_close", "lag_open", "lag_high", "lag_low", "label", "diff_close"]
    )

    # Feature list: Use differences and stationary metrics
    feature_cols = [
        "diff_open",
        "diff_high",
        "diff_low",
        "diff_close",
        "diff_volume",
        "diff_ma_5",
        "diff_ma_10",
        "ret_prev",
    ]

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_assembled")
    scaler = StandardScaler(
        inputCol="features_assembled",
        outputCol="features",
        withStd=True,
        withMean=False,
    )

    return df, assembler, scaler


def time_split(df, train_fraction=0.8):
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")

    # compute cutoff timestamp
    bounds = df.agg(F.min("ts").alias("min_ts"), F.max("ts").alias("max_ts")).collect()[
        0
    ]
    min_ts = bounds["min_ts"].timestamp()
    max_ts = bounds["max_ts"].timestamp()
    cutoff = min_ts + train_fraction * (max_ts - min_ts)
    cutoff_ts = F.from_unixtime(F.lit(int(cutoff))).cast("timestamp")

    train = df.filter(F.col("ts") <= cutoff_ts)
    test = df.filter(F.col("ts") > cutoff_ts)
    return train, test


def train_and_evaluate(train_df, test_df, assembler, scaler):
    model_out = "data/output/"

    num_trees = 50
    max_depth = 10
    feature_subset_strategy = "sqrt"
    subsampling_rate = 0.7

    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="label",
        predictionCol="prediction",
        maxDepth=max_depth,
        numTrees=num_trees,
        featureSubsetStrategy=feature_subset_strategy,
        subsamplingRate=subsampling_rate,
    )
    pipeline = Pipeline(stages=[assembler, scaler, rf])

    mlflow.log_params(
        {
            "model_type": "RandomForestRegressor",
            "num_trees": num_trees,
            "max_depth": max_depth,
            "feature_subset_strategy": feature_subset_strategy,
            "subsampling_rate": subsampling_rate,
            "train_fraction": 0.8,
            "features": assembler.getInputCols(),
        }
    )

    model = pipeline.fit(train_df)

    # Evaluate on test
    test_rows = test_df.count()
    if test_rows == 0:
        raise ValueError(
            "The train/test split produced no test rows. Increase the dataset size or adjust the split fraction."
        )

    preds = model.transform(test_df)

    # Reconstruct absolute price prediction: current close + predicted change
    preds = preds.withColumn("final_prediction", F.col("close") + F.col("prediction"))

    # Evaluate on the absolute price (label_original vs final_prediction)
    evaluator_rmse = RegressionEvaluator(
        labelCol="label_original", predictionCol="final_prediction", metricName="rmse"
    )
    evaluator_mae = RegressionEvaluator(
        labelCol="label_original", predictionCol="final_prediction", metricName="mae"
    )
    evaluator_r2 = RegressionEvaluator(
        labelCol="label_original", predictionCol="final_prediction", metricName="r2"
    )

    rmse = evaluator_rmse.evaluate(preds)
    mae = evaluator_mae.evaluate(preds)
    r2 = evaluator_r2.evaluate(preds)

    mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})

    print(f"Test RMSE: {rmse:.6f}")
    print(f"Test MAE: {mae:.6f}")
    print(f"Test R2: {r2:.6f}")

    mlflow.spark.log_model(model, artifact_path="model")
    print("Model logged to MLflow")

    # Save pipeline model
    model.write().overwrite().save(model_out)
    print(f"Saved model pipeline to: {model_out}")

    return model


def predict_single(model, prev_row: dict, current_row: dict, spark):
    # Build a single-row DataFrame with the features expected by the pipeline.
    # prev_row and current_row are dict-like providing the raw OHLCV and ts if available.
    data = {}

    # Current values
    for c in ["open", "high", "low", "close", "volume"]:
        data[c] = float(current_row.get(c, 0.0))

    # Lags come from prev_row
    for c in ["open", "high", "low", "close", "volume"]:
        data[f"lag_{c}"] = (
            float(prev_row.get(c, data[c]))
            if prev_row.get(c, None) is not None
            else data[c]
        )

    # diff_* features
    for c in ["open", "high", "low", "close", "volume"]:
        data[f"diff_{c}"] = data[c] - data[f"lag_{c}"]

    # For MA diffs in single-row prediction, we approximate with the current row
    # (In a real system, we would pass a window of historical rows)
    data["diff_ma_5"] = 0.0
    data["diff_ma_10"] = 0.0

    # ret_prev
    if data.get("lag_close") != 0:
        data["ret_prev"] = (data["close"] - data["lag_close"]) / data["lag_close"]
    else:
        data["ret_prev"] = 0.0

    # Create Spark DataFrame
    row_df = spark.createDataFrame([data])

    # Predict the CHANGE
    pred_change = model.transform(row_df).select("prediction").collect()[0][0]

    # Reconstruct the absolute price
    final_pred = data["close"] + pred_change
    return final_pred


def main():
    spark = create_spark()

    mlflow.set_experiment("nifty-stock-prediction")

    df = load_data(spark)
    df, assembler, scaler = featurize(df)

    train_df, test_df = time_split(df, train_fraction=0.8)

    with mlflow.start_run(run_name="rf_baseline"):
        train_rows = train_df.count()
        test_rows = test_df.count()
        print(f"Training rows: {train_rows}, Test rows: {test_rows}")
        mlflow.log_metrics({"train_rows": train_rows, "test_rows": test_rows})

        model = train_and_evaluate(train_df, test_df, assembler, scaler)

        # Demonstrate single-row prediction using the last known row as previous
        last_two = df.sort(F.col("ts").desc()).limit(2).collect()
        if len(last_two) >= 2:
            current = {
                c: last_two[0][c]
                for c in ["open", "high", "low", "close", "volume"]
                if c in df.columns
            }
            prev = {
                c: last_two[1][c]
                for c in ["open", "high", "low", "close", "volume"]
                if c in df.columns
            }
            pred = predict_single(model, prev, current, spark)
            print(
                f"Example predicted next-minute close (from the latest available row): {pred:.4f}"
            )
        else:
            print("Not enough rows to demonstrate single-row prediction example.")

        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()

# Test RMSE: 3097.647321
# Test MAE: 2836.951832
# Test R2: -5.196523   <---- negative R^2 score is very bad!

# Example predicted next-minute close (from the latest available row): 22118.7324
