"""Optional PySpark feature job (requires pip install pyspark and configured Java runtime)."""
import argparse

def main():
    from pyspark.sql import SparkSession,Window,functions as F
    a=argparse.ArgumentParser();a.add_argument('--input',default='data/loans.csv');a.add_argument('--output',default='artifacts/spark_features');v=a.parse_args()
    spark=SparkSession.builder.appName('risk-feature-engineering').getOrCreate()
    df=spark.read.option('header',True).option('inferSchema',True).csv(v.input)
    w=Window.partitionBy('customer_id').orderBy('month','loan_id').rowsBetween(-3,-1)
    result=df.withColumn('prior_utilization_mean_3',F.avg('utilization').over(w)).withColumn('prior_delinquency_sum_3',F.sum('delinquency_count').over(w))
    result.write.mode('overwrite').parquet(v.output)
    print('wrote',v.output);spark.stop()
if __name__=='__main__':main()
