from pyspark import pipelines as dp
from pyspark.sql import functions as sf
from pyspark.sql import SparkSession, DataFrame

SCHEMA = "workspace.sephora_products_and_skincare_reviews"
VOLUME_DATA = "/Volumes/workspace/sephora_products_and_skincare_reviews/data/source/"

TABLE_BRONZE_PRODUCTS_INFO = f"{SCHEMA}.3_bronze_products_info"
TABLE_BRONZE_REVIEWS = f"{SCHEMA}.3_bronze_reviews"
TABLE_SILVER_PRODUCTS_RECOMMENDATIONS = f"{SCHEMA}.3_silver_products_recommendations"
TABLE_GOLD_PRODUCTS_RECOMMENDATIONS = f"{SCHEMA}.3_gold_products_recommendations"

VECTOR_SEARCH_ENPOINT = "sephora"
VECTOR_SEARCH_INDEX_NAME = f"{SCHEMA}.3_gold_products_recommendations_index"

EMBEDDING_MODEL_ENDPOINT = "databricks-gte-large-en"

spark: SparkSession


@dp.table(name=TABLE_BRONZE_PRODUCTS_INFO)
def bronze_products_info() -> DataFrame:
    return (
        spark
        .read
        .csv(
            f"{VOLUME_DATA}/product_info.csv",
            inferSchema=True,
            header=True
        )
        .select(
            "*",
            "_metadata.file_modification_time",
            "_metadata.file_path"
        )
    )


@dp.table(name=TABLE_BRONZE_REVIEWS)
def bronze_reviews() -> DataFrame:
    return (
        spark
        .read
        .csv(
            f"{VOLUME_DATA}/reviews*.csv",
            inferSchema=True,
            header=True
        )
        .select(
            "*",
            "_metadata.file_modification_time",
            "_metadata.file_path"
        )
    )


@dp.table(name=TABLE_SILVER_PRODUCTS_RECOMMENDATIONS)
def silver_products_recommendations() -> DataFrame:
    return (
        spark
        .read
        .table(TABLE_BRONZE_PRODUCTS_INFO)
        .select("product_id", "product_name", "primary_category", "secondary_category", "tertiary_category", "highlights", )
            .join(
                spark
                .table(TABLE_BRONZE_REVIEWS)
                .select(
                    "product_id", "rating", "is_recommended", "review_text", "price_usd"
                ),
            on="product_id",
            how="inner"
        )
        .withColumn("id", sf.monotonically_increasing_id())
    )


@dp.table(name=TABLE_GOLD_PRODUCTS_RECOMMENDATIONS)
def gold_products_recommendations() -> DataFrame:
    df_product = (
        spark
        .read
        .table(TABLE_SILVER_PRODUCTS_RECOMMENDATIONS)
        .withColumn("rating", sf.col("rating").try_cast("double"))
        .groupBy(sf.col("product_id"))
        .agg(
            sf.count(sf.col("rating")).alias("cnt"),
            sf.avg(sf.col("rating")).alias("rating_avg"),
            sf.min(sf.col("rating")).alias("rating_min"),
            sf.max(sf.col("rating")).alias("rating_max"),
        )
        .filter(
            (sf.col("rating_min") == 1.0)
            & (sf.col("rating_max") == 5.0)
        )
        .withColumn(
            "diff",
            sf.abs(
                sf.lit(2.5)
                - sf.col("rating_avg")
            )
        )
        .orderBy(sf.col("diff"))
        .filter(sf.col("cnt").between(30, 50))
        .select("product_id")
        .limit(1)
    )

    return (
        spark
        .read
        .table(TABLE_SILVER_PRODUCTS_RECOMMENDATIONS)
        .join(
            df_product,
            on="product_id",
            how="inner"
        )
    )