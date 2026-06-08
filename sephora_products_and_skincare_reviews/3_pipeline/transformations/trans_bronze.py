from pyspark import pipelines as dp
from pyspark.sql import functions as sf
from pyspark.sql import SparkSession

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
def bronze_products_info():
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
def bronze_reviews():
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