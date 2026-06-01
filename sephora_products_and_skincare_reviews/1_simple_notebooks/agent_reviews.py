import mlflow
from sephora_products_and_skincare_reviews.databricks.test_projects.agent import langchain_responses_agent_fact

agent =  langchain_responses_agent_fact('agent_config_reviews.yaml')()
mlflow.models.set_model(agent)