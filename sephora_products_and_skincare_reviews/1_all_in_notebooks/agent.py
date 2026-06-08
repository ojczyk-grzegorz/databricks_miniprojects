import os
from uuid import uuid4
from typing import Any

import yaml
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

from langchain.agents import create_agent
from databricks_langchain import ChatDatabricks, VectorSearchRetrieverTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph


def _load_config(path: str = "agent_config.yaml") -> dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at path: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    
    llm_endpoint_name = cfg["llm_endpoint_name"]
    
    vs: dict = cfg["vector_search"]
    index_name = vs["index_name"]
    num_results = int(vs.get("num_results", 5))
    
    return {
        "llm_endpoint_name": llm_endpoint_name,
        "vs_index_name": vs["index_name"],
        "vs_num_results": int(vs.get("num_results", 5)),
        "tool_description": cfg["tool_description"],
        "tool_name": cfg["tool_name"],
        "system_prompt": cfg["system_prompt"]
    }


def build_agent(llm_endpoint: str, index_name: str, system_prompt: str, tool_name: str, tool_description: str, num_results: int = 3) -> CompiledStateGraph:
    model = ChatDatabricks(
        endpoint=llm_endpoint,
        max_tokens=500
    )

    vs_tool = VectorSearchRetrieverTool(
        index_name=index_name,
        tool_name=tool_name,
        tool_description=tool_description,
        num_results=num_results,
    )

    checkpointer = InMemorySaver()

    agent = create_agent(
        model=model,
        tools=[vs_tool],
        system_prompt=system_prompt,
        checkpointer=checkpointer,
    )

    return agent


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    user_msgs = [m for m in messages if (m.get("role") == "user")]
    if user_msgs:
        return user_msgs[-1].get("content", "")
    elif messages:
        return str(messages[-1].get("content", ""))
    return ""


def langchain_responses_agent_fact(config_file: str) -> type:
    class LangChainResponsesAgent(ResponsesAgent):
        def __init__(self, *args):
            cfg = _load_config(config_file)
            self._cfg = cfg
            self._agent = build_agent(
                llm_endpoint=cfg["llm_endpoint_name"],
                index_name=cfg["vs_index_name"],
                tool_name=cfg["tool_description"],
                tool_description=cfg["tool_description"],
                num_results=cfg["vs_num_results"],
                system_prompt=cfg["system_prompt"],
            )

        def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
            msgs = [m.model_dump() for m in request.input]
            if not msgs:
                raise ValueError("No messages!")
            _ = _last_user_text(msgs)

            thread_id = f"oka-{uuid4()}"

            results = self._agent.invoke(
                {"messages": msgs},
                config={
                    "configurable": {"thread_id": thread_id}
                }
            )

            try:
                text = results["messages"][-1].content
            except Exception:
                text = str(results)

            return ResponsesAgentResponse(
                output=[self.create_text_output_item(text, str(uuid4()))],
                custom_outputs=request.custom_inputs
            )
    return LangChainResponsesAgent