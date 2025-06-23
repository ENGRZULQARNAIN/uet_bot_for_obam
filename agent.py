from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from preprocess import vectorstore
from langchain_anthropic import ChatAnthropic
import os
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain.prompts import PromptTemplate
from langchain import hub
load_dotenv()
# 1. Get the base prompt
# This prompt is specifically designed for ReAct agents
# prompt = hub.pull("hwchase17/react")
prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tool_names", "tools"],
    template="""You are a helpful assistant that can answer questions.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
{agent_scratchpad}"""
)


# 2. Choose the LLM

llm = ChatAnthropic(
    model="claude-3-5-sonnet-latest",
    temperature=0.5,
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

tool = create_retriever_tool(vectorstore.as_retriever(), "search_knowledge_base", "Search the knowledge base for information about the documents that were uploaded.")
tools = [tool]
# 3. Create the agent
# The agent is the "brain" that decides which tool to use.
agent_executor = create_react_agent(llm, tools, prompt=prompt)

# 4. Create the Agent Executor
# The executor is what runs the agent, calls the tools, and gets the results.
agent_executor = AgentExecutor(agent=agent_executor, tools=tools, handle_parsing_errors=True, max_iterations=10)