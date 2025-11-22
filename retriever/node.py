import json
import asyncio
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.types import Command

from utils.state import AgentState
from utils.llm import ChatOpenRouter
from retriever.tools import RETRIEVER_TOOLS_SCHEMA, RETRIEVER_FUNC_MAP

class RetrieverNode:
    """
    [Information Retrieval Agent]
    Search the optimal data source based on the user's question and provide the answer.
    1. DB (SQL): Search specific financial products
    2. Vector DB (RAG): Search financial knowledge, reports
    3. Web/API: Search real-time stock prices and news
    """
    def __init__(self):
        self.llm = ChatOpenRouter(model="openai/gpt-4o", temperature=0) # important for accuracy -> temperature 0

    async def run(self, state: AgentState) -> Command:
        messages = state["messages"]
        user_query = messages[-1].content
        
        print(f"Retriever Analysis: {user_query}")

        system_prompt = """
        You are a Financial Information Specialist.
        Analyze the user's request and retrieve accurate data using tools.

        **Tool Selection Strategy:**
        1. Use `search_products_sql` for product searches.
           - **CRITICAL RULE for `keyword`**: 
             - Use ONLY nouns (Subject/Sector). **NEVER** use adjectives like "cheap", "best".
             - **[NEW] GLOBAL SEARCH:** If the user does NOT specify a subject (e.g., just says "Find lowest fee product"), leave `keyword` and `category` as **null (None)**. Just set `sort_by`.
           - **CRITICAL RULE for `sort_by`**:
             - "Cheap/Low fee" -> `sort_by="fees_asc"`
             - "High return/Best" -> `sort_by="return_desc"`

        2. Use `search_documents_rag` for concepts (e.g., "What is ETF?").
        3. Use `get_realtime_price` for stock prices.
        4. Use `search_web` for news.
        
        **Action:**
        - Do NOT ask clarifying questions (e.g., "Which category?"). **Just search the database first.**
        - Only ask questions AFTER showing the initial search results.
        """

        final_response = await self._execute_react(user_query, system_prompt)

        advisor_prompt = f"""
        User's question: "{user_query}"
        Search results: "{final_response}"

        You are a meticulous investment consultant.
        When showing the search results, if there are additional information needed before writing the report, suggest it specifically.
        (e.g., "Let's narrow down the products with strong dividend preference?", "Let's compare the recent 1-year returns of these products?")
        
        **If the search results are already sufficient, do not ask unnecessary questions.**
        Write in a polite and professional manner.
        """
        
        refinement_msg = await self.llm.ainvoke([HumanMessage(content=advisor_prompt)])

        return Command(
            update={
                "messages": [AIMessage(content=f"{final_response}\n\n---\n💬 {refinement_msg.content}")],
            },
            goto="__end__"
        )

    async def _execute_react(self, query: str, system_prompt: str) -> str:
        messages = [
            SystemMessage(content=system_prompt), 
            HumanMessage(content=query)
        ]

        for _ in range(2):
            response = await self.llm.ainvoke(messages, tools=RETRIEVER_TOOLS_SCHEMA)
            
            if not response.tool_calls:
                return response.content

            messages.append(AIMessage(content=response.content or "", additional_kwargs={"tool_calls": response.tool_calls}))
            
            for tool_call in response.tool_calls:
                func_name = tool_call["function"]["name"]
                args_str = tool_call["function"]["arguments"]
                tool_call_id = tool_call["id"]
                
                print(f"Retriever calls: {func_name}({args_str})")

                try:
                    func = RETRIEVER_FUNC_MAP.get(func_name)
                    if func:
                        args = json.loads(args_str)
                        if asyncio.iscoroutinefunction(func):
                            result = await func(**args)
                        else:
                            result = func(**args)
                    else:
                        result = "Function not found"
                except Exception as e:
                    result = f"Execution Error: {str(e)}"

                messages.append(ToolMessage(content=str(result), tool_call_id=tool_call_id))
        
        final_res = await self.llm.ainvoke(messages)

        return final_res.content