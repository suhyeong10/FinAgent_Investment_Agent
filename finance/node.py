import json
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.types import Command

from utils.state import AgentState
from utils.llm import ChatOpenRouter
from utils.db import get_supabase_client
from finance.tools import FINANCE_TOOLS_SCHEMA, FINANCE_FUNC_MAP

class FinanceNode:
    def __init__(self):
        self.llm = ChatOpenRouter(model="openai/gpt-4o", temperature=0.4)
        self.supabase = get_supabase_client()

    async def run(self, state: AgentState) -> Command:
        print("\n[Finance Node] Identifying report topic and writing report...")

        user_id = state["user_id"]
        user_profile = state.get("user_profile", {})
        collected_data = state.get("collected_data", {})
        
        debate_log = collected_data.get("debate_history", [])
        debate_text = "\n".join(debate_log) if debate_log else "No debate history."

        # 1. identify the real 'topic' of the report (if DebateNode passed 'report_topic', use it, otherwise find from conversation history)
        if "report_topic" in collected_data:
            report_topic = collected_data["report_topic"]
        else:
            # extract the core question from the entire conversation history
            report_topic = await self._identify_report_topic(state["messages"])

        print(f"Report Topic: {report_topic}")

        # 2. system prompt (forced topic)
        user_tone = user_profile.get('preferred_style', 'Professional')
        
        system_prompt = f"""
        You are the **Chief Wealth Manager (CWM)**.
        Your task is to write a **Final Investment Advisory Report**.

        **CRITICAL INSTRUCTION:**
        The report MUST be specifically about: **"{report_topic}"**.
        Do NOT write a generic report. Answer the user's specific question directly.

        **User Profile:**
        - Name: {user_profile.get('name_display', 'Client')}
        - Risk: {user_profile.get('risk_tolerance_level', 'Unknown')}
        - Tone: **"{user_tone}"** (Apply this tone strictly)

        **Context (Debate/Research):**
        {debate_text[-1000:]}

        **Action Plan:**
        1. Use `recommend_products_sql` to find 2-3 REAL products matching the **"{report_topic}"** and user risk.
        2. Write the report in Korean (Markdown).
           - **Title:** Must include "{report_topic}".
           - **Conclusion:** Direct answer to "{report_topic}".
        """

        # 3. ReAct execution
        final_report = await self._execute_react(report_topic, system_prompt)

        # 4. DB sync
        if self.supabase:
            await self._sync_user_db(user_id, user_profile, report_topic, final_report)

        return Command(
            update={
                "messages": [AIMessage(content=final_report)],
                "collected_data": {"final_report": final_report}
            },
            goto="__end__"
        )

    async def _identify_report_topic(self, messages: list) -> str:
        """
        extract the core question from the conversation history that the user is most interested in
        """
        # convert the recent conversations into text
        history_text = ""
        for m in messages[-10:]: # reference the last 10 turns
            role = "User" if isinstance(m, HumanMessage) else "AI"
            history_text += f"{role}: {m.content}\n"

        prompt = f"""
        Read the conversation history below.
        Identify the **CORE FINANCIAL QUESTION** or **TOPIC** the user wants a report on.
        
        - Ignore profile setup chit-chat (age, income).
        - Ignore simple agreements ("Yes", "Okay").
        - Look for keywords like "Should I buy X?", "Analyze Y", "ETF recommendation".
        
        History:
        {history_text}
        
        Output ONLY the topic string in Korean (e.g., "엔비디아 매수 시점 분석", "미국 국채 ETF 추천").
        """
        
        try:
            res = await self.llm.ainvoke([HumanMessage(content=prompt)])
            return res.content.strip()
        except:
            return "Custom investment suggestion" # default value

    async def _execute_react(self, query: str, system_prompt: str) -> str:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Topic: {query}\n\nFind products and write the report.")
        ]

        for _ in range(2):
            response = await self.llm.ainvoke(messages, tools=FINANCE_TOOLS_SCHEMA)
            
            if not response.tool_calls:
                return response.content

            messages.append(AIMessage(content=response.content or "", additional_kwargs={"tool_calls": response.tool_calls}))

            for tool_call in response.tool_calls:
                func_name = tool_call["function"]["name"]
                args_str = tool_call["function"]["arguments"]
                tool_call_id = tool_call["id"]
                
                print(f"  Finance Tool: {func_name}({args_str})")

                try:
                    func = FINANCE_FUNC_MAP.get(func_name)
                    if func:
                        args = json.loads(args_str)
                        result = func(**args)
                    else:
                        result = "Function not found"
                except Exception as e:
                    result = f"Error: {e}"

                messages.append(ToolMessage(content=str(result), tool_call_id=tool_call_id))
        
        final_res = await self.llm.ainvoke(messages)
        return final_res.content

    async def _sync_user_db(self, user_id: str, profile: dict, topic: str, report: str):
        try:
            now = datetime.now().isoformat()
            report_data = {
                "user_id": user_id,
                "query_summary": topic, # save the exact topic
                "final_report": report,
                "created_at": now
            }
            self.supabase.table("advisory_reports").insert(report_data).execute()
            
            # update the profile update time
            self.supabase.table("user_profile").update({"updated_at": now}).eq("external_user_key", user_id).execute()
            print(f"Report Saved: {topic}")

        except Exception as e:
            print(f"DB Sync Error: {e}")