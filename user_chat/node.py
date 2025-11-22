import json
from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.types import Command

from utils.state import AgentState
from utils.llm import ChatOpenRouter
from utils.db import get_supabase_client
from user_chat.models import ExtractedInfo

# IMPORTANT: all required fields to collect
REQUIRED_FIELDS = [
    "name_display", "age_range", "income_bracket", 
    "invest_experience_yr", "financial_knowledge_level", 
    "current_holdings_note", "preferred_asset_types",
    "risk_tolerance_level", "total_investable_amt", 
    "goal_type", "goal_description", "preferred_style"
]

class UserProfileChatNode:
    def __init__(self):
        self.llm = ChatOpenRouter(model="openai/gpt-4o", temperature=0.7)
        self.supabase = get_supabase_client()

    async def run(self, state: AgentState) -> Command:
        user_id = state["user_id"]
        
        db_profile = self._fetch_profile_from_db(user_id)
        current_profile = state.get("user_profile") or {}
        if db_profile:
            current_profile.update(db_profile)
            
        missing_fields = []
        for field in REQUIRED_FIELDS:
            val = current_profile.get(field)
            if val is None or val == "" or val == []:
                missing_fields.append(field)
        
        is_complete = len(missing_fields) == 0
        
        if is_complete:
            print(f"⏩ Profile Fully Completed. Passing to Condition Node.")
            return Command(update={"user_profile": current_profile}, goto="condition")
            
        # 4. system prompt (natural interview guide)
        system_prompt = f"""
        You are a friendly Investment Onboarding Assistant.
        User Profile Status: {json.dumps(current_profile, ensure_ascii=False)}
        Missing Info: {missing_fields}

        **Conversation Strategy:**
        You need to collect ALL missing fields, but **ask only 1-2 questions at a time**.
        Group related topics naturally:
        1. **Basics:** Name, Age, Job/Income
        2. **Experience:** Investment Years (`invest_experience_yr`), Knowledge Level (`financial_knowledge_level`)
        3. **Assets:** Investable Amount, Current Holdings (`current_holdings_note`)
        4. **Preferences:** Asset Types (`preferred_asset_types`), Risk Level
        5. **Goals:** Goal Type (`goal_type`), Description
        6. **Style:** AI Persona (`preferred_style` -> e.g. "밝은친구형", "차분한코치")

        **Extraction Rules:**
        - `financial_knowledge_level`: Map to ['beginner', 'intermediate', 'advanced']
        - `goal_type`: Map to ['retirement', 'wealth_building', 'short_term'...]
        - `preferred_style`: Map to ['공손/설명형', '직설', '안정적/안심', '차분한코치', '밝은친구형']

        Output JSON matching `ExtractedInfo`. `response_message` is required.
        """
        
        structured_llm = self.llm.with_structured_output(ExtractedInfo)
        messages = [SystemMessage(content=system_prompt)] + state["messages"][-6:]
        
        result = await structured_llm.ainvoke(messages)
        
        # 5. save to database
        extracted_data = result.model_dump(exclude={"response_message"}, exclude_none=True)
        if extracted_data:
            current_profile.update(extracted_data)
            self._save_profile_to_db(user_id, current_profile)
            print(f"Database synced: {extracted_data.keys()}")

        # 6. generate response (Fallback: if the LLM didn't give a message)
        ai_response = result.response_message
        if not ai_response:
            # apply the same precise check logic here
            remaining = []
            for field in REQUIRED_FIELDS:
                val = current_profile.get(field)
                if val is None or val == "" or val == []:
                    remaining.append(field)
            
            if remaining:
                ai_response = await self._generate_question_dynamically(remaining[0], current_profile)
            else:
                ai_response = "All information has been collected! Shall we start investing?"

        return Command(
            update={"user_profile": current_profile, "messages": [AIMessage(content=ai_response)]},
            goto="__end__" # wait for the user's response
        )

    async def _generate_question_dynamically(self, target_field: str, profile: dict) -> str:
        prompt = f"""
        User Profile: {json.dumps(profile, ensure_ascii=False)}
        Missing Field: '{target_field}'
        Generate a natural question to ask for this information.
        """
        res = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return res.content

    def _fetch_profile_from_db(self, user_id: str) -> dict:
        if not self.supabase: return {}
        try:
            res = self.supabase.table("user_profile").select("*").eq("external_user_key", user_id).execute()
            if res.data: return res.data[0]
        except Exception as e:
            print(f"Database load error: {e}")
        return {}

    def _save_profile_to_db(self, user_id: str, data: dict):
        if not self.supabase: return
        try:
            data["external_user_key"] = user_id
            data["updated_at"] = datetime.now().isoformat()
            self.supabase.table("user_profile").upsert(data, on_conflict="external_user_key").execute()
        except Exception as e:
            print(f"Database save error: {e}")