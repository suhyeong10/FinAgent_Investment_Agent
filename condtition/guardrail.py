import json
from langchain_core.messages import HumanMessage, SystemMessage
from utils.llm import ChatOpenRouter
from utils.state import AgentState

class GuardrailNode:
    def __init__(self):
        self.llm = ChatOpenRouter(model="openai/gpt-4o", temperature=0)
        
        # add context-aware decision guidelines
        self.system_prompt = """
        You are the Security & Domain Guardrail for a Financial AI.
        Determine if the user input is safe and relevant.

        **Context Awareness Rules:**
        1. **Previous AI Question:** If the AI asked a question (e.g., "Do you prefer growth or dividend?"), and the user answers (e.g., "Growth", "Yes", "No"), this is **'finance'**, NOT 'general_chat' or 'profile_update'.
        2. **Profile Update:** Classify as `'profile_update'` ONLY if the user EXPLICITLY asks to change data (e.g., "Change my income", "Update risk level").
        3. **General Chat:** Greetings ("Hi") or simple thanks ("Thank you").
        4. **Unsafe:** Hate speech, illegal acts, etc.

        **Categories:**
        - 'finance': Investment questions, market data, OR **answers to AI's questions**.
        - 'profile_update': Explicit commands to change DB info.
        - 'general_chat': Irrelevant to finance but safe.
        - 'unsafe': Block this.

        Output JSON: {"is_allowed": bool, "category": "...", "reason": "..."}
        """

    async def run(self, state: AgentState) -> dict:
        user_input = state["messages"][-1].content
        
        history_context = ""
        if len(state["messages"]) > 1:
            last_ai_msg = state["messages"][-2].content
            history_context = f"AI previously asked: \"{last_ai_msg}\"\n"

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"{history_context}User Input: \"{user_input}\"")
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            # JSON parsing (remove markdown)
            content = response.content.strip()
            if content.startswith("```json"): content = content[7:]
            if content.endswith("```"): content = content[:-3]
            
            result = json.loads(content.strip())
            
            # safety fallback: if parsing succeeded but required keys are missing, use default
            if "is_allowed" not in result:
                result = {"is_allowed": True, "category": "finance", "reason": "Default pass"}
                
        except Exception as e:
            # if error, pass for now (Fail-open)
            print(f"Guardrail Error: {e}")
            result = {"is_allowed": True, "category": "finance", "reason": "Error handling"}
            
        # for debugging: log the result
        print(f"Guardrail: [{result.get('category')}] -> Allowed: {result.get('is_allowed')}")
        
        return {"guardrail_result": result}