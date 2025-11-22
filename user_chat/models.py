from typing import Optional, List, Literal
from pydantic import BaseModel, Field

class ExtractedInfo(BaseModel):
    """
    Supabase database schema
    """
    # 1. basic information
    name_display: Optional[str] = Field(None, description="사용자 이름 또는 닉네임")
    age_range: Optional[str] = Field(None, description="연령대 (예: '20대', '30대후')")
    income_bracket: Optional[str] = Field(None, description="소득 구간 (예: '5천~7천', '1억+')")
    
    # 2. investment experience (Numeric)
    invest_experience_yr: Optional[float] = Field(None, description="투자 경험 연수 (숫자만, 예: 3.5)")
    
    # 3. [important] DB Enum mapping (English lowercase)
    risk_tolerance_level: Optional[Literal['conservative', 'moderately_conservative', 'moderate', 'moderately_aggressive', 'aggressive']] = Field(
        None, 
        description="위험 성향: conservative(안정형), moderate(중립형), aggressive(공격형) 중 선택"
    )
    
    financial_knowledge_level: Optional[Literal['beginner', 'intermediate', 'advanced']] = Field(
        None, 
        description="금융 지식 수준: beginner(초보), intermediate(중수), advanced(고수)"
    )
    
    goal_type: Optional[Literal['retirement', 'short_term', 'mid_term', 'long_term', 'wealth_building']] = Field(
        None, 
        description="투자 목표 타입: retirement(은퇴), short_term(단기), mid_term(중기), long_term(장기)"
    )
    
    # 4. text and array fields
    goal_description: Optional[str] = Field(None, description="구체적인 목표 설명 (한글)")
    current_holdings_note: Optional[str] = Field(None, description="현재 보유 자산 메모")
    total_investable_amt: Optional[float] = Field(None, description="총 투자 가능 금액 (원 단위 숫자)")
    
    # Postgres Array type mapping (List[str])
    preferred_asset_types: Optional[List[str]] = Field(
        None, 
        description="선호 자산군 목록 (예: ['주식', 'ETF', '채권', '미장', '국장'])"
    )

    # 5. [important] AI conversation style (recommended to use the Korean values in the database)
    preferred_style: Optional[Literal['공손/설명형', '직설', '안정적/안심', '차분한코치', '탐은친구형', '전문가형']] = Field(
        None, 
        description="AI 답변 스타일: '공손/설명형', '직설', '자분한코치' 등 DB 예시 참고"
    )
    
    # response message for the bot (not saved in the database)
    response_message: Optional[str] = Field(None, description="유저에게 보낼 자연스러운 답변")

    class Config:
        extra = "ignore"