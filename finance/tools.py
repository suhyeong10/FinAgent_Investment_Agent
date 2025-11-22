import os
import json
from typing import Optional
from utils.db import get_supabase_client

# Risk Mapping: User Profile(Korean) -> DB Risk Level(Int)
def _map_risk_level(user_risk_profile: str) -> list:
    """
    convert the user's profile (Korean) to the range of risk_level (1~6) in the DB
    DB criteria: 1=Low Risk, 6=High Risk (estimated)
    """
    profile = user_risk_profile.lower() if user_risk_profile else "moderate"
    
    if "aggressive" in profile: # aggressive -> allow 1~3 levels
        return [1, 2, 3]
    elif "moderate" in profile: # moderate -> allow 3~4 levels
        return [3, 4]
    elif "conservative" in profile: # conservative -> allow 5~6 levels (if not available, allow up to 4)
        return [4, 5, 6]
    else:
        return [1, 2, 3, 4, 5, 6] # all levels

# Product Recommendation Tool (SQL Filter)
def recommend_products_sql(
    risk_level: Optional[str] = None, 
    category: Optional[str] = None, 
    keyword: Optional[str] = None
) -> str:
    """
    Recommend products based on the user's profile.
    
    Args:
        risk_level: user profile (e.g., 'aggressive', 'moderate', 'conservative')
        category: product type (e.g., 'ETF', 'FUND')
        keyword: search keyword (e.g., 'Semiconductor', 'US', 'AI')
    """
    supabase = get_supabase_client()
    if not supabase: return "Error: DB connection failed."

    try:
        query = supabase.table("investment_products").select("*")
        
        # 1. risk level mapping and filtering
        # if the user is 'aggressive', find products with risk_level 1,2,3 in the DB
        if risk_level:
            target_levels = _map_risk_level(risk_level)
            query = query.in_("risk_level", target_levels)

        # 2. category filter (DB column: product_group)
        if category:
            # use ilike for case-insensitive comparison or convert to uppercase
            query = query.ilike("product_group", category)

        # 3. keyword search (product name or description)
        if keyword:
            # search in product name (product_name) or description (description)
            query = query.or_(f"product_name.ilike.%{keyword}%,description.ilike.%{keyword}%")

        # 4. sort and limit
        # sort by expected_return in descending order by default
        query = query.order("expected_return", desc=True).limit(5)

        response = query.execute()
        
        if not response.data:
            return f"No products found for the given conditions: {risk_level}, {category}, {keyword}"

        # format the results (exact DB column names)
        results = []
        for item in response.data:
            info = (
                f"- [상품명]: {item.get('product_name')} ({item.get('product_code')})\n"
                f"  [유형]: {item.get('product_group')} ({item.get('product_type')})\n"
                f"  [위험도]: {item.get('risk_category')} ({item.get('risk_level')}등급)\n"
                f"  [예상수익률]: {item.get('expected_return')}% | [보수]: {item.get('fee')}%\n"
                f"  [설명]: {item.get('description', '')[:100]}..."
            )
            results.append(info)
            
        return "\n\n".join(results)

    except Exception as e:
        return f"Product DB lookup failed: {str(e)}"

# tool schema
FINANCE_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "recommend_products_sql",
            "description": "Search investment products in DB based on user profile.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "risk_level": {"type": "string", "description": "User risk profile (aggressive, moderate, conservative)"},
                    "category": {"type": "string", "description": "Product category (ETF, FUND)"},
                    "keyword": {"type": "string", "description": "Theme keyword (e.g. 'Semiconductor', 'US')"}
                }, 
                "required": [] 
            }
        }
    }
]

FINANCE_FUNC_MAP = {
    "recommend_products_sql": recommend_products_sql
}