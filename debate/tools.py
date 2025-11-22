import os
import json
import re
import asyncio
from datetime import datetime
import yfinance as yf
import FinanceDataReader as fdr
from tavily import TavilyClient

from utils.db import get_supabase_client
from utils.embedding import get_embedding

# [Global Cache] Korean stock list caching (avoid downloading every time)
_KRX_LIST_CACHE = None

def _get_krx_list():
    """get and cache the full list of KRX stocks"""
    global _KRX_LIST_CACHE
    if _KRX_LIST_CACHE is None:
        print("Downloading KRX stock list... (only once)")
        # KRX: KOSPI + KOSDAQ + KONEX
        _KRX_LIST_CACHE = fdr.StockListing('KRX')
    return _KRX_LIST_CACHE

# [settings] index mapping (same as before)
SYMBOL_MAP = {
    "KOSPI": "^KS11", "코스피": "^KS11",
    "KOSDAQ": "^KQ11", "코스닥": "^KQ11",
    "S&P500": "^GSPC", "NASDAQ": "^IXIC", "VIX": "^VIX",
    "USD/KRW": "KRW=X", "GOLD": "GC=F", "BITCOIN": "BTC-USD"
}

# 1. [NEW] search ticker code by company name (Korean stocks first)
def search_ticker(company_name: str) -> str:
    """
    Search stock ticker code by company name. (Korean stocks first)
    Example: "Samsung Electronics" -> "005930", "Ecopro" -> "086520"
    """
    try:
        # 1. search Korean stocks (using FDR)
        df = _get_krx_list()
        # find stocks that contain the search term in the name
        results = df[df['Name'].str.contains(company_name, case=False)]
        
        if not results.empty:
            # no accuracy sorting logic, so return the shortest name or first one
            # (e.g., 'Samsung' search returns 'Samsung Electronics', 'Samsung Life', etc. so return as a list)
            candidates = []
            for _, row in results.head(3).iterrows():
                candidates.append(f"{row['Name']} ({row['Code']})")
            
            return f"Found Korean Stocks: {', '.join(candidates)}"
        
        # 2. if not found in Korea, assume it's a US stock and request ticker guess (hint to LLM)
        # (US stocks don't have separate search API, so need to find with Tavily,
        # here we return a message to guide the LLM to guess the ticker)    
        return f"Couldn't find '{company_name}' in Korea. If it's a US stock, use the Ticker directly (e.g., AAPL) or first search with 'search_news' to get the ticker."

    except Exception as e:
        return f"Ticker search failed: {str(e)}"

# 2. search regulations/regulations by keyword
async def search_regulations(query: str) -> str:
    supabase = get_supabase_client()
    if not supabase: return "Error: DB connection failed."
    try:
        vector = await get_embedding(query)
        response = supabase.rpc("match_documents", {"query_embedding": vector, "match_threshold": 0.5, "match_count": 3}).execute()
        if not response.data: return "관련 문서를 찾을 수 없습니다."
        return "\n".join([f"[문서: {d.get('title')}]\n{d.get('content')}" for d in response.data])
    except Exception as e:
        return f"법률 검색 오류: {e}"

# 3. search news by keyword
def search_news(query: str) -> str:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key: return "Error: Tavily Key missing."
    try:
        client = TavilyClient(api_key=api_key)
        result = client.search(query, search_depth="basic", topic="news", max_results=3)
        return json.dumps(result["results"], ensure_ascii=False)
    except Exception as e:
        return f"News search failed: {e}"

# 4. get market data (Korean stocks first, then global stocks)
def get_market_data(ticker: str) -> str:
    try:
        if bool(re.fullmatch(r"\d{6}", ticker)):
            return _get_korean_stock_data(ticker)
        
        clean_ticker = ticker.upper().strip()
        target_ticker = SYMBOL_MAP.get(clean_ticker, clean_ticker)
        return _get_global_stock_data(target_ticker)
    except Exception as e:
        return f"Data lookup failed: {e}"

def _get_korean_stock_data(code):
    try:
        df = fdr.DataReader(code, datetime.now().year)
        if df.empty: return f"No data: {code}"
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        change_rate = (latest['Close'] - prev['Close']) / prev['Close'] * 100
        return json.dumps({"ticker": code, "source": "Naver(FDR)", "price": int(latest['Close']), "change": f"{change_rate:.2f}%", "date": str(latest.name)}, ensure_ascii=False)
    except: return _get_global_stock_data(f"{code}.KS")

def _get_global_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        try: info = stock.info
        except: return f"Symbol '{ticker}' not found."
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if not price: return f"No data: {ticker}"
        return json.dumps({"ticker": ticker, "source": "Yahoo", "price": price, "pe": info.get("trailingPE")}, ensure_ascii=False)
    except Exception as e: return f"Error: {e}"

# 5. schema and map for tools
DEBATE_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_ticker",
            "description": "Search stock ticker code by company name (Korean stocks supported). Use this if you don't know the ticker.",
            "parameters": {"type": "object", "properties": {"company_name": {"type": "string"}}, "required": ["company_name"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_data",
            "description": "Get market data. Input MUST be a Ticker (e.g., '005930', 'AAPL'). If you only know the name, use 'search_ticker' first.",
            "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_regulations",
            "description": "Search financial laws.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": "Search news.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        }
    }
]

TOOL_FUNC_MAP = {
    "search_ticker": search_ticker,
    "search_regulations": search_regulations,
    "search_news": search_news,
    "get_market_data": get_market_data
}