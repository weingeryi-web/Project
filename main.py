import os
import json
import requests
from fastapi import FastAPI, HTTPException
from openai import OpenAI

app = FastAPI()

# 基础环境变量读取
API_KEY = os.getenv("GEMINI_API_KEY") 
BASE_URL = os.getenv("GEMINI_BASE_URL")   
FEISHU_WEBHOOK = os.getenv("FEISHU_WEBHOOK")
DATA_URL = os.getenv("DATA_URL", "https://news.ycombinator.com/rss")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3.5-flash") 

def brave_web_search(query: str) -> str:
    if not BRAVE_API_KEY:
        return "Error: BRAVE_API_KEY is not configured."
    
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": BRAVE_API_KEY
    }
    params = {"q": query, "count": 3}
    
    try:
        res = requests.get(url, headers=headers, params=params, timeout=5)
        res.raise_for_status()
        results = res.json().get("web", {}).get("results", [])
        snippets = [f"Title: {r.get('title')}\nSnippet: {r.get('description')}" for r in results]
        return "\n\n".join(snippets) if snippets else "No search results found."
    except Exception as e:
        return f"Brave Search failed: {str(e)}"

@app.get("/run")
@app.post("/run")
def generate_briefing():
    # 1. 抓取外部目标数据
    try:
        res = requests.get(DATA_URL, timeout=10)
        res.raise_for_status()
        raw_data = res.text[:25000]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data fetch failed: {str(e)}")

    # 2. 使用 OpenAI 规范调用公司中转站
    try:
       client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

        
        prompt = f"""
        你是一个高级投资与技术趋势分析顾问。请将以下原始 XML/RSS 数据中今日最值得关注的 3-5 个核心动态进行提取。
        
        要求：
        1. 使用中文进行高度精炼的总结。
        2. 给出技术可行性或投资价值的简要点评（每项 100 字以内）。
        3. 输出格式必须为结构清晰、排版优雅的 Markdown 格式。
        
        原始数据：
        {raw_data}
        """
        
        tools = [{
            "type": "function",
            "function": {
                "name": "brave_web_search",
                "description": "Search the internet using Brave Search API to get real-time context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query keyword or phrase."}
                    },
                    "required": ["query"]
                }
            }
        }]
        
        messages = [{"role": "user", "content": prompt}]
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=tools,
            temperature=0.3
        )
        
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        
        if tool_calls:
            messages.append(response_message)
            for tool_call in tool_calls:
                if tool_call.function.name == "brave_web_search":
                    function_args = json.loads(tool_call.function.arguments)
                    search_result = brave_web_search(function_args.get("query"))
                    
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": "brave_web_search",
                        "content": search_result
                    })
            
            second_response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.3
            )
            summary_md = second_response.choices[0].message.content
        else:
            summary_md = response_message.content

        if not summary_md:
            raise ValueError("Empty response from proxy LLM")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Proxy LLM execution failed: {str(e)}")

    # 3. 核心修复：改用飞书标准消息卡片 (interactive) 承载 Markdown
    payload = {
        "msg_type": "interactive",
        "card": {
            "header": {
                "title": {
                    "tag": "plain_text",
                    "content": "🤖 AI 高级工程顾问简报"
                },
                "template": "blue"
            },
            "elements": [
                {
                    "tag": "markdown",
                    "content": summary_md
                }
            ]
        }
    }
    
    try:
        fs_res = requests.post(FEISHU_WEBHOOK, json=payload, timeout=10)
        fs_res.raise_for_status()
        return {"status": "success", "feishu_response": fs_res.json()}
    except Exception as e:
        return {"status": "partial_success", "error": f"Feishu push failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)