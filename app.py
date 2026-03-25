import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import asyncio

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

async def search_tavily(query: str) -> str:
    if not TAVILY_API_KEY:
        print("Warning: TAVILY_API_KEY missing.")
        return "VERIFIED DIRECT PRODUCT LINKS:\n\n"
    
    url = "https://api.tavily.com/search"
    
    queries = [
        f"{query} buy now site:amazon.in OR site:flipkart.com",
        f"{query} buy laptop amazon",
        f"{query} flipkart laptop buy",
        f"{query} i5 16GB laptop amazon india"
    ]
    
    async def fetch_results(client: httpx.AsyncClient, q: str) -> list[dict]:
        payload = {
            "api_key": TAVILY_API_KEY,
            "query": q,
            "search_depth": "advanced",
            "max_results": 10
        }
        try:
            res = await client.post(url, json=payload, timeout=20.0)
            res.raise_for_status()
            data = res.json()
            if isinstance(data, dict) and "results" in data:
                return data["results"]
            return []
        except Exception as e:
            print(f"Tavily error for query {q}: {e}")
            return []
            
    async with httpx.AsyncClient() as client:
        tasks = [fetch_results(client, q) for q in queries]
        results_lists = await asyncio.gather(*tasks)
        
    all_results: list[dict] = []
    for r_list in results_lists:
        all_results.extend(r_list)
        
    unique_items: dict[str, dict] = {}
    for item in all_results:
        link = str(item.get("url", ""))
        if link not in unique_items:
            unique_items[link] = item
            
    strict_valid_items: list[dict] = []
    for link, item in unique_items.items():
        url_lower = link.lower()
        if any(x in url_lower for x in ["search", "category", "collections", "laptops"]):
            continue
        if "amazon.in" in url_lower and "/dp/" in url_lower:
            strict_valid_items.append(item)
        elif "flipkart.com" in url_lower and "/p/" in url_lower:
            strict_valid_items.append(item)
            
    if len(strict_valid_items) < 4:
        for link, item in unique_items.items():
            if item in strict_valid_items: 
                continue
            url_lower = link.lower()
            if any(x in url_lower for x in ["search", "category", "collections", "laptops"]):
                continue
            if "amazon.in" in url_lower and ("/dp/" in url_lower or "/gp/product/" in url_lower):
                strict_valid_items.append(item)
            elif "flipkart.com" in url_lower and ("/p/" in url_lower or "itm" in url_lower):
                strict_valid_items.append(item)
                
    context = "VERIFIED DIRECT PRODUCT LINKS:\n\n"
    for item in strict_valid_items[:15]: 
        context += f"Title: {item.get('title')}\nURL: {item.get('url')}\nContent: {item.get('content')}\n\n"
        
    if not strict_valid_items:
        for link, item in unique_items.items():
            if "amazon.in" in link.lower() or "flipkart.com" in link.lower():
                context += f"Title: {item.get('title')}\nURL: {item.get('url')}\nContent: {item.get('content')}\n\n"
                
    return context

async def call_openrouter(query: str, context: str) -> str:
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY is not set.")
        
    if context == "No verified direct product links found.":
        return "No verified direct product links found."
        
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    system_prompt = """You are a senior production-level AI engineer.
Build a HIGH-QUALITY AI laptop recommendation system.

CRITICAL REQUIREMENTS:
1. Recommend EXACTLY 4 to 5 laptops based on Performance, Latest generation CPU, Reliability, Value for money.
2. Select ONLY the BEST options.
3. Highlight ONE laptop as BEST PICK. The others go under OTHER RECOMMENDATIONS.
4. Every laptop MUST have a DIRECT PRODUCT BUY LINK from the context.
5. NEVER generate your own links. ONLY use the verified URLs from the context.
6. Do NOT hallucinate. Drop ANY product without a valid direct link in the context.

You MUST strictly use the following output format constraint. Do not deviate.

REASONING:
[2-3 lines why these are best]

---

BEST PICK:

NAME: [Laptop Name]

SPECS:
* CPU: ...
* RAM: ...
* SSD: ...
* GPU: ...
* Battery: ...

WHY BEST:
[1 strong reason]

---

OTHER RECOMMENDATIONS:

NAME: [Laptop Name]

SPECS:
* CPU: ...
* RAM: ...
* SSD: ...
* GPU: ...
* Battery: ...

WHY GOOD:
[1 line]

---
(repeat OTHER RECOMMENDATIONS sections separated by --- until 4-5 laptops total)
---

LINKS:
[Laptop Name 1]:
[Valid URL from context]

[Laptop Name 2]:
[Valid URL from context]
"""
    
    user_prompt = f"Query: {query}\n\nContext:\n{context}\n\nRecommend exactly 4 to 5 laptops following the strict output format."
    
    model_name = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3
    }
        
    async with httpx.AsyncClient() as client:
        try:
            res = await client.post(url, json=payload, headers=headers, timeout=60.0)
            res.raise_for_status()
            return str(res.json()["choices"][0]["message"]["content"])
        except Exception as e:
            print(f"OpenRouter error: {e}")
            raise HTTPException(status_code=500, detail="LLM request failed.")

@app.get("/")
def serve_index():
    if not os.path.exists("index.html"):
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse("index.html")

@app.post("/ask")
async def ask(request: QueryRequest):
    context = await search_tavily(request.question)
    answer = await call_openrouter(request.question, context)
    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
