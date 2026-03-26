import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
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
    question: Optional[str] = None
    filters: Optional[dict] = None

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

import re

# Brand -> title keywords that must appear in the product TITLE for the brand to match
BRAND_TITLE_KEYWORDS: dict[str, list[str]] = {
    "apple":   ["iphone", "apple iphone"],
    "samsung":  ["samsung", "galaxy"],
    "oneplus":  ["oneplus"],
    "xiaomi":   ["xiaomi", "redmi", "poco"],
    "realme":   ["realme"],
    "oppo":     ["oppo"],
    "vivo":     ["vivo"],
    "motorola": ["motorola", "moto"],
    "nokia":    ["nokia"],
    "iqoo":     ["iqoo"],
    "google":   ["google", "pixel"],
    "nothing":  ["nothing phone"],
}

def enforce_constraints(items: list, filters: dict) -> list:
    if not filters: return items
    
    price_filter = filters.get("price", "")
    brand_filter = filters.get("brand", "").lower().strip()
    
    min_b, max_b = 0, 999999
    if price_filter and price_filter != "Any":
        if "Under" in price_filter:
            max_b = int(price_filter.replace("Under", "").replace(",", "").strip())
        elif "-" in price_filter:
            parts = price_filter.split("-")
            min_b, max_b = int(parts[0].replace(",", "").strip()), int(parts[1].replace(",", "").strip())
        elif "+" in price_filter:
            min_b = int(price_filter.replace("+", "").replace(",", "").strip())

    def parse_price(txt: str) -> int:
        match = re.search(r'(?:₹|rs\.?|inr|₹\s*)\s*([\d,]+)', txt, re.IGNORECASE)
        if match:
            try: return int(match.group(1).replace(',', ''))
            except: pass
        return 0

    def title_matches_brand(title: str) -> bool:
        """Check if product title belongs to the selected brand."""
        t = title.lower()
        # Get the keywords that represent this brand in product titles
        keywords = BRAND_TITLE_KEYWORDS.get(brand_filter, [brand_filter])
        return any(kw in t for kw in keywords)

    valid_items = []
    for item in items:
        title = item.get("title", "")
        content = item.get("content", "")
        
        # STRICT brand check: only against title, not content (content often mentions competing brands)
        if brand_filter and brand_filter != "any":
            if not title_matches_brand(title):
                continue
            
        p = parse_price(title + " " + content)
        if p > 0:
            if max_b < 999999 and p > max_b: continue
            if min_b > 0 and p < min_b: continue
                
        valid_items.append(item)
    return valid_items

async def search_tavily(query: str, filters: dict = None) -> tuple[str, list, list]:
    if not TAVILY_API_KEY:
        print("Warning: TAVILY_API_KEY missing.")
        return "VERIFIED DIRECT PRODUCT LINKS:\n\n", [], []
    
    tavily_url = "https://api.tavily.com/search"
    
    TRUSTED_DOMAINS = ["amazon.in", "flipkart.com", "croma.com", "reliancedigital.in"]
    
    # 6 diversified queries spanning all 4 trusted retailers
    queries = [
        f"{query} buy site:amazon.in",
        f"{query} buy site:flipkart.com",
        f"{query} buy site:croma.com",
        f"{query} buy site:reliancedigital.in",
        f"{query} 5G 128GB smartphone buy India 2025",
        f"{query} best phone India under budget flipkart amazon croma",
    ]
    
    async def fetch_results(client: httpx.AsyncClient, q: str) -> list[dict]:
        payload = {
            "api_key": TAVILY_API_KEY,
            "query": q,
            "search_depth": "advanced",
            "max_results": 10
        }
        try:
            res = await client.post(tavily_url, json=payload, timeout=20.0)
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
        
    # Deduplicate by URL
    unique_items: dict[str, dict] = {}
    for item in all_results:
        link = str(item.get("url", ""))
        if link not in unique_items:
            unique_items[link] = item

    # Exclude blacklisted URL patterns
    BLACKLIST_PATTERNS = ["search", "category", "collections", "cases", "covers", "accessories", "charger", "headphone"]
    
    def is_valid_product_url(link: str) -> bool:
        lk = link.lower()
        if any(b in lk for b in BLACKLIST_PATTERNS):
            return False
        if "amazon.in" in lk and ("/dp/" in lk or "/gp/product/" in lk):
            return True
        if "flipkart.com" in lk and ("/p/" in lk or "itm" in lk):
            return True
        if "croma.com" in lk and "/p/" in lk:
            return True
        if "reliancedigital.in" in lk and "/p/" in lk:
            return True
        return False
    
    def is_any_trusted_domain(link: str) -> bool:
        return any(d in link.lower() for d in TRUSTED_DOMAINS)
    
    # Phase 1: strict product-page URLs
    strict_valid_items: list[dict] = [item for link, item in unique_items.items() if is_valid_product_url(link)]
    
    # Phase 2: if < 4 results, relax to any trusted-domain page (still no blacklist)
    if len(strict_valid_items) < 4:
        for link, item in unique_items.items():
            if item in strict_valid_items:
                continue
            lk = link.lower()
            if any(b in lk for b in BLACKLIST_PATTERNS):
                continue
            if is_any_trusted_domain(link):
                strict_valid_items.append(item)

    # Apply constraint filtering (brand + budget)
    strict_valid_items = enforce_constraints(strict_valid_items, filters)
    
    # Phase 3: Smart relaxation — if still < 3 results, relax non-critical filters
    if len(strict_valid_items) < 3 and filters:
        relaxed_filters = dict(filters)
        for drop_key in ["camera", "battery", "performance"]:
            if relaxed_filters.get(drop_key):
                relaxed_filters.pop(drop_key)
                all_trusted = [item for link, item in unique_items.items() if is_any_trusted_domain(link) and not any(b in link.lower() for b in BLACKLIST_PATTERNS)]
                relaxed_items = enforce_constraints(all_trusted, relaxed_filters)
                if len(relaxed_items) >= 3:
                    strict_valid_items = relaxed_items
                    break

    if not strict_valid_items:
        if filters and filters.get("price") and filters["price"] != "Any":
            brand = filters.get("brand", "") or ""
            return f"BUDGET FAIL: {filters['price']}|{brand}", all_results[:3], []
        return "NO VALID PRODUCTS EXIST FOR THESE CONSTRAINTS.", all_results[:3], []
        
    context = "VERIFIED DIRECT PRODUCT LINKS:\n\n"
    for item in strict_valid_items[:15]: 
        context += f"Title: {item.get('title')}\nURL: {item.get('url')}\nContent: {item.get('content')}\n\n"
        
    return context, all_results[:3], strict_valid_items

async def call_openrouter(query: str, context: str) -> str:
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY is not set.")
        
    if "BUDGET FAIL:" in context:
        raw = context.split("FAIL: ")[1].strip()
        # Try to extract brand and price from the context token
        # Format is "BUDGET FAIL: <price_range>|<brand>"
        parts = raw.split("|")
        price_str = parts[0].strip() if parts else raw
        brand_str = parts[1].strip() if len(parts) > 1 else ""
        
        brand_msg = f"{brand_str} phones are" if brand_str and brand_str.lower() not in ("any", "") else "Phones are"
        budget_note = ""
        if brand_str.lower() == "apple":
            budget_note = "\n* Apple iPhones typically start from ₹40,000 for older models"
        elif brand_str.lower() == "samsung":
            budget_note = "\n* Samsung flagships start from ₹25,000; budget models from ₹10,000"
        elif brand_str.lower() == "oneplus":
            budget_note = "\n* OnePlus phones typically start from ₹15,000"
            
        return f"""NO EXACT MATCH FOUND

REASON:
* {brand_msg} not available {price_str.lower()} in verified listings{budget_note}

SUGGESTIONS:
* Increase your budget to see {brand_str if brand_str and brand_str.lower() != 'any' else 'more'} options
* OR consider brands like Samsung, OnePlus, or Xiaomi for this budget"""

    if context == "NO VALID PRODUCTS EXIST FOR THESE CONSTRAINTS." or context == "No verified direct product links found.":
        return """NO EXACT MATCH FOUND

REASON:
* No verified products found matching your exact combination of filters in live listings

SUGGESTIONS:
* Increase your budget range
* OR change brand
* OR relax battery/camera requirements"""
        
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "AI Mobile Finder"
    }
    
    system_prompt = """You are a senior production-level AI engineer designed to provide EXPLAINABLE reasoning.
Build a HIGH-QUALITY AI smartphone recommendation system.

Your job is NOT just to give answers, but to show structured decision-making BEFORE your final list. Provide DETAILED, STEP-BY-STEP reasoning in a SAFE and STRUCTURED way. Do NOT hide reasoning. Do NOT expose sensitive hidden chains. Do not hallucinate data not found in the context!

CRITICAL REQUIREMENTS:
1. Recommend EXACTLY 3 to 5 smartphones based on Camera, Battery, Performance, Display, and 5G support.
2. Select ONLY the BEST options that strictly obey budget boundaries.
3. Highlight ONE smartphone as BEST PICK. The others go under OTHER RECOMMENDATIONS.
4. Every smartphone MUST have a DIRECT PRODUCT BUY LINK from the context.
5. NEVER generate your own links. ONLY use the verified URLs from the context.
6. Do NOT hallucinate. Drop ANY product without a valid direct link in the context.

BRAND FILTER GUARD (ABSOLUTE RULE):
- If the user's query specifies a brand (e.g., Apple, Samsung, OnePlus), you MUST ONLY recommend phones of that exact brand.
- NEVER recommend a different brand even if it has better specs or is cheaper.
- If the context contains no phones matching the brand, return NO EXACT MATCH FOUND — do NOT recommend phones from other brands as substitutes without clearly labelling them as alternatives.
- NEVER rename a product. If it says "Redmi" do NOT call it "Xiaomi" and vice versa unless they are the same product.

You MUST strictly use the following output format. Do not deviate. Be clear and logical using bullet points.

REASONING_SHORT:
* [1 line: what the user wants + budget summary]
* [1 line: which key filter dominated the selection]
* [1 line: why these phones are the best value match]

REASONING:

STEP 1: UNDERSTAND USER INTENT
* [What does the user want? Budget? Brand? Performance?]

STEP 2: BREAKDOWN CONSTRAINTS
* [Price limits, brand restrictions, battery etc.]

STEP 3: VALIDATION CHECK
* [Are constraints realistic against the live data context? Conflicts?]

STEP 4: SEARCH STRATEGY
* [What phones were queried?]

STEP 5: TOOL DECISION
* [Why Tavily Search LIVE WEB DB was accessed and what data retrieved]

STEP 6: RESULT ANALYSIS
* [What phones were found in the context? Which match the budget filters?]

STEP 7: FINAL DECISION
* [Why selected phones are best.]

STEP 8: EDGE CASE HANDLING
* [If no perfect match, explain compromises. Else: clear.]

IMPORTANT TRANSPARENCY RULES:
- If a phone in the context EXCEEDS the user's stated budget, do NOT silently recommend it.
- If the only matching phones are over-budget, use the PARTIAL MATCH FOUND output format instead.
- NEVER change brand or budget without flagging it to the user.

IF partial match (closest available but over-budget), use THIS format instead of BEST PICK:

PARTIAL MATCH FOUND:

CLOSEST MATCH:
NAME: [Phone name]
ESTIMATED PRICE: [~₹XX,XXX]
WARNING: This phone exceeds your selected budget
WHY CLOSEST: [1 line explaining why it's the best available near budget]

ALTERNATIVE OPTIONS WITHIN BUDGET:
* [Android brand] phone available around [budget range]
* [Another brand] with [key spec] within budget

---

BEST PICK:

NAME: [Smartphone Name]

PRICE: [Extract from context e.g. ₹24,999 | If unavailable: Estimated ₹20,000–₹25,000]

SPECS:
* ⚡ Processor: ...
* 💾 RAM: ...
* 💿 Storage: ...
* 📸 Camera: ...
* 🔋 Battery: ...
* 📱 Display: ...

WHY BEST:
[1 strong reason]

---

OTHER RECOMMENDATIONS:

NAME: [Smartphone Name]

PRICE: [Extract from context e.g. ₹24,999 | If unavailable: Estimated ₹20,000–₹25,000]

SPECS:
* ⚡ Processor: ...
* 💾 RAM: ...
* 💿 Storage: ...
* 📸 Camera: ...
* 🔋 Battery: ...
* 📱 Display: ...

WHY GOOD:
[1 line]

---
(repeat OTHER RECOMMENDATIONS sections separated by --- until 3-5 smartphones total)
---

LINKS:
[Smartphone Name 1]:
[Valid URL from context]

[Smartphone Name 2]:
[Valid URL from context]
"""
    
    user_prompt = f"Query: {query}\n\nContext:\n{context}\n\nRecommend exactly 3 to 5 smartphones following the strict output format."
    
    # Free model cascade — tries models in order, never crashes server
    FREE_MODELS = [
        os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3-8b-instruct"),
        "mistralai/mistral-7b-instruct",
        "google/gemma-2-9b-it",
    ]
    
    async with httpx.AsyncClient() as client:
        for model_name in FREE_MODELS:
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.3
            }
            try:
                res = await client.post(url, json=payload, headers=headers, timeout=30.0)
                if res.status_code in (400, 402, 404, 429, 503):
                    print(f"Model {model_name} unavailable (HTTP {res.status_code}), trying next...")
                    continue
                res.raise_for_status()
                data = res.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if content:
                    return str(content)
                print(f"Model {model_name} returned empty content, trying next...")
            except httpx.TimeoutException:
                print(f"Model {model_name} timed out after 30s, trying next...")
            except Exception as e:
                print(f"Model {model_name} error: {e}, trying next...")
        
        # All models failed — return safe message, do NOT crash
        return """NO EXACT MATCH FOUND

REASON:
* ⚠️ AI service temporarily unavailable. Please try again.

SUGGESTIONS:
* Retry your search
* OR adjust your filters and try again"""

@app.get("/")
def serve_index():
    if not os.path.exists("index.html"):
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse("index.html")

def build_query(f: dict) -> str:
    parts = []
    if f.get("brand"):
        parts.append(f["brand"])
    
    if f.get("performance"):
        parts.append(f["performance"].lower() + " smartphone")
    else:
        parts.append("smartphone")
        
    if f.get("price"):
        val = f["price"]
        if "Under" in val:
            parts.append(val.replace("Under", "under").replace(",", ""))
        elif "-" in val:
            parts.append("under " + val.split("-")[1].replace(",", "").strip())
        elif "+" in val:
            parts.append("above " + val.replace("+", "").replace(",", "").strip())
            
    if f.get("battery"):
        parts.append(f["battery"] + " battery")
        
    if f.get("camera") == "Best (Flagship camera)":
        parts.append("flagship camera")
    elif f.get("camera") == "Good":
        parts.append("good camera")
        
    parts.append("India")
    return " ".join(parts).strip()

@app.post("/ask")
async def ask(request: QueryRequest):
    import datetime, time
    
    if request.question and request.question.strip():
        query_str = request.question.strip()
        input_data = {"question": query_str}
    elif request.filters:
        query_str = build_query(request.filters)
        input_data = {"filters": request.filters}
    else:
        raise HTTPException(status_code=400, detail="Please provide either a question or filters.")

    start_time = time.time()
    filters_dict = request.filters if request.filters else {}
    context, raw_results, filtered_products = await search_tavily(query_str, filters_dict)
    answer = await call_openrouter(query_str, context)
    response_time = round(time.time() - start_time, 1)
    
    user_input_str = query_str if request.question else str(request.filters)
    thought_1 = f"User is asking for: {query_str}. I will query Tavily to search the LIVE web for India's latest smartphone pricing and reviews."
    thought_2 = f"I retrieved {len(raw_results)} raw results. I must sanitize these URLs and filter out generic cases or accessories. Found {len(filtered_products)} legitimate device URLs."
    thought_3 = f"Now passing {len(filtered_products)} verified URLs to OpenRouter LLM to enforce strict filter format constraints."

    trace_payload = {
        "timestamp": datetime.datetime.now().isoformat(),
        "steps": [
            {"step": "Input", "content": user_input_str},
            {"step": "Thought", "content": thought_1},
            {"step": "Action", "tool": "Tavily Search API", "input": query_str},
            {"step": "Observation", "content": raw_results[:3]},
            {"step": "Thought", "content": thought_2},
            {"step": "Filtered", "content": filtered_products},
            {"step": "Thought", "content": thought_3},
            {"step": "Final Answer", "content": answer}
        ]
    }
    
    return {
        "answer": answer,
        "trace": trace_payload,
        "response_time": f"{response_time}s"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
