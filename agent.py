import os
from dotenv import load_dotenv
from openai import OpenAI

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from tavily import TavilyClient

# =========================
# STEP 1 — Load API Keys
# =========================
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY missing")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY missing")

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

tavily = TavilyClient(api_key=TAVILY_API_KEY)

# =========================
# STEP 2 — CLEAN TITLE
# =========================
def clean_title(title: str):
    title = title.split("|")[0]
    return title.strip()[:80]

# =========================
# STEP 3 — WEB SEARCH
# =========================
TRUSTED_SITES = [
    "amazon.in",
    "flipkart.com",
    "croma.com",
    "reliancedigital.in",
    "vijaysales.com",
    "tatacliq.com"
]

def web_search(query: str):
    response = tavily.search(query=query, max_results=8)

    results = []

    for r in response["results"]:
        url = r["url"]

        # remove review pages
        if "product-reviews" in url:
            continue

        # ✅ only trusted sites
        if any(site in url for site in TRUSTED_SITES):
            results.append({
                "title": clean_title(r["title"]),
                "url": url,
                "content": r["content"]
            })

    return results

# =========================
# STEP 4 — SYSTEM PROMPT
# =========================
SYSTEM_PROMPT = """You are a highly strict smartphone advisor for India.

Your goal is to give accurate, honest, and helpful smartphone recommendations.

=====================
CORE RULES (MANDATORY)
=====================

- NEVER recommend products that DO NOT match user filters strongly.
- For impossible combinations (e.g., Apple under ₹20,000, 5000mAh Apple), DETECT CONSTRAINT CONFLICT.
- DO NOT hallucinate or suggest wrong products that meet only one criteria if they grossly violate another.
- NEVER break user budget silently or change brand silently.
- ONLY use smartphones from PRODUCT_DATA.
- DO NOT invent smartphone names.

=====================
NO EXACT MATCH HANDLING
=====================

If user constraints are impossible or PRODUCT_DATA doesn't match:
1. Explain WHY the mismatch happened honestly.
2. Guide the user (e.g., increase budget, consider other brands).
3. Provide 2-3 ALTERNATIVE phones that match *most* constraints from PRODUCT_DATA.

OUTPUT FORMAT FOR NO EXACT MATCH:

NO EXACT MATCH FOUND

REASON:
- [Explain the constraint conflict specifically, e.g. Apple phones are not available under ₹20,000]

SUGGESTIONS:
- [Tell user how to adjust filters or alternate brand suggestions]

ALTERNATIVE PHONES:

NAME: <product name>
SPECS:
- Processor:
- RAM:
- Storage:
- Camera:
- Battery:
- Display:
WHY GOOD:
- <Benefit>

(Repeat for 2-3 alternatives)

=====================
NORMAL OUTPUT FORMAT (If constraints CAN be met)
=====================

REASONING:
- 2–3 short lines explaining selection based on user needs

---------------------

RECOMMENDATIONS:

BEST PICK:
NAME: <exact product name>

SPECS:
- Processor:
- RAM:
- Storage:
- Camera:
- Battery:
- Display:

WHY BEST:
- 1 line benefit based on user use case

---------------------

OTHER RECOMMENDATIONS:

NAME: <exact product name>

SPECS:
- Processor:
- RAM:
- Storage:
- Camera:
- Battery:
- Display:

WHY GOOD:
- 1 line benefit

(Repeat for remaining products)

=====================
LINKS (SEPARATE SECTION — VERY IMPORTANT)
=====================

- Links must be separate from recommendations
- ONLY use links from PRODUCT_DATA
- exactly ONE link per product

FORMAT:

LINKS:

<Product Name 1>:
<clickable URL>

<Product Name 2>:
<clickable URL>

=====================
FAIL-SAFE & TONE
=====================
Be honest, transparent, and helpful. Never hallucinate. Focus on helping the user make a confident buying decision.
"""

# =========================
# STEP 5 — MAIN FUNCTION
# =========================
def ask_rag(question):

    # 🔍 Search query (better)
    search_query = f"{question} best smartphone mobile phone India 2025 buy site:amazon.in OR site:flipkart.com"

    results = web_search(search_query)

    if not results:
        return "No reliable data found."

    # =========================
    # Extract + Filter Products
    # =========================
    products = []

    for r in results:
        title = r["title"].lower()

        # ❌ remove accessories / irrelevant items
        if any(x in title for x in ["case", "cover", "protector", "charger", "cable"]):
            continue

        products.append({
            "name": r["title"],
            "link": r["url"]
        })

    # =========================
    # Validation Layer
    # =========================
    valid_products = []

    for p in products:
        if len(p["name"]) < 10:
            continue
        if "review" in p["link"]:
            continue

        valid_products.append(p)

    # keep 3–5 products
    valid_products = valid_products[:5]

    if len(valid_products) < 2:
        return "Not enough reliable latest smartphones found."

    # =========================
    # LLM (Formatter only)
    # =========================
    response = client.chat.completions.create(
        model="meta-llama/llama-3-8b-instruct",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT + "\n\nPRODUCT_DATA:\n" + str(valid_products)
            },
            {
                "role": "user",
                "content": question
            }
        ],
        temperature=0.1
    )

    return response.choices[0].message.content


# =========================
# STEP 6 — TEST LOOP
# =========================
if __name__ == "__main__":

    while True:
        q = input("\nAsk: ")

        if q.lower() == "exit":
            break

        answer = ask_rag(q)
        print("\nAnswer:\n", answer)
