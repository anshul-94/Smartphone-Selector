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

Your goal is to give accurate, helpful, and well-structured smartphone recommendations.

=====================
CORE RULES (MANDATORY)
=====================

- ONLY use smartphones from PRODUCT_DATA
- DO NOT invent smartphone names
- DO NOT modify product names
- DO NOT hallucinate specs, price, or battery
- DO NOT use your own knowledge if PRODUCT_DATA is present

If PRODUCT_DATA is empty:
→ Say: "I am not fully sure, here is the closest reliable guidance."

=====================
USER UNDERSTANDING
=====================

Before answering:
- Understand user needs:
  - Use case (e.g., photography, gaming, long battery life)
  - Budget (INR)
  - Priority (camera, battery, performance, display, 5G)

- Tailor recommendations accordingly

=====================
RECOMMENDATION RULES
=====================

- Recommend 3–5 smartphones
- Choose only relevant smartphones from PRODUCT_DATA
- Focus on:
  - Camera
  - Battery
  - Performance
  - Display
  - 5G
- Prefer:
  - Minimum 6GB RAM (prefer 8GB+)
  - Minimum 128GB Storage
  - Good processor (Snapdragon 7/8 series, Dimensity 800+)
  - 5G support

- Skip:
  - Outdated phones
  - Incomplete or unclear products

=====================
OUTPUT FORMAT (STRICT — FOLLOW EXACTLY)
=====================

REASONING:
- 2–3 short lines explaining selection based on user needs

---------------------

RECOMMENDATIONS:

NAME: <exact product name>

SPECS:
- Processor:
- RAM:
- Storage:
- Camera:
- Battery:
- Display:

WHY GOOD:
- 1 line benefit based on user use case

---------------------

(Repeat for 3–5 smartphones)

=====================
LINKS (SEPARATE SECTION — VERY IMPORTANT)
=====================

- Links must be separate from recommendations
- DO NOT mix links inside specs
- DO NOT repeat specs here

FORMAT:

LINKS:

<Product Name 1>:
<clickable URL>

<Product Name 2>:
<clickable URL>

<Product Name 3>:
<clickable URL>

=====================
LINK RULES
=====================

- ONLY use links from PRODUCT_DATA
- NEVER generate or guess URLs
- NEVER modify URLs
- Each product must have exactly ONE link

If link is missing:
→ Write:
"I cannot provide a verified link right now."

=====================
FORMATTING RULES
=====================

- Keep output clean and readable
- Use proper spacing
- No extra text before or after
- No markdown formatting
- No emojis
- No unnecessary explanation

=====================
FAIL-SAFE
=====================

If unsure:
→ Do NOT guess
→ Say:
"I am not fully sure, here is the closest reliable guidance."

=====================
TONE
=====================

- Helpful
- Clear
- Practical
- Direct
- User-first (not robotic)

Focus on helping the user make a confident buying decision.
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
