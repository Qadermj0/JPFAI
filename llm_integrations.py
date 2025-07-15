import httpx
import re
import database
import base64
import requests
from fastapi.concurrency import run_in_threadpool
from vertexai.generative_models import GenerativeModel, Content, Part, GenerationResponse
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import google.auth
import google.auth.transport.requests
from async_lru import alru_cache
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel

import config
from visualizer import execute_python_code, execute_diagram_generation

# Global objects initialized in main.py
http_client: Optional[httpx.AsyncClient] = None
text_answer_model: Optional[GenerativeModel] = None
code_execution_model: Optional[GenerativeModel] = None
planner_model: Optional[GenerativeModel] = None
creds: Optional[google.auth.credentials.Credentials] = None

class ConversationContext(BaseModel):
    last_query: str
    last_response_type: str
    last_response_content: str
    last_raw_data: Any
    last_artifacts: List[int]

def get_text_from_response(response: GenerationResponse) -> str:
    """Safely extracts text from model response."""
    if not response or not response.candidates:
        return ""
    return "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))

def get_fresh_token():
    """Synchronous function to get fresh auth token."""
    if not creds:
        raise Exception("Credentials not initialized")
    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)
    return creds.token

def _sync_query_vertex_search(query: str) -> List[Dict]:
    """Synchronous search function using requests."""
    try:
        fresh_token = get_fresh_token()
        headers = {
            "Authorization": f"Bearer {fresh_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "query": query,
            "pageSize": 5,
            "contentSearchSpec": {
                "extractiveContentSpec": {
                    "maxExtractiveAnswerCount": 5
                }
            }
        }
        response = requests.post(
            config.VERTEX_SEARCH_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        
        results = response.json().get('results', [])
        structured_context = []
        for res in results:
            doc = res.get('document', {})
            title = doc.get('derivedStructData', {}).get('title', doc.get('name', ''))
            extracts = doc.get('derivedStructData', {}).get('extractive_answers', [])
            content_block = "\n".join(
                re.sub('<[^<]+?>', '', e.get('content', '')) 
                for e in extracts if e.get('content')
            )
            if content_block:
                structured_context.append({
                    "source": title,
                    "content": content_block
                })
        return structured_context
    except Exception as e:
        print(f"❌ SYNC SEARCH ERROR: {e}")
        return [{"source": "Error", "content": "Search failed"}]

@alru_cache(maxsize=128)
async def query_vertex_search(query: str) -> List[Dict]:
    """Async wrapper with caching."""
    return await run_in_threadpool(_sync_query_vertex_search, query)

def format_text_response(raw_text: str) -> str:
    """Cleans up model response text."""
    if not isinstance(raw_text, str):
        return ""
    return "\n\n".join([
        line.strip() 
        for line in raw_text.replace('*', '').splitlines() 
        if line.strip()
    ])

async def rewrite_query_for_search(history: List[Content], query: str) -> str:
    """Rewrites conversational queries for better search results."""
    if not history:
        return query
    if not planner_model: # Added check for planner_model
        print("❌ ERROR (llm_integrations): planner_model not initialized for rewrite_query_for_search.")
        return query # Fallback to original query

    history_text = "\n".join([
        f"{h.role}: {h.parts[0].text}" 
        for h in history
    ])
    
    prompt = f"""You are a search query rewriter. Transform this query into a standalone search query using the chat history. Output ONLY the rewritten query.

**History:**
{history_text}

**Latest Query:** "{query}"

**Rewritten Query:**"""
    
    try:
        response = await planner_model.generate_content_async(prompt)
        rewritten = get_text_from_response(response).strip().replace('"', '')
        print(f"INFO: Rewrote '{query}' to '{rewritten}'")
        return rewritten
    except Exception as e:
        print(f"ERROR: Query rewrite failed: {e}")
        return query

async def get_kuwaiti_answer(query: str, context: List[Dict], history: List[Content]) -> str:
    """Generates answers in Kuwaiti dialect with citations."""
    if not text_answer_model: # Added check for text_answer_model
        print("❌ ERROR (llm_integrations): text_answer_model not initialized for get_kuwaiti_answer.")
        return "عفواً، واجهتني مشكلة داخلية."

    context_text = "\n\n".join([
        f"--- From: {item['source']} ---\n{item['content']}" 
        for item in context
    ]) or "No context found"
    
    history_text = "\n".join([
        f"{'User' if entry.role == 'user' else 'Assistant'}: {entry.parts[0].text}" 
        for entry in history
    ])

    prompt = f"""You are a professional Kuwaiti assistant. Provide a comprehensive answer using the documents.

**History:**
{history_text or "No history"}

**Documents:**
{context_text}

**Query:** {query}

**Instructions:**
1. Answer ONLY based on the user’s query and the documents provided.
2. If the user specifies a **year** (e.g., 2018), return data for that year ONLY.
3. If no year is mentioned, provide a **summary of all available years**.
4. Answer in **Kuwaiti dialect**, naturally and respectfully. Avoid English letters.
5. Use clear formatting: headings, bullets, and proper grouping.
6. Cite the source at the end of each section clearly (e.g., الكتاب السنوي لديوان المحاسبة 2018).
7. If you couldn’t find an answer, say **"ما لقيت معلومات بهالخصوص"** instead of apologizing or making up anything.
8. Do NOT mention that the information is limited or unavailable unless it truly is."""
    
    try:
        chat = text_answer_model.start_chat(history=history)
        response = await chat.send_message_async(prompt)
        return get_text_from_response(response)
    except Exception as e:
        print(f"ERROR in get_kuwaiti_answer: {e}")
        return "عفواً، واجهتني مشكلة أثناء إعداد الجواب."

async def get_chit_chat_response(query: str, history: List[Content]) -> str:
    """Handles casual conversation in Kuwaiti dialect."""
    if not text_answer_model: # Added check
        print("❌ ERROR (llm_integrations): text_answer_model not initialized for get_chit_chat_response.")
        return "أهلاً وسهلاً! (خطأ داخلي)"
    try:
        chat = text_answer_model.start_chat(history=history)
        response = await chat.send_message_async(
            f"You are a friendly Kuwaiti assistant. Respond to this naturally in Kuwaiti dialect only, no translations or any english letters: '{query}'"
        )
        return get_text_from_response(response)
    except Exception as e:
        print(f"Chit-chat error: {e}")
        return "أهلاً وسهلاً!"

async def classify_visual_type(query: str, context: str) -> str:
    """Determines the best visualization type."""
    if not planner_model: # Added check
        print("❌ ERROR (llm_integrations): planner_model not initialized for classify_visual_type.")
        return 'bar_chart' # Fallback
    prompt = f"""Analyze this request and data to choose the best visual type.
Options: `bar_chart`, `line_chart`, `pie_chart`, `table`, `diagram`.

**Data:**
{context}

**Request:**
"{query}"

Respond with ONLY the type name."""
    
    try:
        response = await planner_model.generate_content_async(prompt)
        decision = get_text_from_response(response).strip().lower()
        valid_types = ['bar_chart', 'line_chart', 'pie_chart', 'table', 'diagram']
        return decision if decision in valid_types else 'table'
    except Exception as e:
        print(f"Visual type error: {e}")
        return 'table'

async def generate_visual_content(
    query: str, 
    context: List[Dict], 
    history: List[Content]
) -> Tuple[Optional[bytes], Optional[str]]:
    """Generates visual content with self-correction."""
    if not code_execution_model: # Added check
        print("❌ ERROR (llm_integrations): code_execution_model not initialized for generate_visual_content.")
        return None, "Failed: Code execution model not ready."

    flat_context = "\n\n".join([
        f"--- From: {item['source']} ---\n{item['content']}" 
        for item in context
    ])
    visual_type = await classify_visual_type(query, flat_context)
    
    for attempt in range(2):  # Two attempts
        print(f"Attempt {attempt+1} for {visual_type}")
        
        if visual_type == 'diagram':
            prompt = f"Write complete DOT code for Graphviz to create a diagram for: '{query}'. Context: {flat_context}. Enclose in ```dot...```."
        else:
            prompt = f"""Write Python code to generate a {visual_type} using this data:
---
{flat_context}
---
Request: "{query}"
Instructions:
1. Use ONLY Python in ```python...```
2. Handle Arabic text with reshape_arabic_text()
3. No plt.show()/savefig()"""

        try:
            response = await code_execution_model.generate_content_async(prompt)
            full_text = get_text_from_response(response)

            if visual_type == 'diagram':
                code_match = re.search(r"```dot(.*?)```", full_text, re.DOTALL)
            else:
                code_match = re.search(r"```python(.*?)```", full_text, re.DOTALL)

            if not code_match:
                print(f"WARN: No code found in response:\n{full_text}")
                continue

            code = code_match.group(1).strip()
            
            if visual_type == 'diagram':
                image_bytes, error = await run_in_threadpool(
                    execute_diagram_generation, code
                )
            else:
                image_bytes, error = await run_in_threadpool(
                    execute_python_code, code
                )

            if error:
                print(f"Execution error: {error}")
                continue
            
            return image_bytes, None

        except Exception as e:
            print(f"Visual generation error: {e}")
    
    return None, "Failed to generate visual after 2 attempts"


async def planner_decision(
    query: str, 
    history: List[Content], 
    context: ConversationContext
) -> str:
    """Enhanced intent detection with context awareness."""
    if not planner_model:
        print("❌ ERROR (llm_integrations): planner_model not initialized for planner_decision.")
        return "text_answer"

    lower_query = query.strip().lower()
    
    # Define keywords for different intents
    visual_keywords = ["رسم", "جدول", "صمم", "بيانيا", "صورة", "visual", "chart", "graph", "diagram"]
    email_keywords = ["ايميل", "email", "بريد", "دزها", "ارسل", "ابعت", "send", "mail"]

    # --- Priority 1: Combined Intent (Visual Report AND Email in the SAME query) ---
    is_visual_request_in_query = any(kw in lower_query for kw in visual_keywords)
    is_email_request_in_query = any(kw in lower_query for kw in email_keywords)

    if is_visual_request_in_query and is_email_request_in_query:
        print(f"DEBUG (llm_integrations): Planner detected combined 'visual_report_and_email' request from query keywords.")
        return "visual_report_and_email"

    # --- Priority 2: Implicit References (follow-up requests like "send it") ---
    # This comes after combined requests because a combined request is explicit about what to do.
    # A short query is more likely to be a follow-up.
    # Added "ابعتها" and slightly refined length check.
    if any(kw in lower_query for kw in ["ارسله", "ابعثها", "حطها", "ارسلها","دزها", "ابعتها"]) and len(lower_query.split()) < 5: 
        if context.last_response_type == 'image':
            print(f"DEBUG (llm_integrations): Planner selected 'email_image' based on implicit keywords and last_response_type: {context.last_response_type}")
            return "email_image"
        elif context.last_response_type == 'text':
            print(f"DEBUG (llm_integrations): Planner selected 'email_text' based on implicit keywords and last_response_type: {context.last_response_type}")
            return "email_text"
    
    # --- Priority 3: Explicit Single Intent Requests (if not already handled by combined/implicit) ---
    if is_email_request_in_query:
        print(f"DEBUG (llm_integrations): Planner selected 'email' based on explicit keywords (single intent).")
        return "email" 

    if is_visual_request_in_query:
        print(f"DEBUG (llm_integrations): Planner selected 'visual_report' based on explicit keywords (single intent).")
        return "visual_report"
    
    
    history_lines = "\n".join([f"{h.role}: {h.parts[0].text}" for h in history]) if history else "No history"

    prompt = f"""Classify this query into ONE of:
- 'chit_chat'
- 'text_answer'
- 'visual_report'
- 'email'
- 'visual_report_and_email'
- 'email_text'
- 'email_image'

**History:**
{history_lines}

**Last Output Type:** {context.last_response_type or "None"}

**Query:** "{query}"

Classification:"""
    
    try:
        response = await planner_model.generate_content_async(prompt)
        decision = get_text_from_response(response).strip().lower()
        print(f"DEBUG (llm_integrations): Planner default classification result: '{decision}' for query: '{query}'")
        
        valid_decisions = ['chit_chat', 'text_answer', 'visual_report', 'email', 'visual_report_and_email', 'email_text', 'email_image']
        if decision == 'knowledge_question':
            return "text_answer"
        return decision if decision in valid_decisions else "text_answer"
    except Exception as e:
        print(f"ERROR: Planner error: {e}")
        return "text_answer"

def send_email(
    to_address: str,
    subject: str,
    body: str,
    image_bytes: Optional[bytes] = None
) -> str:
    """Temporarily disabled email sending."""
    print("❌ Email sending is currently disabled.")
    return "ميزة الإيميل معطّلة حالياً."

async def create_and_send_email(
    query: str,
    subject: str,
    history: List[Content],
    image_bytes_to_send: Optional[bytes] = None,
    image_database_id: Optional[int] = None
) -> str:
    """Temporarily disabled email creation and sending."""
    print("❌ create_and_send_email is disabled. Email feature is currently unavailable.")
    return "ميزة الإيميل معطّلة حالياً."
