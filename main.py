import uvicorn
import vertexai
import httpx
import json
import base64
import asyncio
import google.auth
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager # <--- إضافة مهمة للـ Lifespan الجديد

# استيراد الملفات الخاصة بالمشروع
import config
import database
import llm_integrations as llm
from config import setup_matplotlib_style

# Enhanced context tracking system
class ConversationContext(BaseModel):
    last_query: str
    last_response_type: str  # 'image', 'text', 'email', etc.
    last_response_content: str
    last_raw_data: Any
    last_artifacts: list  # List of artifact IDs

conversation_context_cache: Dict[int, ConversationContext] = {}

# --- بداية التعديلات هنا ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events using the new pool connection.
    """
    print("--- Application Startup ---")

    # 1. تهيئة الـ Connection Pool لقاعدة البيانات الجديدة
    await database.init_db_pool()
    
    # 2. (اختياري) التأكد من وجود الجداول وإنشائها إذا لزم الأمر
    await database.setup_database_tables()
    
    # باقي كود بدء التشغيل كما هو
    setup_matplotlib_style()
    
    print("INFO: Initializing Google Cloud credentials...")
    llm.creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform", "https://www.googleapis.com/auth/gmail.send"])
    print("INFO: Credentials initialized.")

    print("INFO: Initializing Vertex AI models...")
    vertexai.init(project=config.PROJECT_ID, location=config.VERTEXAI_REGION)
    llm.text_answer_model = llm.GenerativeModel(config.KUWAITI_CHAT_MODEL)
    llm.code_execution_model = llm.GenerativeModel(config.CODE_EXECUTION_MODEL)
    llm.planner_model = llm.GenerativeModel(config.PLANNER_MODEL)
    print("INFO: Models initialized.")

    llm.http_client = httpx.AsyncClient(http2=True, timeout=40.0)
    
    print("✅ Application startup complete.")
    yield
    
    # --- الكود الذي يعمل عند إيقاف تشغيل التطبيق ---
    print("--- Application Shutdown ---")
    
    # 3. إغلاق الـ Connection Pool بشكل آمن
    await database.close_db_pool()
    
    if llm.http_client and not llm.http_client.is_closed:
        await llm.http_client.aclose()
    conversation_context_cache.clear()
    print("✅ Resources cleaned up.")
# --- نهاية التعديلات هنا ---


app = FastAPI(title="Watira Pro - Final Backend", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

response_queue = asyncio.Queue()

class RenameRequest(BaseModel):
    title: str

class ChatRequest(BaseModel):
    query: str
    conversation_id: Optional[int] = None

@app.get("/")
def read_root():
    return {"status": "Watira Pro Backend is running!"}

# --- لا يوجد أي تعديل على الـ Endpoints ---
# الكود التالي يبقى كما هو تماماً لأنك تستدعي دوال من database.py
# والتعديل سيتم على محتوى تلك الدوال وليس على أسمائها
# --------------------------------------------------

@app.get("/conversations")
async def get_conversations_list():
    return await database.get_all_conversations()

@app.get("/conversations/{conversation_id}")
async def get_conversation_messages(conversation_id: int):
    return await database.get_messages_for_conversation(conversation_id)

@app.put("/conversations/{conversation_id}")
async def http_rename_conversation(conversation_id: int, request: RenameRequest):
    await database.rename_conversation(conversation_id, request.title)
    return {"status": "success"}

@app.delete("/conversations/{conversation_id}")
async def http_delete_conversation(conversation_id: int):
    await database.delete_conversation(conversation_id)
    if conversation_id in conversation_context_cache:
        del conversation_context_cache[conversation_id]
    return {"status": "success"}


@app.post("/chat")
async def process_chat_message(request: ChatRequest):
    """Enhanced chat processing with advanced context tracking"""
    structured_context = None
    query = request.query.strip()
    cid = request.conversation_id

    if not query:
        return {"status": "empty query"}

    # Initialize or retrieve conversation
    if cid is None:
        cid = await database.create_new_conversation(first_query=query)
        await response_queue.put(json.dumps({
            "type": "session_created",
            "conversation_id": cid
        }))
        history = []
        conversation_context_cache[cid] = ConversationContext(
            last_query="",
            last_response_type="",
            last_response_content="",
            last_raw_data=None,
            last_artifacts=[]
        )
    else:
        history = await database.get_conversation_history(cid)
        if cid not in conversation_context_cache:
            conversation_context_cache[cid] = ConversationContext(
                last_query="",
                last_response_type="",
                last_response_content="",
                last_raw_data=None,
                last_artifacts=[]
            )

    await database.add_message_to_history(cid, "user", query)

    # Get current context
    context = conversation_context_cache[cid]
    print(f"DEBUG (main.py): Current context for CID {cid}: last_query='{context.last_query}', last_response_type='{context.last_response_type}', last_artifacts={context.last_artifacts}")


    # Enhanced intent detection
    tool = await llm.planner_decision(query, history, context)
    print(f"INFO: Planner decided tool: '{tool}'")

    final_response_for_db = ""
    rewritten_query = query

    # Process based on tool
    if tool == "chit_chat":
        await response_queue.put(json.dumps({
            "type": "status",
            "message": "Typing...",
            "conversation_id": cid
        }))
        final_response_for_db = await llm.get_chit_chat_response(query, history)
        await response_queue.put(json.dumps({
            "type": "text",
            "content": final_response_for_db,
            "conversation_id": cid
        }))
        context.last_response_type = "text" # Update context for chit_chat
        context.last_response_content = final_response_for_db # Important for potential follow-up text emails
        context.last_query = query


    elif tool == "text_answer":
        await response_queue.put(json.dumps({
            "type": "status",
            "message": "Searching documents...",
            "conversation_id": cid
        }))
        structured_context = await llm.query_vertex_search(rewritten_query)
        await response_queue.put(json.dumps({
            "type": "status",
            "message": "Drafting response...",
            "conversation_id": cid
        }))
        current_output = await llm.get_kuwaiti_answer(query, structured_context, history)
        await response_queue.put(json.dumps({
            "type": "text",
            "content": current_output,
            "conversation_id": cid
        }))

        final_response_for_db = llm.format_text_response(current_output) # Use current_output for final_response_for_db

        context.last_query = rewritten_query
        context.last_response_type = "text"
        context.last_response_content = final_response_for_db
        context.last_raw_data = structured_context
        context.last_artifacts = []

    elif tool == "visual_report":
        # Determine the query to use for visual generation (original or from context)
        if any(kw in query.lower() for kw in ["it", "them", "ارسمها", "صممها"]) and context.last_query:
            rewritten_query = context.last_query
            print(f"INFO: Using last query context: '{rewritten_query}' for visual_report.")
        else:
            rewritten_query = query # If no context, use original query
            print(f"INFO: Using current query: '{rewritten_query}' for visual_report.")


        await response_queue.put(json.dumps({
            "type": "status",
            "message": "Checking cache for visual...",
            "conversation_id": cid
        }))
        cached_image_data = await database.get_artifact(cid, rewritten_query)

        if cached_image_data:
            final_response_for_db = cached_image_data
            await response_queue.put(json.dumps({
                "type": "image",
                "content": final_response_for_db,
                "conversation_id": cid
            }))
            # If from cache, we don't re-save. The artifact_id should already be in the DB.
            # We assume if it was cached, its ID was saved previously.
            # To be robust, `get_artifact` could return the ID.
            context.last_query = rewritten_query
            context.last_response_type = "image"
            context.last_response_content = final_response_for_db
            print(f"DEBUG (main.py): Image served from cache. context.last_artifacts: {context.last_artifacts}")
        else:
            await response_queue.put(json.dumps({
                "type": "status",
                "message": "Generating visualization...",
                "conversation_id": cid
            }))
            structured_context = None
            try:
                structured_context = await llm.query_vertex_search(rewritten_query)
            except Exception as e:
                print(f"ERROR (main.py): Search for visual_report failed: {e}")
                structured_context = []

            image_bytes, error = await llm.generate_visual_content(
                rewritten_query, structured_context, history
            )

            if image_bytes:
                encoded_image = base64.b64encode(image_bytes).decode('utf-8')
                final_response_for_db = f"data:image/png;base64,{encoded_image}"
                artifact_id = await database.save_artifact(
                    cid, rewritten_query, "image", final_response_for_db
                )

                await response_queue.put(json.dumps({
                    "type": "image",
                    "content": final_response_for_db,
                    "conversation_id": cid
                }))

                context.last_artifacts = [artifact_id]
                print(f"DEBUG (main.py): New image generated and saved. Artifact ID: {artifact_id}. context.last_artifacts: {context.last_artifacts}")
            else:
                final_response_for_db = f"Sorry, I couldn't create the visualization. Error: {error}"
                await response_queue.put(json.dumps({
                    "type": "text",
                    "content": final_response_for_db,
                    "conversation_id": cid
                }))
                context.last_artifacts = []

        context.last_query = rewritten_query
        context.last_response_type = "image"
        context.last_response_content = final_response_for_db
        context.last_raw_data = structured_context

    elif tool == "email_image": 
        print("DEBUG (main.py): Handling 'email_image' tool.")
        if not context.last_artifacts:
            final_response_for_db = "عفواً، لا توجد صورة حديثة لإرسالها بالبريد."
            print("DEBUG (main.py): No recent image artifacts found for 'email_image'.")
        else:
            artifact_id = context.last_artifacts[0]
            print(f"DEBUG (main.py): Attempting to retrieve artifact ID {artifact_id} for email_image.")
            image_data = await database.get_artifact_by_id(artifact_id)
            
            if image_data and "," in image_data:
                try:
                    image_bytes_to_send = base64.b64decode(image_data.split(",")[1])
                    print(f"DEBUG (main.py): Decoded image for 'email_image'. Bytes length: {len(image_bytes_to_send)}")
                    email_result = await llm.create_and_send_email(
                        query="إرسال الصورة المطلوبة",
                        subject="الصورة المطلوبة من JPFA Assistant AI",
                        history=history,
                        image_bytes_to_send=image_bytes_to_send,
                    )
                    final_response_for_db = email_result
                    print(f"DEBUG (main.py): email_image result: {final_response_for_db}")
                except Exception as e:
                    final_response_for_db = f"حدث خطأ أثناء تجهيز أو إرسال الصورة عبر الإيميل: {e}"
                    print(f"ERROR (main.py): Error processing image for 'email_image': {e}")
            else:
                final_response_for_db = "عفواً، تعذر الوصول إلى بيانات الصورة لإرسالها."
                print("ERROR (main.py): Image data invalid or not found for 'email_image'.")

        await response_queue.put(json.dumps({
            "type": "text",
            "content": final_response_for_db,
            "conversation_id": cid
        }))

    elif tool == "email_text":
        print("DEBUG (main.py): Handling 'email_text' tool.")
        if context.last_response_type == 'text' and context.last_response_content:
            email_result = await llm.create_and_send_email(
                query=query, 
                subject="محتوى نصي من محادثتك مع JPFA Assistant AI",
                history=history,
                image_bytes_to_send=None
            )
            final_response_for_db = email_result
            print(f"DEBUG (main.py): email_text result: {final_response_for_db}")
        else:
            final_response_for_db = "عفواً، لا توجد إجابة نصية حديثة لإرسالها بالبريد."
            print("DEBUG (main.py): No recent text content found for 'email_text'.")
        
        await response_queue.put(json.dumps({
            "type": "text",
            "content": final_response_for_db,
            "conversation_id": cid
        }))

    elif tool == "email":
        print(f"DEBUG (main.py): Handling general 'email' tool. Context last_response_type: {context.last_response_type}")
        email_sent_status = False 

        if context.last_response_type == 'image' and context.last_artifacts:
            artifact_id = context.last_artifacts[0]
            print(f"DEBUG (main.py): 'email' tool: Found previous image artifact with ID {artifact_id}. Attempting to send.")
            image_data = await database.get_artifact_by_id(artifact_id)
            if image_data and "," in image_data:
                try:
                    image_bytes_to_send = base64.b64decode(image_data.split(",")[1])
                    email_result = await llm.create_and_send_email(
                        query=query, 
                        subject="مرفق من محادثتك مع JPFA Assistant AI",
                        history=history,
                        image_bytes_to_send=image_bytes_to_send,
                    )
                    final_response_for_db = email_result
                    await response_queue.put(json.dumps({
                        "type": "text",
                        "content": final_response_for_db,
                        "conversation_id": cid
                    }))
                    print(f"DEBUG (main.py): 'email' tool (image): {final_response_for_db}")
                    email_sent_status = True
                except Exception as e:
                    final_response_for_db = f"حدث خطأ في تجهيز وإرسال الصورة عبر الإيميل: {e}"
                    print(f"ERROR (main.py): 'email' tool (image) processing error: {e}")
                    await response_queue.put(json.dumps({
                        "type": "text",
                        "content": final_response_for_db,
                        "conversation_id": cid
                    }))
            else:
                final_response_for_db = "عفواً، لم أجد صورة سابقة صالحة لإرسالها بالبريد."
                print("DEBUG (main.py): 'email' tool: No valid image data found from previous context.")
        
        if not email_sent_status: 
            if context.last_response_type == 'text' and context.last_response_content:
                print(f"DEBUG (main.py): 'email' tool: Found previous text response. Attempting to send as text email.")
                email_result = await llm.create_and_send_email(
                    query=query, 
                    subject="تفاصيل من محادثتك مع JPFA Assistant AI",
                    history=history,
                    image_bytes_to_send=None
                )
                final_response_for_db = email_result
                await response_queue.put(json.dumps({
                    "type": "text",
                    "content": final_response_for_db,
                    "conversation_id": cid
                }))
                print(f"DEBUG (main.py): 'email' tool (text): {final_response_for_db}")
                email_sent_status = True
            elif not final_response_for_db: 
                print("DEBUG (main.py): 'email' tool: No clear previous context (or image failed, or no text to send). Asking LLM for general text email content.")
                email_result = await llm.create_and_send_email(
                    query=query, 
                    subject=f"استفسار بخصوص: {query}",
                    history=history,
                    image_bytes_to_send=None 
                )
                final_response_for_db = email_result
                await response_queue.put(json.dumps({
                    "type": "text",
                    "content": final_response_for_db,
                    "conversation_id": cid
                }))
                print(f"DEBUG (main.py): 'email' tool (general text): {final_response_for_db}")
                email_sent_status = True

        if not email_sent_status and not final_response_for_db: 
             final_response_for_db = "عفواً، لم أتمكن من تحديد ما يجب إرساله بالبريد."
             await response_queue.put(json.dumps({
                "type": "text",
                "content": final_response_for_db,
                "conversation_id": cid
            }))
             print("ERROR (main.py): 'email' tool: No email logic executed, defaulting final_response_for_db.")


    elif tool == "visual_report_and_email": # This block now handles the combined request
        print(f"DEBUG (main.py): Handling 'visual_report_and_email' tool for query: '{query}'")
        await response_queue.put(json.dumps({
            "type": "status",
            "message": "analyzing...", # New status message
            "conversation_id": cid
        }))

        # 1. Generate the visual report
        structured_context_for_visual = None 
        try:
            structured_context_for_visual = await llm.query_vertex_search(query) 
        except Exception as e:
            print(f"ERROR (main.py): Search failed for visual_report_and_email visual generation: {e}")
            structured_context_for_visual = [] 
        
        await response_queue.put(json.dumps({
            "type": "status",
            "message": "Generating visual ...", # New status message
            "conversation_id": cid
        }))

        image_bytes, error = await llm.generate_visual_content(
            query, structured_context_for_visual, history 
        )

        final_response_for_db = ""
        if image_bytes:
            encoded_image = base64.b64encode(image_bytes).decode('utf-8')
            image_b64_url = f"data:image/png;base64,{encoded_image}"
            artifact_id = await database.save_artifact(
                cid, query, "image", image_b64_url 
            )
            context.last_artifacts = [artifact_id] 
            context.last_response_type = "image"
            context.last_response_content = image_b64_url
            context.last_query = query 
            print(f"DEBUG (main.py): Image generated for 'visual_report_and_email'. Artifact ID: {artifact_id}")

            await response_queue.put(json.dumps({
                "type": "image",
                "content": image_b64_url,
                "conversation_id": cid
            }))
            await response_queue.put(json.dumps({
                "type": "status",
                "message": "Visual report generated, preparing to send email...", # New status message
                "conversation_id": cid
            }))

            # 2. Now send the email with the generated image
            try:
                email_result = await llm.create_and_send_email(
                    query=query, 
                    subject=f"التقرير المطلوب: {query[:50]}...", 
                    history=history,
                    image_bytes_to_send=image_bytes, 
                )
                final_response_for_db = email_result
                print(f"DEBUG (main.py): 'visual_report_and_email' - Email sent result: {final_response_for_db}")
            except Exception as e:
                final_response_for_db = f"تم توليد الصورة، لكن فشل إرسالها بالبريد: {e}"
                print(f"ERROR (main.py): 'visual_report_and_email' - Email sending failed: {e}")

            await response_queue.put(json.dumps({
                "type": "text", 
                "content": final_response_for_db,
                "conversation_id": cid
            }))

        else: # Image generation failed for visual_report_and_email
            final_response_for_db = f"عفواً، فشلت في توليد الصورة المطلوبة لإرسالها بالبريد: {error}"
            print(f"ERROR (main.py): 'visual_report_and_email' - Image generation failed: {error}")
            await response_queue.put(json.dumps({
                "type": "text",
                "content": final_response_for_db,
                "conversation_id": cid
            }))
            context.last_response_type = "text"
            context.last_response_content = final_response_for_db
            context.last_artifacts = []
    
    if not final_response_for_db:
        final_response_for_db = "Sorry, I couldn't process your request for some reason."
        print(f"ERROR (main.py): final_response_for_db was not set for tool '{tool}'. Defaulting.")

    await database.add_message_to_history(cid, "model", final_response_for_db)
    conversation_context_cache[cid] = context

    return {"status": "ok"}

@app.get("/stream")
async def message_stream(request: Request):
    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            try:
                message = await asyncio.wait_for(response_queue.get(), timeout=30.0)
                yield {"data": message}
            except asyncio.TimeoutError:
                yield {"data": ":"} 
    
    return EventSourceResponse(event_generator())