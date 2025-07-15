import asyncpg
import asyncio
from typing import Optional, List, Dict, Any
from vertexai.generative_models import Content, Part
from google.cloud.sql.connector import Connector
# استيراد الإعدادات من ملف config.py
# لم نعد بحاجة لاسم ملف قاعدة البيانات هنا
import config

# هذا المتغير سيحتوي على الـ Connection Pool
DB_POOL = None


# --------------------------------------------------------------------------
# دوال إدارة الاتصال (Connection Pool) - لا تعدل عليها
# --------------------------------------------------------------------------

async def init_db_pool():
    global DB_POOL
    if DB_POOL is None:
        try:
            connector = Connector()
            
            def get_conn():
                return connector.connect(
                    "watira-genai:us-central1:watira-db",  # معرّف Cloud SQL
                    "asyncpg",
                    user=config.DB_USER,
                    password=config.DB_PASSWORD,
                    db=config.DB_NAME
                )
            
            DB_POOL = await asyncpg.create_pool(
                connect=get_conn,
                min_size=1,
                max_size=10
            )
            print("✅ تم إنشاء connection pool بنجاح")
        except Exception as e:
            print(f"❌ فشل إنشاء اتصال بقاعدة البيانات: {e}")
            raise  # لإظهار الخطأ في السجلات

async def close_db_pool():
    """
    تُغلق الـ connection pool عند إيقاف تشغيل التطبيق.
    """
    global DB_POOL
    if DB_POOL:
        await DB_POOL.close()
        print("INFO: Database connection pool closed.")

# --------------------------------------------------------------------------
# دالة إعداد الجداول - تم تعديلها لـ PostgreSQL
# --------------------------------------------------------------------------
async def setup_database_tables():
    """
    تُنشئ الجداول المطلوبة في قاعدة بيانات PostgreSQL إذا لم تكن موجودة.
    """
    async with DB_POOL.acquire() as connection:
        # جدول المحادثات
        # SERIAL PRIMARY KEY: هو المعادل لـ AUTOINCREMENT في PostgreSQL
        # TIMESTAMPTZ: يخزن الوقت مع المنطقة الزمنية، وهي ممارسة أفضل
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id SERIAL PRIMARY KEY,
                title TEXT NOT NULL DEFAULT 'New Conversation',
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # جدول الرسائل
        # ON DELETE CASCADE: عند حذف محادثة، يتم حذف كل رسائلها تلقائياً
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id SERIAL PRIMARY KEY,
                conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # جدول الملفات (Artifacts)
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS artifacts (
                id SERIAL PRIMARY KEY,
                conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                request_query TEXT NOT NULL,
                artifact_type TEXT NOT NULL,
                artifact_content TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
        """)
    print("✅ Tables checked/created successfully in PostgreSQL.")


# --------------------------------------------------------------------------
# دوال التعامل مع البيانات - تم تعديلها بالكامل
# --------------------------------------------------------------------------

async def get_all_conversations() -> List[Dict]:
    """
    تجلب كل المحادثات مرتبة حسب آخر تحديث.
    """
    async with DB_POOL.acquire() as conn:
        records = await conn.fetch("""
            SELECT id, title, created_at, updated_at 
            FROM conversations 
            ORDER BY updated_at DESC
        """)
        return [dict(row) for row in records]

async def create_new_conversation(first_query: str) -> int:
    """
    تُنشئ محادثة جديدة وتستخدم أول استعلام كعنوان لها.
    RETURNING id: طريقة PostgreSQL لإرجاع قيمة الـ ID الجديد بعد الإضافة.
    """
    title = (first_query[:50] + '...') if len(first_query) > 50 else first_query
    
    query = "INSERT INTO conversations (title) VALUES ($1) RETURNING id;"
    async with DB_POOL.acquire() as conn:
        # fetchval ترجع قيمة واحدة مباشرة
        new_id = await conn.fetchval(query, title)
        return new_id
        
async def rename_conversation(conversation_id: int, new_title: str):
    """
    تُحدّث عنوان المحادثة وتاريخ التحديث.
    استخدام $1, $2, ... لتمرير المتغيرات بشكل آمن.
    """
    query = """
        UPDATE conversations 
        SET title = $1, updated_at = CURRENT_TIMESTAMP 
        WHERE id = $2
    """
    async with DB_POOL.acquire() as conn:
        await conn.execute(query, new_title, conversation_id)

async def delete_conversation(conversation_id: int):
    """
    تحذف محادثة. بفضل ON DELETE CASCADE، سيتم حذف كل الرسائل والملفات المرتبطة تلقائياً.
    """
    query = "DELETE FROM conversations WHERE id = $1"
    async with DB_POOL.acquire() as conn:
        await conn.execute(query, conversation_id)

async def add_message_to_history(conversation_id: int, role: str, content: str):
    """
    تضيف رسالة إلى سجل المحادثة وتُحدّث تاريخ آخر نشاط للمحادثة.
    """
    async with DB_POOL.acquire() as conn:
        # استخدام Transaction لضمان تنفيذ العمليتين معاً
        async with conn.transaction():
            await conn.execute(
                "INSERT INTO messages (conversation_id, role, content) VALUES ($1, $2, $3)",
                conversation_id, role, content
            )
            await conn.execute(
                "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = $1",
                conversation_id
            )

async def get_messages_for_conversation(conversation_id: int) -> List[Dict]:
    """
    تجلب كل الرسائل لمحاثة معينة.
    """
    query = """
        SELECT role, content, timestamp 
        FROM messages 
        WHERE conversation_id = $1
        ORDER BY timestamp ASC
    """
    async with DB_POOL.acquire() as conn:
        records = await conn.fetch(query, conversation_id)
        return [dict(row) for row in records]

async def get_conversation_history(conversation_id: int, limit: int = 20) -> List[Content]:
    """
    تجلب سجل المحادثة على شكل كائنات Vertex AI Content.
    """
    query = """
        SELECT role, content 
        FROM messages 
        WHERE conversation_id = $1
        ORDER BY timestamp DESC 
        LIMIT $2
    """
    async with DB_POOL.acquire() as conn:
        rows = await conn.fetch(query, conversation_id, limit)
        return [
            Content(
                role=row['role'],
                parts=[Part.from_text(
                    "Here is the visual" if row['content'].startswith("data:image") 
                    else row['content']
                )]
            )
            for row in reversed(rows) # عكس الترتيب ليكون تسلسل زمني صحيح
        ]

async def save_artifact(conversation_id: int, request_query: str, artifact_type: str, artifact_content: str) -> int:
    """
    تحفظ ملف (artifact) مُنشأ وتُرجع الـ ID الخاص به.
    """
    query = """
        INSERT INTO artifacts (
            conversation_id, request_query, artifact_type, artifact_content
        ) VALUES ($1, $2, $3, $4) RETURNING id
    """
    async with DB_POOL.acquire() as conn:
        artifact_id = await conn.fetchval(query, conversation_id, request_query, artifact_type, artifact_content)
        return artifact_id

async def get_artifact(conversation_id: int, request_query: str) -> Optional[str]:
    """
    تجلب ملف (artifact) من الكاش عبر مطابقة تقريبية للاستعلام.
    """
    query = """
        SELECT artifact_content 
        FROM artifacts 
        WHERE conversation_id = $1 AND request_query LIKE $2
        ORDER BY created_at DESC 
        LIMIT 1
    """
    async with DB_POOL.acquire() as conn:
        # نرسل الـ query مع % من جهة البايثون
        content = await conn.fetchval(query, conversation_id, f"%{request_query}%")
        return content

async def get_artifact_by_id(artifact_id: int) -> Optional[str]:
    """
    تجلب ملف (artifact) عبر الـ ID الدقيق.
    """
    query = "SELECT artifact_content FROM artifacts WHERE id = $1"
    async with DB_POOL.acquire() as conn:
        content = await conn.fetchval(query, artifact_id)
        return content