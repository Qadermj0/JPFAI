# 1. استخدام نسخة بايثون رسمية
FROM python:3.11-slim

# 2. تحديد مجلد العمل داخل الحاوية
WORKDIR /app

# 3. نسخ ملف المكتبات وتثبيتها
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. نسخ باقي ملفات المشروع
COPY . .

# 5. الأمر النهائي لتشغيل الخادم


# Cloud Run يعطينا متغير اسمه $PORT تلقائياً
CMD python -m gunicorn -w 4 -k uvicorn.workers.UvicornWorker --bind "0.0.0.0:${PORT}" main:app
