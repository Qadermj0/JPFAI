# backend/visualizer.py
import pandas as pd
import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display
import matplotlib
matplotlib.use('Agg') # مهم جداً للسيرفرات بدون واجهة رسومية
import io
import traceback
from typing import Tuple, Optional

# المكتبات الجديدة بديلة Graphviz
import networkx as nx
import pydot
import matplotlib.pyplot as plt


def reshape_arabic_text(text: str) -> str:
    """
    تجهز النصوص العربية للعرض الصحيح في المكتبات الرسومية.
    """
    return get_display(arabic_reshaper.reshape(str(text)))


def execute_python_code(code: str) -> Tuple[Optional[bytes], Optional[str]]:
    """
    تنفذ كود بايثون لتوليد رسومات matplotlib بجودة عالية وأمان محسّن.
    """
    image_buffer = io.BytesIO()
    try:
        cleaned_code = "\n".join([line for line in code.split('\n')
                                  if "plt.show()" not in line
                                  and "plt.savefig" not in line])
        
        # --- تحسين الأمان ---
        # تقييد الوصول للـ builtins الخطيرة مثل open, exec, eval
        safe_builtins = {
            'print': print, 'len': len, 'range': range, 'list': list, 'dict': dict,
            'str': str, 'int': int, 'float': float, 'bool': bool, 'zip': zip,
            'enumerate': enumerate, 'max': max, 'min': min, 'sum': sum, 'sorted': sorted,
            'abs': abs, 'round': round,
        }
        
        safe_globals = {
            'pd': pd, 'plt': plt, 'np': np,
            'reshape_arabic_text': reshape_arabic_text,
            '__builtins__': safe_builtins
        }
        
        exec(cleaned_code, safe_globals)
        
        # تحسينات جودة المخرجات
        plt.gcf().tight_layout(pad=1.5)
        plt.savefig(
            image_buffer,
            format='png',
            dpi=300,
            bbox_inches='tight',
            facecolor='white',
            transparent=False
        )
        
        image_buffer.seek(0)
        image_bytes = image_buffer.read()
        
        if not image_bytes or len(image_bytes) < 100:
            return None, "The generated image was empty."
        
        print("INFO: High-quality Matplotlib image generated successfully.")
        return image_bytes, None
            
    except Exception as e:
        error_message = f"Error during Python code execution: {str(e)}\n{traceback.format_exc()}"
        print(f"ERROR: {error_message}")
        return None, error_message
    finally:
        # تأكد من إغلاق كل الأشكال الرسومية لتوفير الذاكرة
        plt.close('all')


# --- الدالة الجديدة كلياً: بديل Graphviz ---
def execute_diagram_generation(dot_script: str) -> Tuple[Optional[bytes], Optional[str]]:
    """
    تنفذ سكربت DOT باستخدام مكتبات بايثون فقط (NetworkX & Matplotlib)
    لتجنب الاعتماد على تثبيت Graphviz على النظام.
    """
    print("INFO: Generating diagram using pure Python libraries (NetworkX, Matplotlib)...")
    image_buffer = io.BytesIO()
    try:
        # 1. قراءة سكربت الـ DOT باستخدام pydot
        # pydot قادر على تحليل لغة DOT وهو مكتبة بايثون صافية
        pydot_graphs = pydot.graph_from_dot_data(dot_script)
        if not pydot_graphs:
            return None, "Failed to parse DOT script."
        pydot_graph = pydot_graphs[0]

        # 2. تحويل الرسمة من pydot إلى NetworkX
        # NetworkX مكتبة قوية جداً للتعامل مع الرسوم البيانية
        nx_graph = nx.drawing.nx_pydot.from_pydot(pydot_graph)

        # 3. رسم الـ graph باستخدام Matplotlib
        plt.figure(figsize=(12, 8)) # حجم أكبر للرسمة

        # تحديد布局 (layout) للعقد
        pos = nx.spring_layout(nx_graph, seed=42)
        
        # استخراج العناوين والألوان من خصائص الرسمة الأصلية (إن وجدت)
        node_labels = {node: data.get('label', node) for node, data in nx_graph.nodes(data=True)}
        # تطبيق تشكيل النص العربي على العناوين
        reshaped_labels = {node: reshape_arabic_text(label) for node, label in node_labels.items()}
        
        nx.draw(
            nx_graph, 
            pos,
            labels=reshaped_labels,
            with_labels=True,
            node_color='skyblue',
            node_size=3000,
            edge_color='gray',
            font_size=10,
            font_family='DejaVu Sans' # خط يدعم العربية بشكل أفضل
        )

        # 4. حفظ الرسمة كصورة في الذاكرة
        plt.savefig(image_buffer, format='png', dpi=300, bbox_inches='tight')
        image_buffer.seek(0)
        image_bytes = image_buffer.read()

        if not image_bytes or len(image_bytes) < 100:
            return None, "The generated diagram was empty."

        print("INFO: Pure Python diagram generated successfully.")
        return image_bytes, None

    except Exception as e:
        error_message = f"ERROR: Unexpected error in diagram generation: {e}\n{traceback.format_exc()}"
        print(error_message)
        return None, error_message
    finally:
        plt.close('all')