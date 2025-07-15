# backend/config.py

import os
from dotenv import load_dotenv
from matplotlib import rcParams, cycler
import matplotlib.font_manager as fm

# This line loads the variables from your .env file into the environment
load_dotenv()

# --- Project Settings from Environment Variables ---
# os.getenv will read the variable. If not found, it uses the default value after the comma.
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
VERTEXAI_REGION = os.getenv("GCP_REGION")
VERTEX_SEARCH_URL = os.getenv("DISCOVERY_ENGINE_URL")
STATIC_EMAIL = os.getenv("STATIC_EMAIL_ADDRESS")

# --- Model Names from Environment Variables ---
KUWAITI_CHAT_MODEL = os.getenv("KUWAITI_CHAT_MODEL")
CODE_EXECUTION_MODEL = os.getenv("CODE_EXECUTION_MODEL")
PLANNER_MODEL = os.getenv("PLANNER_MODEL")

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_HOST = os.getenv("DB_HOST") 

# --- Matplotlib Style Setup ---
def setup_matplotlib_style():
    """Sets a professional style for Matplotlib charts."""
    font_path = "Amiri-Regular.ttf"
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        # A professional, corporate color palette
        PROFESSIONAL_COLORS = ['#0b6efd', '#F59E0B', '#198754', '#6F42C1', '#DC3545', '#6C757D']
        rcParams.update({
            'font.family': fm.FontProperties(fname=font_path).get_name(),
            'axes.prop_cycle': cycler(color=PROFESSIONAL_COLORS),
            'axes.facecolor': '#F8F9FA',
            'figure.facecolor': 'white',
            'axes.grid': True,
            'grid.color': '#DEE2E6',
            'grid.linestyle': '--',
            'grid.linewidth': 0.8,
            'axes.titlepad': 20,
            'axes.labelpad': 15,
            'xtick.labelsize': 'medium',
            'ytick.labelsize': 'medium',
            'axes.titlesize': 'x-large',
            'axes.labelsize': 'large',
            'legend.fontsize': 'medium',
            'font.size': 14,
            'axes.unicode_minus': False,
            'figure.dpi': 150,
        })
        print("INFO: Matplotlib style configured.")
    else:
        print("WARN: Font file 'Amiri-Regular.ttf' not found. Arabic text in charts may not render correctly.")