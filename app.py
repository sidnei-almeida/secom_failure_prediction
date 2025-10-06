import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import io

# Configuração de cores para gráficos (Design System)
PLOT_COLORS = {
    'bg': 'rgba(0,0,0,0)',  # Fundo transparente para usar o tema do Streamlit
    'primary': '#ff6b35',    # Laranja quente
    'secondary': '#f7931e',  # Laranja médio
    'tertiary': '#ffb627',   # Amarelo dourado
    'success': '#6bff6b',    # Verde para status normal
    'grid': 'rgba(255, 107, 53, 0.08)',  # Grid sutil
    'text': '#fafafa',       # Texto claro
    'font': 'Inter',         # Fonte consistente
    'font_size': 11          # Tamanho de fonte padrão para gráficos
}

# Configuração da página
st.set_page_config(
    page_title="SECOM Failure Prediction - Anomaly Detection System",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado - Design Profissional com Acent os Quentes
st.markdown("""
    <style>
    /* Importar fonte moderna e profissional */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Design System - Variáveis CSS */
    :root {
        /* Paleta de Acentos Quentes */
        --accent-primary: #ff6b35;
        --accent-secondary: #f7931e;
        --accent-tertiary: #ffb627;
        --accent-gradient: linear-gradient(135deg, #ff6b35 0%, #f7931e 50%, #ffb627 100%);
        
        /* Efeitos Visuais */
        --glow-sm: 0 0 8px rgba(255, 107, 53, 0.12);
        --glow-md: 0 0 16px rgba(255, 107, 53, 0.2);
        --glow-lg: 0 0 24px rgba(255, 107, 53, 0.3);
        --shadow-sm: 0 2px 8px rgba(255, 107, 53, 0.08);
        --shadow-md: 0 4px 16px rgba(255, 107, 53, 0.12);
        --shadow-lg: 0 8px 24px rgba(255, 107, 53, 0.16);
        
        /* Bordas */
        --border-subtle: rgba(255, 107, 53, 0.12);
        --border-accent: rgba(255, 107, 53, 0.35);
        
        /* Espaçamentos Padronizados */
        --spacing-xs: 0.25rem;
        --spacing-sm: 0.5rem;
        --spacing-md: 1rem;
        --spacing-lg: 1.5rem;
        --spacing-xl: 2rem;
        --spacing-2xl: 3rem;
        
        /* Tipografia - Hierarquia Refinada */
        --font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        --font-size-xs: 0.6875rem;    /* 11px */
        --font-size-sm: 0.8125rem;    /* 13px */
        --font-size-base: 0.9375rem;  /* 15px */
        --font-size-lg: 1.0625rem;    /* 17px */
        --font-size-xl: 1.25rem;      /* 20px */
        --font-size-2xl: 1.5rem;      /* 24px */
        --font-size-3xl: 1.875rem;    /* 30px */
        --font-size-4xl: 2.25rem;     /* 36px */
        
        /* Line Heights */
        --line-height-tight: 1.2;
        --line-height-normal: 1.5;
        --line-height-relaxed: 1.75;
        
        /* Border Radius */
        --radius-sm: 6px;
        --radius-md: 10px;
        --radius-lg: 14px;
        --radius-xl: 20px;
        
        /* Transições */
        --transition-fast: 150ms cubic-bezier(0.4, 0, 0.2, 1);
        --transition-base: 250ms cubic-bezier(0.4, 0, 0.2, 1);
        --transition-slow: 350ms cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* Reset Global - Manter cores padrão do Streamlit Dark Theme */
    * {
        font-family: var(--font-family);
    }
    
    /* Container Principal */
    .main .block-container {
        padding-top: var(--spacing-2xl);
        padding-bottom: var(--spacing-2xl);
        max-width: 1400px;
    }
    
    /* Sidebar com Borda Accent */
    [data-testid="stSidebar"] {
        border-right: 1px solid var(--border-subtle);
    }
    
    /* Título Principal - Minimalista e Elegante */
    .gradient-title {
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: var(--font-size-2xl);
        font-weight: 600;
        text-align: center;
        margin: 0;
        letter-spacing: -0.03em;
        line-height: var(--line-height-tight);
    }
    
    /* Subtítulo - Mais Sutil */
    .subtitle {
        text-align: center;
        opacity: 0.5;
        font-size: var(--font-size-xs);
        margin: var(--spacing-xs) 0 0 0;
        font-weight: 400;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    
    /* Cards de Métricas - Minimalistas e Elegantes */
    .metric-card {
        background: transparent;
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: var(--spacing-md);
        text-align: center;
        transition: all var(--transition-base);
        position: relative;
        min-height: 80px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: var(--accent-gradient);
        opacity: 0;
        transition: opacity var(--transition-base);
    }
    
    .metric-card:hover {
        border-color: var(--border-accent);
        background: rgba(255, 107, 53, 0.02);
    }
    
    .metric-card:hover::before {
        opacity: 1;
    }
    
    .metric-value {
        font-size: var(--font-size-2xl);
        font-weight: 600;
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: var(--spacing-xs) 0;
        line-height: var(--line-height-tight);
        letter-spacing: -0.02em;
    }
    
    .metric-label {
        font-size: var(--font-size-xs);
        opacity: 0.5;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        line-height: var(--line-height-normal);
    }
    
    /* Info Boxes - Design Clean e Elegante */
    .info-box {
        background: transparent;
        border-left: 2px solid var(--accent-primary);
        border-radius: 0;
        padding: var(--spacing-md) var(--spacing-lg);
        margin: var(--spacing-md) 0;
        transition: all var(--transition-base);
    }
    
    .info-box:hover {
        background: rgba(255, 107, 53, 0.02);
        border-left-width: 3px;
    }
    
    .info-box h3, .info-box h4 {
        color: var(--accent-primary);
        margin-top: 0;
        font-weight: 600;
        font-size: var(--font-size-base);
        letter-spacing: -0.01em;
    }
    
    .info-box p {
        font-size: var(--font-size-sm);
        line-height: var(--line-height-relaxed);
        opacity: 0.8;
    }
    
    /* Divisores Sutis */
    hr {
        border: none;
        height: 1px;
        background: var(--border-subtle);
        margin: var(--spacing-xl) 0;
        opacity: 0.5;
    }
    
    /* Botões - Minimalistas e Elegantes */
    .stButton>button {
        background: var(--accent-gradient);
        color: white;
        border: none;
        border-radius: var(--radius-sm);
        padding: 0.5rem 1.25rem;
        font-weight: 500;
        font-size: var(--font-size-sm);
        transition: all var(--transition-base);
        box-shadow: none;
        letter-spacing: 0;
    }
    
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: var(--shadow-sm);
        opacity: 0.9;
    }
    
    .stButton>button:active {
        transform: translateY(0);
        box-shadow: none;
    }
    
    /* Botões Secundários */
    .stButton>button[kind="secondary"] {
        background: transparent;
        border: 1px solid var(--border-accent);
        color: var(--accent-primary);
    }
    
    .stButton>button[kind="secondary"]:hover {
        background: rgba(255, 107, 53, 0.05);
    }
    
    /* Tabs - Design Limpo */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background-color: transparent;
        padding: 0;
        border-radius: 0;
        margin-bottom: var(--spacing-lg);
        border-bottom: 1px solid var(--border-subtle);
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 0;
        font-weight: 500;
        font-size: var(--font-size-sm);
        padding: var(--spacing-sm) var(--spacing-md);
        transition: all var(--transition-fast);
        border-bottom: 2px solid transparent;
        margin-bottom: -1px;
    }
    
    .stTabs [aria-selected="true"] {
        background: transparent;
        color: var(--accent-primary);
        font-weight: 600;
        border-bottom-color: var(--accent-primary);
    }
    
    .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
        color: var(--accent-secondary);
        background-color: rgba(255, 107, 53, 0.02);
    }
    
    /* Dataframes com Bordas Accent */
    .dataframe {
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        overflow: hidden;
    }
    
    /* Expander com Efeitos Sutis */
    .streamlit-expanderHeader {
        background-color: rgba(255, 107, 53, 0.02);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        transition: all var(--transition-fast);
    }
    
    .streamlit-expanderHeader:hover {
        background-color: rgba(255, 107, 53, 0.04);
        border-color: var(--border-accent);
    }
    
    /* Upload area - Minimalista */
    [data-testid="stFileUploader"] {
        background-color: transparent;
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-sm);
        padding: var(--spacing-md);
        transition: all var(--transition-base);
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: var(--border-accent);
        background-color: rgba(255, 107, 53, 0.02);
    }
    
    [data-testid="stFileUploader"] section {
        border: none;
        border-radius: 0;
        background-color: transparent;
    }
    
    [data-testid="stFileUploader"] section > div {
        border: none !important;
    }
    
    [data-testid="stFileUploader"] section button {
        background: transparent;
        color: var(--accent-primary);
        border: 1px solid var(--border-accent);
        border-radius: var(--radius-sm);
        font-weight: 500;
        font-size: var(--font-size-sm);
        padding: 0.4rem 1rem;
        transition: all var(--transition-base);
    }
    
    [data-testid="stFileUploader"] section button:hover {
        background: rgba(255, 107, 53, 0.08);
        border-color: var(--accent-primary);
    }
    
    [data-testid="stFileUploader"] small {
        font-size: var(--font-size-xs);
        opacity: 0.6;
    }
    
    /* Metrics Nativas do Streamlit */
    [data-testid="stMetricValue"] {
        font-size: var(--font-size-2xl);
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(var(--glow-sm));
    }
    
    /* Imagens com Bordas Padronizadas */
    img {
        border-radius: var(--radius-md);
        border: 1px solid var(--border-subtle);
        transition: all var(--transition-base);
    }
    
    img:hover {
        border-color: var(--border-accent);
        box-shadow: var(--shadow-md);
    }
    
    /* Headers - Hierarquia Tipográfica Refinada */
    h1 {
        font-size: var(--font-size-3xl);
        font-weight: 600;
        letter-spacing: -0.03em;
        line-height: var(--line-height-tight);
        margin-bottom: var(--spacing-md);
    }
    
    h2 {
        font-size: var(--font-size-xl);
        font-weight: 600;
        letter-spacing: -0.02em;
        line-height: var(--line-height-tight);
        margin-top: var(--spacing-xl);
        margin-bottom: var(--spacing-sm);
    }
    
    h3 {
        font-size: var(--font-size-lg);
        font-weight: 600;
        letter-spacing: -0.01em;
        line-height: var(--line-height-tight);
        margin-top: var(--spacing-lg);
        margin-bottom: var(--spacing-sm);
        padding-bottom: var(--spacing-xs);
        border-bottom: 1px solid var(--border-subtle);
    }
    
    h4 {
        font-size: var(--font-size-base);
        font-weight: 600;
        letter-spacing: 0;
        line-height: var(--line-height-normal);
        margin-top: var(--spacing-md);
        margin-bottom: var(--spacing-xs);
        opacity: 0.9;
    }
    
    /* Parágrafos */
    p {
        font-size: var(--font-size-sm);
        line-height: var(--line-height-relaxed);
        margin-bottom: var(--spacing-md);
    }
    
    /* Inputs e Selects Profissionais */
    .stTextInput>div>div>input,
    .stSelectbox>div>div>select,
    .stNumberInput>div>div>input {
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-sm);
        transition: all var(--transition-fast);
    }
    
    .stTextInput>div>div>input:focus,
    .stSelectbox>div>div>select:focus,
    .stNumberInput>div>div>input:focus {
        border-color: var(--accent-primary);
        box-shadow: 0 0 0 3px rgba(255, 107, 53, 0.1);
    }
    
    /* Scrollbar Estilizada */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 107, 53, 0.2);
        border-radius: var(--radius-sm);
        transition: background var(--transition-fast);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-primary);
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background: var(--accent-gradient);
    }
    
    /* Spinners */
    .stSpinner > div {
        border-top-color: var(--accent-primary);
    }
    
    /* Alert Boxes */
    .stAlert {
        border-radius: var(--radius-md);
        border-left-width: 4px;
    }
    
    /* Status Badges - Minimalistas */
    .status-badge {
        display: inline-block;
        padding: 0.125rem 0.5rem;
        border-radius: 3px;
        font-size: var(--font-size-xs);
        font-weight: 500;
        letter-spacing: 0.03em;
        transition: all var(--transition-fast);
        text-transform: uppercase;
    }
    
    .status-normal {
        background: rgba(107, 255, 107, 0.08);
        color: #6bff6b;
        border: none;
    }
    
    .status-anomaly {
        background: rgba(255, 107, 53, 0.08);
        color: var(--accent-primary);
        border: none;
    }
    
    /* Radio Buttons - Espaçamento Reduzido */
    .stRadio > div {
        gap: var(--spacing-sm);
    }
    
    .stRadio label {
        font-size: var(--font-size-sm);
    }
    
    /* Slider - Accent Color e Refinado */
    .stSlider > div > div > div {
        background: var(--accent-gradient);
    }
    
    .stSlider > div > div > div > div {
        background-color: var(--accent-primary);
    }
    
    /* Slider labels */
    .stSlider > label {
        font-size: var(--font-size-sm);
        font-weight: 500;
    }
    
    /* Expander - Mais Compacto e Elegante */
    .streamlit-expanderHeader {
        font-size: var(--font-size-sm);
        padding: var(--spacing-sm) var(--spacing-md);
        font-weight: 500;
    }
    
    .streamlit-expanderHeader:hover {
        color: var(--accent-primary);
    }
    
    /* Selectbox e outras inputs - Fontes Menores */
    .stSelectbox label, .stTextInput label, .stNumberInput label {
        font-size: var(--font-size-sm);
        font-weight: 500;
    }
    
    /* Info/Success/Warning/Error boxes - Mais Elegantes */
    .stInfo, .stSuccess, .stWarning, .stError {
        font-size: var(--font-size-sm);
        padding: var(--spacing-sm) var(--spacing-md);
        border-radius: var(--radius-sm);
    }
    
    .stSuccess {
        background-color: rgba(107, 255, 107, 0.08);
        border-left: 3px solid #6bff6b;
    }
    
    /* Caption - Mais sutil */
    .caption, [data-testid="stCaptionContainer"] {
        font-size: var(--font-size-xs);
        opacity: 0.6;
        margin-top: 0.25rem;
    }
    
    /* Download button */
    .stDownloadButton > button {
        font-size: var(--font-size-sm);
        padding: 0.5rem 1rem;
    }
    
    /* Dataframe - Fonte Menor */
    .dataframe {
        font-size: var(--font-size-xs);
    }
    
    /* Código - Fonte Menor */
    code {
        font-size: var(--font-size-xs);
        padding: 0.125rem 0.375rem;
        border-radius: 3px;
    }
    
    /* Ajustes Globais de Espaçamento */
    .element-container {
        margin-bottom: var(--spacing-sm);
    }
    
    /* Markdown - Espaçamento Otimizado */
    .stMarkdown {
        font-size: var(--font-size-sm);
    }
    
    .stMarkdown ul, .stMarkdown ol {
        font-size: var(--font-size-sm);
        line-height: var(--line-height-relaxed);
    }
    
    /* Container - Mais elegante */
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
        gap: var(--spacing-md);
    }
    
    /* Remover ícones desnecessários de success/info */
    .stSuccess > div > div:first-child,
    .stInfo > div > div:first-child {
        display: none;
    }
    
    /* Estilo para seções de título */
    .stMarkdown h4 {
        color: var(--accent-primary);
        font-weight: 600;
        font-size: var(--font-size-base);
        margin-bottom: var(--spacing-sm);
    }
    
    /* Ajustes finos nos inputs */
    input[type="number"]:focus,
    select:focus,
    textarea:focus {
        outline: none;
        border-color: var(--accent-primary) !important;
    }
    </style>
""", unsafe_allow_html=True)

# Funções auxiliares
# URLs dos arquivos no GitHub
GITHUB_REPO = "https://raw.githubusercontent.com/sidnei-almeida/secom_failure_prediction/main"
METADATA_URL = f"{GITHUB_REPO}/training/secom_autoencoder_metadata.json"
DATASET_URL = f"{GITHUB_REPO}/data/secom_cleaned_dataset.csv"
MODEL_URL = f"{GITHUB_REPO}/models/secom_autoencoder_model.keras"

@st.cache_data
def load_metadata():
    """Carrega os metadados do treinamento do GitHub"""
    try:
        import requests
        response = requests.get(METADATA_URL)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Erro ao carregar metadados do GitHub: {e}")
        return None

@st.cache_data
def load_dataset():
    """Carrega o dataset limpo do GitHub"""
    try:
        return pd.read_csv(DATASET_URL)
    except Exception as e:
        st.error(f"Erro ao carregar dataset do GitHub: {e}")
        return None

@st.cache_resource
def load_model():
    """Carrega o modelo autoencoder do GitHub"""
    try:
        import requests
        import tempfile
        
        # Baixar modelo para arquivo temporário
        response = requests.get(MODEL_URL)
        response.raise_for_status()
        
        # Salvar temporariamente e carregar
        with tempfile.NamedTemporaryFile(delete=False, suffix='.keras') as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name
        
        model = tf.keras.models.load_model(tmp_path)
        
        # Remover arquivo temporário
        import os
        os.unlink(tmp_path)
        
        return model
    except Exception as e:
        st.error(f"Erro ao carregar modelo do GitHub: {e}")
        return None

def show_header():
    """Exibe o header principal"""
    st.markdown("""
        <div style='text-align: center; padding: 1.5rem 0 1rem 0;'>
            <h1 class="gradient-title">SECOM Failure Prediction</h1>
            <p class="subtitle">Sistema Avançado de Detecção de Anomalias em Manufatura de Semicondutores</p>
        </div>
        <div style='margin: 2rem 0; height: 1px; background: var(--border-color);'></div>
    """, unsafe_allow_html=True)

# Sidebar com menu
with st.sidebar:
    # Logo/Header - Minimalista
    st.markdown("""
        <div style='text-align: center; padding: 0.75rem 0 1.25rem 0;'>
            <h2 style='margin: 0; background: var(--accent-gradient); -webkit-background-clip: text; 
                       -webkit-text-fill-color: transparent; font-size: 1.125rem; font-weight: 600; 
                       letter-spacing: -0.02em;'>
                SECOM AI
            </h2>
            <p style='margin: 0.25rem 0 0 0; font-size: 0.625rem; opacity: 0.5; 
                      letter-spacing: 0.08em; text-transform: uppercase;'>
                Anomaly Detection
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Menu de navegação - Elegante e Compacto
    selected = option_menu(
        menu_title=None,
        options=["Home", "Análise de Dados", "Modelo", "Treinamento", "Teste"],
        icons=["house-fill", "graph-up-arrow", "cpu-fill", "layers-fill", "play-circle-fill"],
        menu_icon=None,
        default_index=0,
        styles={
            "container": {"padding": "0", "background-color": "transparent"},
            "icon": {"color": "#ff6b35", "font-size": "0.9375rem"},
            "nav-link": {
                "font-size": "0.8125rem",
                "text-align": "left",
                "margin": "0 0 0.25rem 0",
                "padding": "0.5rem 0.75rem",
                "border-radius": "6px",
                "color": "rgba(255, 255, 255, 0.6)",
                "font-weight": "500",
                "background-color": "transparent",
                "--hover-color": "rgba(255, 107, 53, 0.08)",
            },
            "nav-link-selected": {
                "background": "rgba(255, 107, 53, 0.08)",
                "color": "#ff6b35",
                "border-left": "2px solid #ff6b35",
                "font-weight": "600",
            },
        }
    )
    
    st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
    
    # Status do sistema - Compacto e Elegante
    metadata = load_metadata()
    if metadata:
        st.markdown(f"""
            <div style='background: transparent; border-radius: 6px; padding: 0.75rem; 
                        margin-bottom: 0.75rem; border: 1px solid var(--border-subtle);'>
                <div style='font-size: 0.6875rem; font-weight: 600; margin-bottom: 0.5rem; 
                           opacity: 0.9; letter-spacing: 0.05em; text-transform: uppercase;'>
                    Status do Sistema
                </div>
                <div style='font-size: 0.6875rem; line-height: 1.6; opacity: 0.7;'>
                    <div style='margin-bottom: 0.25rem;'>• Modelo: {metadata['model_type']}</div>
                    <div style='margin-bottom: 0.25rem;'>• Threshold: {metadata['final_anomaly_threshold']}</div>
                    <div>• Accuracy: {metadata['final_performance_on_test_set']['accuracy']:.1%}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Thresholds - Minimalista
    st.markdown("""
        <div style='background: transparent; border-radius: 6px; padding: 0.75rem; 
                    border: 1px solid var(--border-subtle);'>
            <div style='font-size: 0.6875rem; font-weight: 600; margin-bottom: 0.5rem; 
                       opacity: 0.9; letter-spacing: 0.05em; text-transform: uppercase;'>
                Thresholds
            </div>
            <div style='font-size: 0.6875rem; line-height: 1.6; opacity: 0.7;'>
                <div style='display: flex; justify-content: space-between; margin-bottom: 0.25rem;'>
                    <span>Balanced</span>
                    <span style='color: #ff6b35; font-weight: 600;'>0.45</span>
                </div>
                <div style='display: flex; justify-content: space-between;'>
                    <span>Conservative</span>
                    <span style='color: #f7931e; font-weight: 600;'>0.50</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# PÁGINA: HOME
if selected == "Home":
    show_header()
    
    metadata = load_metadata()
    df = load_dataset()
    
    if metadata and df is not None:
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        
        perf = metadata['final_performance_on_test_set']
        
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Recall (Falhas)</div>
                    <div class="metric-value">{perf['recall_for_anomaly']:.1%}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Precision (Falhas)</div>
                    <div class="metric-value">{perf['precision_for_anomaly']:.1%}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">F1-Score</div>
                    <div class="metric-value">{perf['f1_score_for_anomaly']:.2f}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Accuracy Geral</div>
                    <div class="metric-value">{perf['accuracy']:.1%}</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Conteúdo principal em 2 colunas
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("### Sobre o Projeto")
            st.markdown(f"""
            <div class="info-box">
            <p>
            O <b>SECOM Failure Prediction</b> é um sistema avançado de detecção de anomalias 
            desenvolvido para identificar falhas em processos de manufatura de semicondutores. 
            Utilizando um <b>Autoencoder Neural Network</b>, o sistema aprende padrões de 
            operação normal e detecta desvios que podem indicar potenciais falhas.
            </p>
            <p>
            O dataset SECOM contém <b>{len(df)} registros</b> de sensores, com 
            <b>{len(df.columns)-2} features</b> após limpeza, representando medições 
            de diversos pontos do processo de fabricação.
            </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### Tecnologias Utilizadas")
            tech_cols = st.columns(3)
            with tech_cols[0]:
                st.markdown("**🧠 TensorFlow/Keras**\n\nDeep Learning Framework")
            with tech_cols[1]:
                st.markdown("**📊 Autoencoder**\n\nNeural Network Architecture")
            with tech_cols[2]:
                st.markdown("**⚡ Anomaly Detection**\n\nUnsupervised Learning")
        
        with col2:
            st.markdown("### Distribuição de Classes")
            
            # Calcular distribuição
            class_counts = df['Pass/Fail'].value_counts()
            total = len(df)
            normal_count = class_counts.get(-1, 0)
            anomaly_count = class_counts.get(1, 0)
            
            fig = go.Figure(data=[go.Pie(
                labels=['Normal', 'Falha'],
                values=[normal_count, anomaly_count],
                hole=0.6,
                marker=dict(colors=[PLOT_COLORS['success'], PLOT_COLORS['primary']]),
                textinfo='label+percent',
                textfont=dict(size=14, color=PLOT_COLORS['text']),
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            )])
            
            fig.update_layout(
                showlegend=False,
                paper_bgcolor=PLOT_COLORS['bg'],
                plot_bgcolor=PLOT_COLORS['bg'],
                font=dict(family=PLOT_COLORS['font'], color=PLOT_COLORS['text'], size=PLOT_COLORS['font_size']),
                height=280,
                margin=dict(t=10, b=10, l=10, r=10),
                annotations=[dict(
                    text=f'<b>{total}</b><br><span style="font-size: 10px; opacity: 0.7;">Total</span>',
                    x=0.5, y=0.5,
                    font_size=18,
                    showarrow=False,
                    font_color=PLOT_COLORS['text']
                )]
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"""
                <div style='text-align: center; margin-top: 1rem;'>
                    <div style='display: inline-block; margin: 0 1rem;'>
                        <span class='status-badge status-normal'>Normal: {normal_count} ({normal_count/total:.1%})</span>
                    </div>
                    <div style='display: inline-block; margin: 0 1rem;'>
                        <span class='status-badge status-anomaly'>Falha: {anomaly_count} ({anomaly_count/total:.1%})</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Insights principais
        st.markdown("### Principais Insights")
        
        insight_cols = st.columns(3)
        
        with insight_cols[0]:
            st.markdown("""
            <div class="info-box">
                <h4>Desbalanceamento Extremo</h4>
                <p>O dataset apresenta desbalanceamento severo (~93% normal vs ~7% falhas), 
                tornando a abordagem de detecção de anomalias ideal para este cenário.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with insight_cols[1]:
            st.markdown("""
            <div class="info-box">
                <h4>Alta Dimensionalidade</h4>
                <p>Com 558 features após limpeza, o autoencoder comprime os dados para 
                32 dimensões, capturando apenas os padrões essenciais.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with insight_cols[2]:
            st.markdown("""
            <div class="info-box">
                <h4>Erro de Reconstrução</h4>
                <p>O modelo detecta anomalias através do erro de reconstrução: produtos 
                normais são reconstruídos com baixo erro, falhas com alto erro.</p>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        st.error("Não foi possível carregar os dados do projeto. Verifique os arquivos.")

# PÁGINA: ANÁLISE DE DADOS
elif selected == "Análise de Dados":
    show_header()
    
    df = load_dataset()
    
    if df is not None:
        st.markdown("### Exploração do Dataset SECOM")
        
        # Informações gerais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total de Registros</div>
                    <div class="metric-value">{len(df)}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Features</div>
                    <div class="metric-value">{len(df.columns)-2}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            normal_count = (df['Pass/Fail'] == -1).sum()
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Amostras Normais</div>
                    <div class="metric-value">{normal_count}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            anomaly_count = (df['Pass/Fail'] == 1).sum()
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Falhas Detectadas</div>
                    <div class="metric-value">{anomaly_count}</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Tabs com análises
        tab1, tab2, tab3 = st.tabs(["Estatísticas Descritivas", "Distribuições", "Correlações"])
        
        with tab1:
            st.markdown("#### Estatísticas das Features")
            
            # Remover colunas Time e Pass/Fail
            features_df = df.drop(columns=['Time', 'Pass/Fail'])
            
            # Estatísticas descritivas
            stats = features_df.describe().T
            stats['missing'] = features_df.isnull().sum()
            stats['missing_pct'] = (stats['missing'] / len(features_df)) * 100
            
            st.dataframe(
                stats[['mean', 'std', 'min', '50%', 'max', 'missing', 'missing_pct']].head(20),
                use_container_width=True
            )
            
            st.info(f"Mostrando as primeiras 20 features de um total de {len(features_df.columns)} features.")
        
        with tab2:
            st.markdown("#### Distribuição de Valores nas Features")
            
            # Selecionar algumas features aleatórias para visualizar
            sample_features = features_df.columns[:6]
            
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=sample_features
            )
            
            for idx, feature in enumerate(sample_features):
                row = idx // 3 + 1
                col = idx % 3 + 1
                
                fig.add_trace(
                    go.Histogram(
                        x=features_df[feature],
                        name=feature,
                        marker=dict(color=PLOT_COLORS['primary'], opacity=0.7),
                        showlegend=False
                    ),
                    row=row, col=col
                )
            
            fig.update_layout(
                height=600,
                paper_bgcolor=PLOT_COLORS['bg'],
                plot_bgcolor=PLOT_COLORS['bg'],
                font=dict(family=PLOT_COLORS['font'], size=PLOT_COLORS['font_size'], color=PLOT_COLORS['text']),
                showlegend=False
            )
            
            fig.update_xaxes(showgrid=True, gridcolor=PLOT_COLORS['grid'])
            fig.update_yaxes(showgrid=True, gridcolor=PLOT_COLORS['grid'])
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("#### Análise de Correlação")
            
            # Calcular matriz de correlação para um subset de features
            sample_features = features_df.columns[:15]
            corr_matrix = features_df[sample_features].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale=[[0, '#0e1117'], [0.5, PLOT_COLORS['secondary']], [1, PLOT_COLORS['primary']]],
                colorbar=dict(title="Correlação")
            ))
            
            fig.update_layout(
                height=600,
                paper_bgcolor=PLOT_COLORS['bg'],
                plot_bgcolor=PLOT_COLORS['bg'],
                font=dict(family=PLOT_COLORS['font'], size=10, color=PLOT_COLORS['text'])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("Mostrando correlação entre as primeiras 15 features para melhor visualização.")
    
    else:
        st.error("Dataset não encontrado. Verifique o arquivo 'data/secom_cleaned_dataset.csv'.")

# PÁGINA: MODELO
elif selected == "Modelo":
    show_header()
    
    st.markdown("### Arquitetura do Autoencoder")
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>O que é um Autoencoder?</h4>
            <p style='line-height: 1.8;'>
            Um <b>Autoencoder</b> é um tipo de rede neural que aprende a comprimir dados 
            em uma representação latente menor e depois reconstruí-los. É composto por duas partes:
            </p>
            <ul style='line-height: 2;'>
                <li><b>Encoder:</b> Comprime os dados de 558 features para 32 dimensões</li>
                <li><b>Decoder:</b> Reconstrói os dados originais a partir da representação comprimida</li>
            </ul>
            <p style='line-height: 1.8; margin-top: 1rem;'>
            Para detecção de anomalias, o modelo é treinado <b>apenas com dados normais</b>. 
            Quando encontra uma anomalia (falha), o erro de reconstrução é significativamente maior.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Métricas da arquitetura
        st.markdown("#### Especificações Técnicas")
        st.markdown("""
        <div class="info-box">
            <ul style='line-height: 2;'>
                <li><b>Input:</b> 558 features (sensores SECOM)</li>
                <li><b>Encoder:</b> 558 → 128 → 64 → 32</li>
                <li><b>Latent Space:</b> 32 dimensões (bottleneck)</li>
                <li><b>Decoder:</b> 32 → 64 → 128 → 558</li>
                <li><b>Ativação:</b> ReLU (camadas ocultas)</li>
                <li><b>Loss Function:</b> MAE (Mean Absolute Error)</li>
                <li><b>Optimizer:</b> Adam</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### Visualização da Arquitetura")
        
        # Criar visualização da arquitetura com Plotly
        fig = go.Figure()
        
        # Definir posições das camadas
        layers = [
            {"name": "Input", "size": 558, "y": 0, "color": PLOT_COLORS['primary']},
            {"name": "Dense 128", "size": 128, "y": 1, "color": PLOT_COLORS['secondary']},
            {"name": "Dense 64", "size": 64, "y": 2, "color": PLOT_COLORS['tertiary']},
            {"name": "Latent 32", "size": 32, "y": 3, "color": PLOT_COLORS['primary']},
            {"name": "Dense 64", "size": 64, "y": 4, "color": PLOT_COLORS['tertiary']},
            {"name": "Dense 128", "size": 128, "y": 5, "color": PLOT_COLORS['secondary']},
            {"name": "Output", "size": 558, "y": 6, "color": PLOT_COLORS['primary']},
        ]
        
        # Adicionar nós
        for layer in layers:
            # Normalizar tamanho para visualização
            node_size = 20 + (layer["size"] / 558) * 40
            
            fig.add_trace(go.Scatter(
                x=[0],
                y=[layer["y"]],
                mode='markers+text',
                marker=dict(
                    size=node_size,
                    color=layer["color"],
                    line=dict(color='white', width=2)
                ),
                text=f"{layer['name']}<br>({layer['size']} neurons)",
                textposition="middle right",
                textfont=dict(size=12, color='white'),
                name=layer["name"],
                showlegend=False,
                hovertemplate=f"<b>{layer['name']}</b><br>{layer['size']} neurons<extra></extra>"
            ))
        
        # Adicionar linhas conectando as camadas
        for i in range(len(layers) - 1):
            fig.add_trace(go.Scatter(
                x=[0, 0],
                y=[layers[i]["y"], layers[i+1]["y"]],
                mode='lines',
                line=dict(color='rgba(255, 107, 53, 0.3)', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            height=500,
            paper_bgcolor=PLOT_COLORS['bg'],
            plot_bgcolor=PLOT_COLORS['bg'],
            font=dict(family=PLOT_COLORS['font'], color=PLOT_COLORS['text']),
            xaxis=dict(
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                range=[-0.5, 1]
            ),
            yaxis=dict(
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                range=[-0.5, 6.5]
            ),
            margin=dict(l=20, r=200, t=20, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Explicação do bottleneck
        st.markdown("""
        <div class="info-box">
            <h4>🔍 Bottleneck (Gargalo)</h4>
            <p style='line-height: 1.8;'>
            A camada de 32 neurônios é o <b>bottleneck</b> - o ponto mais estreito da rede. 
            Ela força o modelo a aprender apenas as características mais importantes dos dados, 
            descartando ruído e informações redundantes. Esta compressão é essencial para 
            a detecção de anomalias.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Explicação do processo de detecção
    st.markdown("### Como Funciona a Detecção de Anomalias")
    
    process_cols = st.columns(4)
    
    with process_cols[0]:
        st.markdown("""
        <div class="info-box" style="text-align: center;">
            <h3 style="font-size: 2.5em; margin: 0.5rem 0;">1️⃣</h3>
            <h4>Treinamento</h4>
            <p>Modelo aprende com dados normais apenas (1170 amostras)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with process_cols[1]:
        st.markdown("""
        <div class="info-box" style="text-align: center;">
            <h3 style="font-size: 2.5em; margin: 0.5rem 0;">2️⃣</h3>
            <h4>Reconstrução</h4>
            <p>Dados de teste passam pelo encoder e decoder</p>
        </div>
        """, unsafe_allow_html=True)
    
    with process_cols[2]:
        st.markdown("""
        <div class="info-box" style="text-align: center;">
            <h3 style="font-size: 2.5em; margin: 0.5rem 0;">3️⃣</h3>
            <h4>Erro de Reconstrução</h4>
            <p>Calcula MAE entre input e output reconstruído</p>
        </div>
        """, unsafe_allow_html=True)
    
    with process_cols[3]:
        st.markdown("""
        <div class="info-box" style="text-align: center;">
            <h3 style="font-size: 2.5em; margin: 0.5rem 0;">4️⃣</h3>
            <h4>Classificação</h4>
            <p>Se erro > threshold: anomalia; caso contrário: normal</p>
        </div>
        """, unsafe_allow_html=True)

# PÁGINA: TREINAMENTO
elif selected == "Treinamento":
    show_header()
    
    metadata = load_metadata()
    
    if metadata:
        st.markdown("### Histórico de Treinamento do Autoencoder")
        
        # Métricas finais
        col1, col2, col3, col4 = st.columns(4)
        
        history = metadata['training_history']
        final_loss = history['loss'][-1]
        final_val_loss = history['val_loss'][-1]
        total_epochs = len(history['loss'])
        threshold = metadata['final_anomaly_threshold']
        
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Épocas Treinadas</div>
                    <div class="metric-value">{total_epochs}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Loss Final (Train)</div>
                    <div class="metric-value">{final_loss:.4f}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Loss Final (Val)</div>
                    <div class="metric-value">{final_val_loss:.4f}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Threshold</div>
                    <div class="metric-value">{threshold}</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Gráfico de evolução do loss
        st.markdown("#### Evolução da Loss Durante o Treinamento")
        
        epochs = list(range(1, total_epochs + 1))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=epochs,
            y=history['loss'],
            name='Training Loss',
            line=dict(color=PLOT_COLORS['primary'], width=3),
            mode='lines'
        ))
        
        fig.add_trace(go.Scatter(
            x=epochs,
            y=history['val_loss'],
            name='Validation Loss',
            line=dict(color=PLOT_COLORS['tertiary'], width=3),
            mode='lines'
        ))
        
        fig.update_layout(
            xaxis_title="Época",
            yaxis_title="Loss (MAE)",
            hovermode='x unified',
            height=500,
            paper_bgcolor=PLOT_COLORS['bg'],
            plot_bgcolor=PLOT_COLORS['bg'],
                font=dict(family=PLOT_COLORS['font'], size=PLOT_COLORS['font_size'], color=PLOT_COLORS['text']),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(255, 107, 53, 0.05)',
                bordercolor=PLOT_COLORS['primary'],
                borderwidth=1
            )
        )
        
        fig.update_xaxes(showgrid=True, gridcolor=PLOT_COLORS['grid'])
        fig.update_yaxes(showgrid=True, gridcolor=PLOT_COLORS['grid'])
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Performance final
        st.markdown("#### Performance Final no Conjunto de Teste")
        
        perf = metadata['final_performance_on_test_set']
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Métricas detalhadas
            st.markdown("""
            <div class="info-box">
                <h4>Métricas de Classificação</h4>
            </div>
            """, unsafe_allow_html=True)
            
            metrics_data = {
                'Métrica': ['Precision (Anomalia)', 'Recall (Anomalia)', 'F1-Score', 'Accuracy'],
                'Valor': [
                    f"{perf['precision_for_anomaly']:.2%}",
                    f"{perf['recall_for_anomaly']:.2%}",
                    f"{perf['f1_score_for_anomaly']:.3f}",
                    f"{perf['accuracy']:.2%}"
                ]
            }
            
            st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)
        
        with col2:
            # Gráfico de barras das métricas
            fig = go.Figure(data=[
                go.Bar(
                    x=['Precision', 'Recall', 'F1-Score', 'Accuracy'],
                    y=[
                        perf['precision_for_anomaly'],
                        perf['recall_for_anomaly'],
                        perf['f1_score_for_anomaly'],
                        perf['accuracy']
                    ],
                    marker=dict(
                        color=[PLOT_COLORS['primary'], PLOT_COLORS['secondary'], PLOT_COLORS['tertiary'], '#ff8c42'],
                        line=dict(color=PLOT_COLORS['text'], width=1)
                    ),
                    text=[
                        f"{perf['precision_for_anomaly']:.1%}",
                        f"{perf['recall_for_anomaly']:.1%}",
                        f"{perf['f1_score_for_anomaly']:.2f}",
                        f"{perf['accuracy']:.1%}"
                    ],
                    textposition='outside',
                    textfont=dict(size=14, color=PLOT_COLORS['text'])
                )
            ])
            
            fig.update_layout(
                height=300,
                paper_bgcolor=PLOT_COLORS['bg'],
                plot_bgcolor=PLOT_COLORS['bg'],
                font=dict(family=PLOT_COLORS['font'], color=PLOT_COLORS['text']),
                showlegend=False,
                yaxis=dict(title="Valor", range=[0, 1]),
                xaxis=dict(title="")
            )
            
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=True, gridcolor=PLOT_COLORS['grid'])
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Informações sobre o treinamento
        st.markdown("### Configuração do Treinamento")
        
        config_cols = st.columns(3)
        
        with config_cols[0]:
            st.markdown("""
            <div class="info-box">
                <h4>Dados de Treino</h4>
                <ul style='line-height: 2;'>
                    <li><b>Amostras:</b> 1170</li>
                    <li><b>Tipo:</b> Apenas normais</li>
                    <li><b>Validação:</b> 20% split</li>
                    <li><b>Escalonamento:</b> StandardScaler</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with config_cols[1]:
            st.markdown("""
            <div class="info-box">
                <h4>Hiperparâmetros</h4>
                <ul style='line-height: 2;'>
                    <li><b>Batch Size:</b> 32</li>
                    <li><b>Epochs:</b> 150 (max)</li>
                    <li><b>Optimizer:</b> Adam</li>
                    <li><b>Early Stopping:</b> Patience 10</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with config_cols[2]:
            st.markdown("""
            <div class="info-box">
                <h4>Resultados</h4>
                <ul style='line-height: 2;'>
                    <li><b>Stopped at:</b> Época 53</li>
                    <li><b>Best Loss:</b> 0.4055</li>
                    <li><b>Overfitting:</b> Baixo</li>
                    <li><b>Convergência:</b> Estável</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        st.error("Metadados de treinamento não encontrados.")

# PÁGINA: TESTE
elif selected == "Teste":
    show_header()
    
    st.markdown("### Teste do Modelo")
    
    model = load_model()
    metadata = load_metadata()
    
    if model and metadata:
        # Seleção de threshold - Layout Compacto
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            threshold_option = st.radio(
                "**Threshold de Detecção**",
                options=["Balanced (0.45)", "Conservative (0.50)"],
                horizontal=True,
                help="Balanced: equilíbrio precision/recall. Conservative: menos falsos positivos."
            )
        
        threshold = 0.45 if "Balanced" in threshold_option else 0.50
        
        with col2:
            st.markdown(f"""
                <div class="metric-card" style="margin-top: 0;">
                    <div class="metric-label">Threshold</div>
                    <div class="metric-value">{threshold}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div class="metric-card" style="margin-top: 0;">
                    <div class="metric-label">Tipo</div>
                    <div class="metric-value" style="font-size: 1rem;">{'BAL' if threshold == 0.45 else 'CON'}</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
        
        # Tabs para Upload ou Dados de Amostra
        test_tab1, test_tab2 = st.tabs(["Dados de Amostra", "Upload de CSV"])
        
        with test_tab1:
            # Carregar dataset
            df_demo = load_dataset()
            
            if df_demo is not None:
                # Controles lado a lado
                ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 2, 1])
                
                with ctrl_col1:
                    sample_size = st.slider(
                        "Tamanho da amostra",
                        min_value=10,
                        max_value=min(100, len(df_demo)),
                        value=50,
                        step=10
                    )
                
                with ctrl_col2:
                    sample_type = st.selectbox(
                        "Tipo de dados",
                        options=["Misto", "Apenas Normais", "Apenas Falhas"]
                    )
                
                with ctrl_col3:
                    st.markdown("<div style='margin-top: 1.75rem;'></div>", unsafe_allow_html=True)
                    analyze_button = st.button("Analisar", type="primary", use_container_width=True, key="analyze_sample")
                
                # Preparar amostra
                if sample_type == "Apenas Normais":
                    df_sample = df_demo[df_demo['Pass/Fail'] == -1].sample(n=min(sample_size, (df_demo['Pass/Fail'] == -1).sum()), random_state=42)
                elif sample_type == "Apenas Falhas":
                    df_sample = df_demo[df_demo['Pass/Fail'] == 1].sample(n=min(sample_size, (df_demo['Pass/Fail'] == 1).sum()), random_state=42)
                else:  # Misto
                    df_sample = df_demo.sample(n=min(sample_size, len(df_demo)), random_state=42)
                
                # Info da amostra
                st.caption(f"{len(df_sample)} registros • {sample_type}")
                
                # Preview compacto
                with st.expander("Visualizar dados da amostra", expanded=False):
                    st.dataframe(df_sample.head(10), use_container_width=True, height=250)
                
                st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
                
                # Processar quando botão for clicado
                if analyze_button:
                    with st.spinner("Processando amostra através do autoencoder..."):
                        # Preparar dados
                        y_true = df_sample['Pass/Fail'].replace({-1: 0, 1: 1})
                        X_sample = df_sample.drop(columns=['Time', 'Pass/Fail'])
                        
                        # Normalizar
                        scaler = StandardScaler()
                        X_sample_scaled = scaler.fit_transform(X_sample)
                        
                        # Predições
                        reconstructions = model.predict(X_sample_scaled, verbose=0)
                        reconstruction_errors = tf.keras.losses.mae(reconstructions, X_sample_scaled).numpy()
                        predictions = (reconstruction_errors > threshold).astype(int)
                    
                    st.markdown("<div style='margin: 2rem 0 1.5rem 0;'></div>", unsafe_allow_html=True)
                    
                    # Container de Resultados - Premium
                    with st.container():
                        st.markdown("#### Resultados")
                        
                        # Estatísticas
                        normal_count = (predictions == 0).sum()
                        anomaly_count = (predictions == 1).sum()
                        anomaly_rate = (anomaly_count / len(predictions)) * 100
                        
                        # Cards em grid harmonioso
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-label">Amostras</div>
                                    <div class="metric-value">{len(predictions)}</div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-label">Normal</div>
                                    <div class="metric-value">{normal_count}</div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-label">Anomalias</div>
                                    <div class="metric-value">{anomaly_count}</div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with col4:
                            st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-label">Taxa</div>
                                    <div class="metric-value">{anomaly_rate:.1f}%</div>
                                </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
                    
                    # Container de Visualizações
                    with st.container():
                        st.markdown("#### Análise Visual")
                        col1, col2 = st.columns([1.3, 1])
                        
                        with col1:
                            fig = go.Figure()
                            
                            fig.add_trace(go.Histogram(
                                x=reconstruction_errors[predictions == 0],
                                name='Normal',
                                marker=dict(color=PLOT_COLORS['success'], opacity=0.7),
                                nbinsx=30
                            ))
                            
                            fig.add_trace(go.Histogram(
                                x=reconstruction_errors[predictions == 1],
                                name='Anomalia',
                                marker=dict(color=PLOT_COLORS['primary'], opacity=0.7),
                                nbinsx=30
                            ))
                            
                            fig.add_vline(
                                x=threshold,
                                line_dash="dash",
                                line_color="#ffb627",
                                line_width=3,
                                annotation_text=f"Threshold ({threshold})",
                                annotation_position="top"
                            )
                            
                            fig.update_layout(
                                barmode='overlay',
                                height=350,
                                paper_bgcolor=PLOT_COLORS['bg'],
                                plot_bgcolor=PLOT_COLORS['bg'],
                                font=dict(family=PLOT_COLORS['font'], size=PLOT_COLORS['font_size'], color=PLOT_COLORS['text']),
                                xaxis_title="Erro de Reconstrução",
                                yaxis_title="Contagem",
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                margin=dict(t=30, b=50, l=50, r=20)
                            )
                            
                            fig.update_xaxes(showgrid=True, gridcolor=PLOT_COLORS['grid'])
                            fig.update_yaxes(showgrid=True, gridcolor=PLOT_COLORS['grid'])
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            fig = go.Figure(data=[go.Pie(
                                labels=['Normal', 'Anomalia'],
                                values=[normal_count, anomaly_count],
                                hole=0.5,
                                marker=dict(colors=[PLOT_COLORS['success'], PLOT_COLORS['primary']]),
                                textinfo='label+percent+value',
                                textfont=dict(size=14, color=PLOT_COLORS['text'])
                            )])
                            
                            fig.update_layout(
                                height=350,
                                paper_bgcolor=PLOT_COLORS['bg'],
                                plot_bgcolor=PLOT_COLORS['bg'],
                                font=dict(family=PLOT_COLORS['font'], size=PLOT_COLORS['font_size'], color=PLOT_COLORS['text']),
                                showlegend=False,
                                margin=dict(t=10, b=10, l=10, r=10)
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Container de Performance
                    st.markdown("<div style='margin: 2rem 0 1.5rem 0;'></div>", unsafe_allow_html=True)
                    
                    with st.container():
                        st.markdown("#### Performance do Modelo")
                        
                        from sklearn.metrics import classification_report, confusion_matrix
                        
                        report = classification_report(y_true, predictions, output_dict=True)
                        
                        # Métricas em grid
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-label">Precision</div>
                                    <div class="metric-value">{report['1']['precision']:.1%}</div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-label">Recall</div>
                                    <div class="metric-value">{report['1']['recall']:.1%}</div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-label">F1-Score</div>
                                    <div class="metric-value">{report['1']['f1-score']:.3f}</div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with col4:
                            st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-label">Accuracy</div>
                                    <div class="metric-value">{report['accuracy']:.1%}</div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("<div style='margin: 1.5rem 0 1rem 0;'></div>", unsafe_allow_html=True)
                        
                        # Matriz de confusão centralizada
                        cm = confusion_matrix(y_true, predictions)
                        
                        col_cm1, col_cm2, col_cm3 = st.columns([0.5, 2, 0.5])
                        with col_cm2:
                            fig = go.Figure(data=go.Heatmap(
                                z=cm,
                                x=['Predito: Normal', 'Predito: Anomalia'],
                                y=['Real: Normal', 'Real: Anomalia'],
                                colorscale=[[0, '#0e1117'], [1, PLOT_COLORS['primary']]],
                                text=cm,
                                texttemplate='%{text}',
                                textfont=dict(size=18, color=PLOT_COLORS['text']),
                                showscale=False
                            ))
                            
                            fig.update_layout(
                                height=300,
                                paper_bgcolor=PLOT_COLORS['bg'],
                                plot_bgcolor=PLOT_COLORS['bg'],
                                font=dict(family=PLOT_COLORS['font'], size=PLOT_COLORS['font_size'], color=PLOT_COLORS['text']),
                                margin=dict(t=10, b=50, l=80, r=10)
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Dataset não encontrado para demonstração.")
        
        with test_tab2:
            # Upload Section - Premium
            uploaded_file = st.file_uploader(
                "Selecione um arquivo CSV",
                type=['csv'],
                help="O arquivo deve conter 558 colunas de features",
                label_visibility="collapsed"
            )
            
            if uploaded_file:
                try:
                    # Carregar dados
                    test_data = pd.read_csv(uploaded_file)
                    
                    st.caption(f"✓ {len(test_data)} registros carregados")
                    
                    # Preview e botão lado a lado
                    col_preview, col_btn = st.columns([3, 1])
                    
                    with col_preview:
                        with st.expander("Visualizar dados do arquivo", expanded=False):
                            st.dataframe(test_data.head(10), use_container_width=True, height=250)
                    
                    with col_btn:
                        st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
                        analyze_upload_button = st.button("Analisar", type="primary", use_container_width=True, key="analyze_upload")
                    
                    st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
                    
                    # Botão para processar
                    if analyze_upload_button:
                        with st.spinner("Processando dados através do autoencoder..."):
                            # Preparar dados
                            # Remover colunas Time e Pass/Fail se existirem
                            cols_to_drop = []
                            if 'Time' in test_data.columns:
                                cols_to_drop.append('Time')
                            if 'Pass/Fail' in test_data.columns:
                                y_true = test_data['Pass/Fail'].replace({-1: 0, 1: 1})
                                cols_to_drop.append('Pass/Fail')
                                has_labels = True
                            else:
                                has_labels = False
                            
                            X_test = test_data.drop(columns=cols_to_drop) if cols_to_drop else test_data
                            
                            # Normalizar dados
                            scaler = StandardScaler()
                            X_test_scaled = scaler.fit_transform(X_test)
                            
                            # Fazer predições
                            reconstructions = model.predict(X_test_scaled, verbose=0)
                            
                            # Calcular erro de reconstrução
                            reconstruction_errors = tf.keras.losses.mae(reconstructions, X_test_scaled).numpy()
                            
                            # Classificar
                            predictions = (reconstruction_errors > threshold).astype(int)
                        
                        st.markdown("<div style='margin: 2rem 0 1.5rem 0;'></div>", unsafe_allow_html=True)
                        
                        # Container de Resultados - Premium
                        with st.container():
                            st.markdown("#### Resultados")
                            
                            # Estatísticas
                            normal_count = (predictions == 0).sum()
                            anomaly_count = (predictions == 1).sum()
                            anomaly_rate = (anomaly_count / len(predictions)) * 100
                            
                            # Cards em grid harmonioso
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.markdown(f"""
                                    <div class="metric-card">
                                        <div class="metric-label">Amostras</div>
                                        <div class="metric-value">{len(predictions)}</div>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(f"""
                                    <div class="metric-card">
                                        <div class="metric-label">Normal</div>
                                        <div class="metric-value">{normal_count}</div>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            with col3:
                                st.markdown(f"""
                                    <div class="metric-card">
                                        <div class="metric-label">Anomalias</div>
                                        <div class="metric-value">{anomaly_count}</div>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            with col4:
                                st.markdown(f"""
                                    <div class="metric-card">
                                        <div class="metric-label">Taxa</div>
                                        <div class="metric-value">{anomaly_rate:.1f}%</div>
                                    </div>
                                """, unsafe_allow_html=True)
                        
                        st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
                        
                        # Container de Visualizações
                        with st.container():
                            st.markdown("#### Análise Visual")
                            col1, col2 = st.columns([1.3, 1])
                            
                            with col1:
                                fig = go.Figure()
                                
                                fig.add_trace(go.Histogram(
                                    x=reconstruction_errors[predictions == 0],
                                    name='Normal',
                                    marker=dict(color=PLOT_COLORS['success'], opacity=0.7),
                                    nbinsx=30
                                ))
                                
                                fig.add_trace(go.Histogram(
                                    x=reconstruction_errors[predictions == 1],
                                    name='Anomalia',
                                    marker=dict(color=PLOT_COLORS['primary'], opacity=0.7),
                                    nbinsx=30
                                ))
                                
                                # Adicionar linha do threshold
                                fig.add_vline(
                                    x=threshold,
                                    line_dash="dash",
                                    line_color="#ffb627",
                                    line_width=3,
                                    annotation_text=f"Threshold ({threshold})",
                                    annotation_position="top"
                                )
                                
                                fig.update_layout(
                                    barmode='overlay',
                                    height=350,
                                    paper_bgcolor=PLOT_COLORS['bg'],
                                    plot_bgcolor=PLOT_COLORS['bg'],
                                    font=dict(family=PLOT_COLORS['font'], size=PLOT_COLORS['font_size'], color=PLOT_COLORS['text']),
                                    xaxis_title="Erro de Reconstrução",
                                    yaxis_title="Contagem",
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                    margin=dict(t=30, b=50, l=50, r=20)
                                )
                                
                                fig.update_xaxes(showgrid=True, gridcolor=PLOT_COLORS['grid'])
                                fig.update_yaxes(showgrid=True, gridcolor=PLOT_COLORS['grid'])
                                
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                fig = go.Figure(data=[go.Pie(
                                    labels=['Normal', 'Anomalia'],
                                    values=[normal_count, anomaly_count],
                                    hole=0.5,
                                    marker=dict(colors=[PLOT_COLORS['success'], PLOT_COLORS['primary']]),
                                    textinfo='label+percent+value',
                                    textfont=dict(size=14, color=PLOT_COLORS['text'])
                                )])
                                
                                fig.update_layout(
                                    height=350,
                                    paper_bgcolor=PLOT_COLORS['bg'],
                                    plot_bgcolor=PLOT_COLORS['bg'],
                                    font=dict(family=PLOT_COLORS['font'], size=PLOT_COLORS['font_size'], color=PLOT_COLORS['text']),
                                    showlegend=False,
                                    margin=dict(t=10, b=10, l=10, r=10)
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Se tiver labels verdadeiros, mostrar métricas
                        if has_labels:
                            st.markdown("<div style='margin: 2rem 0 1.5rem 0;'></div>", unsafe_allow_html=True)
                            
                            with st.container():
                                st.markdown("#### Performance do Modelo")
                                
                                from sklearn.metrics import classification_report, confusion_matrix
                                
                                # Relatório de classificação
                                report = classification_report(y_true, predictions, output_dict=True)
                                
                                # Métricas em grid
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.markdown(f"""
                                        <div class="metric-card">
                                            <div class="metric-label">Precision</div>
                                            <div class="metric-value">{report['1']['precision']:.1%}</div>
                                        </div>
                                    """, unsafe_allow_html=True)
                                
                                with col2:
                                    st.markdown(f"""
                                        <div class="metric-card">
                                            <div class="metric-label">Recall</div>
                                            <div class="metric-value">{report['1']['recall']:.1%}</div>
                                        </div>
                                    """, unsafe_allow_html=True)
                                
                                with col3:
                                    st.markdown(f"""
                                        <div class="metric-card">
                                            <div class="metric-label">F1-Score</div>
                                            <div class="metric-value">{report['1']['f1-score']:.3f}</div>
                                        </div>
                                    """, unsafe_allow_html=True)
                                
                                with col4:
                                    st.markdown(f"""
                                        <div class="metric-card">
                                            <div class="metric-label">Accuracy</div>
                                            <div class="metric-value">{report['accuracy']:.1%}</div>
                                        </div>
                                    """, unsafe_allow_html=True)
                            
                            st.markdown("<div style='margin: 1.5rem 0 1rem 0;'></div>", unsafe_allow_html=True)
                            
                            # Matriz de confusão centralizada
                            cm = confusion_matrix(y_true, predictions)
                            
                            col_cm1, col_cm2, col_cm3 = st.columns([0.5, 2, 0.5])
                            with col_cm2:
                                fig = go.Figure(data=go.Heatmap(
                                    z=cm,
                                    x=['Predito: Normal', 'Predito: Anomalia'],
                                    y=['Real: Normal', 'Real: Anomalia'],
                                    colorscale=[[0, '#0e1117'], [1, PLOT_COLORS['primary']]],
                                    text=cm,
                                    texttemplate='%{text}',
                                    textfont=dict(size=18, color=PLOT_COLORS['text']),
                                    showscale=False
                                ))
                                
                                fig.update_layout(
                                    height=300,
                                    paper_bgcolor=PLOT_COLORS['bg'],
                                    plot_bgcolor=PLOT_COLORS['bg'],
                                    font=dict(family=PLOT_COLORS['font'], size=PLOT_COLORS['font_size'], color=PLOT_COLORS['text']),
                                    margin=dict(t=10, b=50, l=80, r=80)
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("<div style='margin: 2rem 0 1.5rem 0;'></div>", unsafe_allow_html=True)
                        
                        # Tabela de resultados
                        with st.container():
                            st.markdown("#### Detalhes")
                            
                            results_df = pd.DataFrame({
                                'Índice': range(len(predictions)),
                                'Erro de Reconstrução': reconstruction_errors,
                                'Classificação': ['Normal' if p == 0 else 'Anomalia' for p in predictions],
                                'Status': ['✅' if p == 0 else '⚠️' for p in predictions]
                            })
                            
                            if has_labels:
                                results_df['Label Real'] = ['Normal' if y == 0 else 'Anomalia' for y in y_true]
                                results_df['Correto'] = ['✅' if p == y else '❌' for p, y in zip(predictions, y_true)]
                            
                            # Filtros
                            show_option = st.radio(
                                "Filtrar:",
                                options=["Todos", "Apenas Anomalias", "Apenas Normais"],
                                horizontal=True
                            )
                            
                            if show_option == "Apenas Anomalias":
                                results_df = results_df[results_df['Classificação'] == 'Anomalia']
                            elif show_option == "Apenas Normais":
                                results_df = results_df[results_df['Classificação'] == 'Normal']
                            
                            st.dataframe(results_df, use_container_width=True, height=350)
                        
                        # Download dos resultados
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Resultados (CSV)",
                            data=csv,
                            file_name="secom_predictions.csv",
                            mime="text/csv"
                        )
                    
                except Exception as e:
                    st.error(f"❌ Erro ao processar arquivo: {str(e)}")
                    st.info("Certifique-se de que o arquivo CSV está no formato correto com 558 features.")
            
            else:
                # Instruções quando não há arquivo
                st.markdown("""
                <div class="info-box">
                    <h4>Instruções</h4>
                    <ol style='line-height: 1.8; font-size: 0.875rem;'>
                        <li>Prepare um arquivo CSV com dados de sensores SECOM</li>
                        <li>O arquivo deve conter 558 colunas de features numéricas</li>
                        <li>Opcionalmente, inclua colunas 'Time' e 'Pass/Fail' para validação</li>
                        <li>Selecione o threshold desejado antes da análise</li>
                    </ol>
                    <p style='margin-top: 1rem; font-size: 0.8125rem; opacity: 0.8;'>
                    <b>Nota:</b> Use o arquivo <code>data/secom_cleaned_dataset.csv</code> 
                    do projeto para testar o sistema.
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    else:
        st.error("❌ Modelo não encontrado. Verifique se o arquivo 'models/secom_autoencoder_model.keras' existe.")

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; color: var(--text-secondary); font-family: Poppins; 
                padding: 2rem 0 1rem 0; margin-top: 3rem; border-top: 1px solid var(--border-color);'>
        <p style='font-size: 0.7em; opacity: 0.5; margin: 0;'>
            SECOM Failure Prediction · Autoencoder Neural Network · Anomaly Detection · 
            Threshold 0.45 (Balanced) / 0.50 (Conservative)
        </p>
    </div>
""", unsafe_allow_html=True)

