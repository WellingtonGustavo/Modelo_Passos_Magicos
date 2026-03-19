import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path

# --- CONFIGURAÇÃO DE PÁGINA ---
st.set_page_config(page_title="Dashboard Passos Mágicos", layout="wide")

# --- CAMINHOS SIMPLIFICADOS (RAIZ) ---
# Como o app.py agora está na raiz, buscamos as pastas diretamente
BASE_DIR = Path(__file__).resolve().parent

MODEL_FILE = BASE_DIR / "models" / "modelo_risco_rf.pkl"
FEATURES_FILE = BASE_DIR / "models" / "features_modelo.pkl"
# Verifique se o CSV está dentro de 'data' ou 'data/raw' conforme sua imagem
DATA_FILE = BASE_DIR / "data" / "pede_consolidado_limpo.csv" 

# --- FUNÇÕES DE CARREGAMENTO ---
@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_FILE)
    features = joblib.load(FEATURES_FILE)
    return model, features

@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)

# --- INICIALIZAÇÃO ---
try:
    modelo, features_treino = load_assets()
    df = load_data()
    model_loaded = True
except Exception as e:
    st.error(f"Erro ao carregar arquivos da raiz: {e}")
    st.info(f"O app tentou acessar: {DATA_FILE}")
    model_loaded = False

# --- INTERFACE ---
st.title("🧙‍♂️ Inteligência Educacional - Passos Mágicos")

if model_loaded:
    aba = st.sidebar.radio("Navegação", ["Dashboard Geral", "Simulador de Risco IA"])

    if aba == "Dashboard Geral":
        st.header("📊 Visão Geral dos Indicadores")
        
        # Filtros Dinâmicos
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            anos = st.multiselect("Anos Letivos", df['ANO_LETIVO'].unique(), default=df['ANO_LETIVO'].unique())
        with col_f2:
            pedras = st.multiselect("Pedras", df['Pedra'].unique(), default=df['Pedra'].unique())

        df_filt = df[(df['ANO_LETIVO'].isin(anos)) & (df['Pedra'].isin(pedras))]

        # Métricas Principais
        m1, m2, m3 = st.columns(3)
        m1.metric("Alunos Analisados", len(df_filt))
        m2.metric("Média INDE", round(df_filt['INDE'].mean(), 2))
        
        # Métrica de Risco (Caso a coluna exista no seu CSV)
        if 'PROBABILIDADE_RISCO' in df_filt.columns:
            risco_count = len(df_filt[df_filt['PROBABILIDADE_RISCO'] > 0.6])
            m3.metric("Alunos em Risco (IA)", risco_count)
        else:
            m3.metric("Alunos em Risco (IA)", "N/A")

        st.subheader("Distribuição do INDE por Pedra")
        fig = px.box(df_filt, x='Pedra', y='INDE', color='Pedra', points="all")
        st.plotly_chart(fig, use_container_width=True)

    elif aba == "Simulador de Risco IA":
        st.header("🔮 Simulador de Risco Preventivo")
        
        with st.form("form_ia"):
            col_a, col_b = st.columns(2)
            with col_a:
                ieg = st.slider("Engajamento (IEG)", 0.0, 10.0, 7.0)
                ips = st.slider("Social (IPS)", 0.0, 10.0, 7.0)
                ipp = st.slider("Psicopedagógico (IPP)", 0.0, 10.0, 7.0)
            with col_b:
                ipv = st.slider("Ponto de Virada (IPV)", 0.0, 10.0, 7.0)
                iaa = st.slider("Autoavaliação (IAA)", 0.0, 10.0, 7.0)
                pedra_map = {'Quartzo': 1, 'Ágata': 2, 'Ametista': 3, 'Topázio': 4}
                pedra_sel = st.selectbox("Pedra Atual", list(pedra_map.keys()))
                pv_sel = st.selectbox("Já atingiu PV?", ["Sim", "Não"])

            if st.form_submit_button("Calcular Risco"):
                input_data = pd.DataFrame([{
                    'IAA': iaa, 'IEG': ieg, 'IPS': ips, 'IPP': ipp, 
                    'IPV': ipv, 'PEDRA_NUM': pedra_map[pedra_sel], 
                    'PV_BIN': 1 if pv_sel == "Sim" else 0
                }])
                input_data = input_data[features_treino]
                prob = modelo.predict_proba(input_data)[0][1]
                
                if prob > 0.6:
                    st.error(f"⚠️ Alerta de Risco Alto: {prob:.1%}")
                elif prob > 0.3:
                    st.warning(f"🟡 Risco Moderado: {prob:.1%}")
                else:
                    st.success(f"✅ Risco Baixo: {prob:.1%}")