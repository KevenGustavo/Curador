import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import time
import pandas as pd 

# --- 1. CONFIGURAÇÃO INICIAL ---
st.set_page_config(
    page_title="Curador.IA",
    page_icon="🏛️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- 2. ESTILIZAÇÃO (CSS APRIMORADO) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=Lato:wght@300;400;700&display=swap');

    .stApp {
        background-color: #0E0E0E; /* Preto quase absoluto */
        color: #E0E0E0;
    }

    /* Tipografia */
    h1 {
        font-family: 'Cinzel', serif !important;
        color: #D4AF37 !important;
        text-align: center;
        font-weight: 700;
        letter-spacing: 4px;
        margin-bottom: 0px;
    }
    
    .subtitle {
        font-family: 'Lato', sans-serif;
        color: #888;
        text-align: center;
        font-size: 14px;
        letter-spacing: 1px;
        margin-top: -10px;
        margin-bottom: 30px;
    }

    /* Esconde elementos padrão */
    #MainMenu, footer, header {visibility: hidden;}

    /* MOLDURA DA OBRA (NOVO) */
    .art-frame {
        border: 8px solid #333;
        border-radius: 4px;
        box-shadow: 0px 10px 30px rgba(0,0,0,0.8);
        padding: 10px;
        background-color: #1a1a1a;
        margin-bottom: 20px;
    }

    /* CARD DO MUSEU */
    .museum-card {
        background: linear-gradient(145deg, #1E1E1E, #252525);
        padding: 25px;
        border-radius: 2px;
        border-top: 3px solid #D4AF37;
        box-shadow: 0 10px 20px rgba(0,0,0,0.5);
        text-align: center;
        animation: fadeIn 1s;
    }
    
    @keyframes fadeIn {
        0% {opacity: 0; transform: translateY(20px);}
        100% {opacity: 1; transform: translateY(0);}
    }
    
    .artist-name {
        font-family: 'Cinzel', serif;
        font-size: 32px;
        color: #D4AF37;
        margin-bottom: 5px;
        text-shadow: 0px 2px 4px rgba(0,0,0,0.5);
    }
    
    .art-meta {
        font-size: 14px;
        color: #AAA;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-bottom: 20px;
        border-bottom: 1px solid #444;
        padding-bottom: 15px;
    }
    
    .art-desc {
        font-family: 'Lato', sans-serif;
        font-size: 16px;
        line-height: 1.7;
        color: #DDD;
        font-weight: 300;
        text-align: justify;
    }
    
    /* Ajuste da câmera para parecer um scanner */
    div[data-testid="stCameraInput"] {
        border: 2px dashed #444;
        border-radius: 10px;
        padding: 10px;
    }
    
    div[data-testid="stCameraInput"]:hover {
        border-color: #D4AF37;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. DADOS ---
INFO_ARTISTAS = {
    'claude_monet': {'nome': 'Claude Monet', 'movimento': 'Impressionismo', 'ano': '1840 - 1926', 'desc': 'O mestre da luz e da cor. Monet não pintava objetos, pintava a luz refletida neles. Suas pinceladas rápidas capturam o momento efêmero.'},
    'leonardo_da_vinci': {'nome': 'Leonardo da Vinci', 'movimento': 'Renascimento', 'ano': '1452 - 1519', 'desc': 'O arquétipo do gênio. Mestre do "Sfumato" (suavização de contornos). Uniu ciência, anatomia e arte para criar rostos misteriosos e realistas.'},
    'pablo_picasso': {'nome': 'Pablo Picasso', 'movimento': 'Cubismo', 'ano': '1881 - 1973', 'desc': 'Picasso quebrou a perspectiva tradicional. Ele mostrava o objeto de frente e de lado ao mesmo tempo, usando formas geométricas para desconstruir a realidade.'},
    'salvador_dali': {'nome': 'Salvador Dalí', 'movimento': 'Surrealismo', 'ano': '1904 - 1989', 'desc': 'Explorador do inconsciente. Suas obras são ilógicas e oníricas, misturando técnica clássica rigorosa com alucinações visuais e objetos derretendo.'},
    'vincent_van_gogh': {'nome': 'Vincent van Gogh', 'movimento': 'Pós-Impressionismo', 'ano': '1853 - 1890', 'desc': 'Emoção pura na tela. Usava cores vibrantes, contrastes fortes e pinceladas grossas em espiral para expressar sua turbulência mental.'}
}

CLASSES = ['claude_monet', 'leonardo_da_vinci', 'pablo_picasso', 'salvador_dali', 'vincent_van_gogh']

# --- 4. BACKEND ---
@st.cache_resource
def carregar_modelo():
    return tf.keras.models.load_model('modelo_artes_v2.h5')

def processar_imagem(image):
    img = ImageOps.exif_transpose(image)
    img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array

# --- 5. UI PRINCIPAL ---
st.markdown("<h1>CURADOR.IA</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>SISTEMA DE VISÃO COMPUTACIONAL</div>", unsafe_allow_html=True)

# Carregamento silencioso
with st.spinner("Inicializando redes neurais..."):
    model = carregar_modelo()

# Input Direto (Sem abas)
img_file = st.camera_input("Aponte para a obra", label_visibility="collapsed")

if img_file:
    # Efeito de "Escaneando" 
    progress_text = "Escaneando obra..."
    my_bar = st.progress(0, text=progress_text)

    # Simulação de etapas de processamento
    etapas = [
        (20, "Normalizando pixels..."),
        (50, "Extraindo características visuais..."),
        (80, "Consultando banco de dados de estilos..."),
        (100, "Finalizando análise.")
    ]
    
    for percent, label in etapas:
        time.sleep(0.15) 
        my_bar.progress(percent, text=label)
    
    time.sleep(0.2)
    my_bar.empty() 

    # Processamento Real
    img_original = Image.open(img_file)
    img_exibicao, img_ia = processar_imagem(img_original)
    
    prediction = model.predict(img_ia)
    indice = np.argmax(prediction)
    confianca = np.max(prediction) * 100
    artista_key = CLASSES[indice]
    info = INFO_ARTISTAS.get(artista_key)

    # --- RESULTADOS ---
    
    # 1. Imagem com Moldura
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.markdown('<div class="art-frame">', unsafe_allow_html=True)
        st.image(img_exibicao, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Card Principal
    if confianca > 60:
        html_card = f"""
        <div class="museum-card">
            <div class="artist-name">{info['nome']}</div>
            <div class="art-meta">{info['movimento']} • {info['ano']}</div>
            <div class="art-desc">
                {info['desc']}
            </div>
        </div>
        """
        st.markdown(html_card, unsafe_allow_html=True)
        
        # 3. Dados Técnicos
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📊 Ver Dados da Rede Neural (Visão Técnica)"):
            st.write("Distribuição de probabilidade entre as classes:")
            
            probs = prediction[0] * 100
            df_probs = pd.DataFrame({
                'Artista': [INFO_ARTISTAS[k]['nome'] for k in CLASSES],
                'Confiança (%)': probs
            })
            
            st.bar_chart(df_probs.set_index('Artista'), color="#D4AF37")
            
            st.caption(f"Tempo de inferência: Instantâneo (MobileNetV2)")
            
    else:
        st.error("⚠️ Identificação Incerta")
        st.markdown(f"""
        <div style="background-color: #2a1a1a; padding: 20px; border-radius: 5px; border-left: 5px solid #ff4b4b; text-align: center;">
            <h3 style="color: #ff4b4b !important; font-size: 20px;">Análise inconclusiva</h3>
            <p>O algoritmo detectou traços de <b>{info['nome']}</b> ({confianca:.1f}%), 
            mas não atingiu o limiar de segurança.</p>
            <p style="font-size: 12px; color: #888;">Tente ajustar o ângulo ou reduzir o brilho da tela.</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)