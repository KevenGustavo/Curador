import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import time
import pandas as pd
import base64
from io import BytesIO
from dados import INFO_ARTISTAS

# --- 1. CONFIGURAÇÃO INICIAL ---
st.set_page_config(
    page_title="Curador",
    page_icon="🏛️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- 2. ESTILIZAÇÃO ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=Lato:wght@300;400;700&display=swap');

    .stApp { background-color: #0E0E0E; color: #E0E0E0; }

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

    #MainMenu, footer, header {visibility: hidden;}

    .art-frame {
        border: 8px solid #333;
        border-radius: 4px;
        box-shadow: 0px 10px 30px rgba(0,0,0,0.8);
        padding: 10px;
        background-color: #1a1a1a;
        margin-bottom: 20px;
    }

    .museum-card {
        background: linear-gradient(145deg, #1E1E1E, #252525);
        padding: 20px;
        border-radius: 4px;
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
        font-size: 28px;
        color: #D4AF37;
        margin-bottom: 5px;
        text-shadow: 0px 2px 4px rgba(0,0,0,0.5);
    }
    
    .art-meta {
        font-size: 12px;
        color: #AAA;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 20px;
        border-bottom: 1px solid #444;
        padding-bottom: 15px;
    }
    
    .art-desc {
        font-family: 'Lato', sans-serif;
        font-size: 15px;
        line-height: 1.6;
        color: #DDD;
        font-weight: 300;
        text-align: justify;
    }
    
    div[data-testid="stCameraInput"] {
        border: 2px dashed #444;
        border-radius: 10px;
        padding: 10px;
    }
    div[data-testid="stCameraInput"]:hover { border-color: #D4AF37; }

    /* --- CORREÇÃO BARRA CINZA (CSS) --- */
    div[data-testid="stEmpty"] {
        display: none;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. CONSTANTES ---
CLASSES = [
    'Albrecht_Durer', 
    'Pablo_Picasso', 
    'Pierre_Auguste_Renoir', 
    'Salvador_Dali', 
    'Vincent_van_Gogh'
]

# --- 4. BACKEND ---
@st.cache_resource
def carregar_modelo():
    return tf.keras.models.load_model('App/modelo_artes.h5') 

def processar_imagem(image):
    img = ImageOps.exif_transpose(image)
    img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array

def get_image_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

# --- 5. UI PRINCIPAL ---
st.markdown("<h1>CURADOR DE ARTE</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>SISTEMA DE VISÃO COMPUTACIONAL</div>", unsafe_allow_html=True)

lista_artistas_html = ""
for chave, dados_artista in INFO_ARTISTAS.items():
    lista_artistas_html += f'<span class="artist-badge">{dados_artista["nome"]}</span>'

st.markdown(f"""
<style>
    .catalog-card {{
        background-color: #262730;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #444;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }}
    .catalog-title {{
        font-size: 1.1rem;
        font-weight: 600;
        color: #E0E0E0;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
    }}
    .artist-badge {{
        display: inline-block;
        background-color: #3E3F4B;
        color: #F0F2F6;
        padding: 6px 12px;
        border-radius: 20px;
        margin-right: 8px;
        margin-bottom: 8px;
        font-size: 0.85rem;
        border: 1px solid #555;
        transition: all 0.3s ease;
    }}
    .artist-badge:hover {{
        background-color: #E69138;
        color: #121212;
        border-color: #E69138;
        cursor: default;
    }}
</style>

<div class="catalog-card">
    <div class="catalog-title">
         Acervo Disponível para Identificação
    </div>
    <div>
        {lista_artistas_html}
    </div>
    <div style="margin-top: 10px; font-size: 0.8rem; color: #888;">
        <i>*O modelo foi otimizado para reconhecer obras destes 5 mestres.</i>
    </div>
</div>
""", unsafe_allow_html=True)
# -----------------------------------

st.markdown("""
    <div style="text-align: center; margin-bottom: 20px; font-family: 'Lato', sans-serif;">
        <p style="color: #666; font-size: 13px; font-style: italic;">
             Aponte a câmera para a Obra e tire uma foto.
            <br>(Evite reflexos fortes e centralize a obra na foto para maior precisão)
        </p>
    </div>
""", unsafe_allow_html=True)

with st.spinner("Inicializando redes neurais..."):
    model = carregar_modelo()

img_file = st.camera_input("Aponte para a obra", label_visibility="collapsed")

if img_file:
    loading_placeholder = st.empty()
    
    progress_text = "Processando..."
    my_bar = loading_placeholder.progress(0, text=progress_text)
    
    for percent in [20, 50, 80, 100]:
        time.sleep(0.05)
        my_bar.progress(percent)
    time.sleep(0.1)
    
    loading_placeholder.empty()

    # Processamento
    img_original = Image.open(img_file)
    img_exibicao, img_ia = processar_imagem(img_original)
    
    prediction = model.predict(img_ia)
    indice = np.argmax(prediction)
    confianca = np.max(prediction) * 100
    artista_key = CLASSES[indice]
    
    info = INFO_ARTISTAS.get(artista_key)

    if info is None:
        st.error(f"Erro de Sistema: As informações para a classe '{artista_key}' não foram encontradas no arquivo 'dados.py'. Verifique se o nome no dicionário bate com a lista CLASSES.")
    else:
        col1, col2, col3 = st.columns([1, 10, 1])
        with col2:
            img_b64 = get_image_base64(img_exibicao)
            
            st.markdown(f"""
                <div class="art-frame">
                    <img src="data:image/jpeg;base64,{img_b64}">
                </div>
            """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        if confianca > 60:
            html_card = f"""
    <div class="museum-card">
        <div class="artist-name">{info['nome']}</div>
        <div class="art-meta">{info['movimento']} • {info['ano']}</div>
        <div class="art-desc">{info['desc']}</div>
        <hr style="border: 0; border-top: 1px solid #444; margin: 15px 0;">
        <div style="text-align: left; margin-bottom: 8px;">
            <span style="color: #D4AF37; font-weight: bold;">Obra mais famosa:</span> 
            <span style="color: #CCC;">{info['obra_prima']}</span>
        </div>
        <div style="text-align: left; margin-bottom: 8px;">
            <span style="color: #D4AF37; font-weight: bold;">Técnica:</span> 
            <span style="color: #CCC;">{info['tecnica']}</span>
        </div>
        <div style="background-color: #252525; padding: 10px; border-radius: 5px; margin-top: 15px; font-size: 13px; font-style: italic; color: #888; text-align: left;">
            <b>Curiosidade:</b> {info['curiosidade']}
        </div>
    </div>
    """
            st.markdown(html_card, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("Dados Técnicos"):
                probs = prediction[0] * 100
                df_probs = pd.DataFrame({
                    'Artista': [INFO_ARTISTAS.get(k, {'nome': k})['nome'] for k in CLASSES],
                    'Confiança (%)': probs
                })
                st.bar_chart(df_probs.set_index('Artista'), color="#D4AF37")
                
        else:
            st.error("Identificação Incerta")
            st.markdown(f"""
    <div style="background-color: #2a1a1a; padding: 20px; border-radius: 5px; border-left: 5px solid #ff4b4b; text-align: center;">
        <h3 style="color: #ff4b4b !important; font-size: 20px;">Análise inconclusiva</h3>
        <p>O algoritmo detectou traços de <b>{info['nome']}</b> ({confianca:.1f}%), 
        mas não atingiu o limiar de segurança.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)