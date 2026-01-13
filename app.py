import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

st.set_page_config(
    page_title="Curador.IA",
    page_icon="🎨",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    /* Importando fontes elegantes do Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=Lato:wght@400;700&display=swap');

    /* Fundo geral da aplicação (Dark Mode Profundo) */
    .stApp {
        background-color: #121212;
        color: #E0E0E0;
    }

    /* Títulos (Fonte Cinzel - Estilo Museu) */
    h1, h2, h3 {
        font-family: 'Cinzel', serif !important;
        color: #D4AF37 !important; /* Dourado Metálico */
        text-align: center;
        font-weight: 700;
    }
    
    /* Texto normal (Fonte Lato - Leitura fácil) */
    p, div, label {
        font-family: 'Lato', sans-serif;
        color: #CCCCCC;
    }

    /* Esconder menu padrão do Streamlit e rodapé */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Estilo do Card de Resultado */
    .museum-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #333;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-top: 20px;
        text-align: center;
    }
    
    .artist-name {
        font-family: 'Cinzel', serif;
        font-size: 28px;
        color: #D4AF37;
        margin-bottom: 5px;
    }
    
    .art-movement {
        font-size: 16px;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 20px;
        border-bottom: 1px solid #444;
        padding-bottom: 10px;
    }
    
    .art-desc {
        font-size: 16px;
        line-height: 1.6;
        text-align: justify;
        color: #DDD;
    }

    /* Botões */
    .stButton>button {
        width: 100%;
        background-color: #333333;
        color: #D4AF37;
        border: 1px solid #D4AF37;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #D4AF37;
        color: #000;
    }
    
    /* Box de Confiança */
    .confidence-box {
        background-color: #252525;
        padding: 10px;
        border-radius: 5px;
        font-size: 14px;
        margin-top: 15px;
        color: #888;
    }
    </style>
""", unsafe_allow_html=True)

INFO_ARTISTAS = {
    'claude_monet': {
        'nome': 'Claude Monet',
        'movimento': 'Impressionismo',
        'desc': 'O mestre da luz e da cor. Monet não pintava objetos, pintava a luz refletida neles. Suas pinceladas rápidas e soltas capturam o momento efêmero da natureza.',
        'ano': '1840 - 1926'
    },
    'leonardo_da_vinci': {
        'nome': 'Leonardo da Vinci',
        'movimento': 'Renascimento',
        'desc': 'O arquétipo do gênio. Usou a técnica "Sfumato" para eliminar contornos bruscos, criando rostos misteriosos e realistas. Uniu ciência e arte como ninguém.',
        'ano': '1452 - 1519'
    },
    'pablo_picasso': {
        'nome': 'Pablo Picasso',
        'movimento': 'Cubismo',
        'desc': 'Picasso quebrou a perspectiva tradicional. Ele mostrava o objeto de frente e de lado ao mesmo tempo, usando formas geométricas para desconstruir a realidade.',
        'ano': '1881 - 1973'
    },
    'salvador_dali': {
        'nome': 'Salvador Dalí',
        'movimento': 'Surrealismo',
        'desc': 'Explorador do inconsciente e dos sonhos. Suas obras são ilógicas, bizarras e perturbadoras, misturando técnica clássica perfeita com alucinações visuais.',
        'ano': '1904 - 1989'
    },
    'vincent_van_gogh': {
        'nome': 'Vincent van Gogh',
        'movimento': 'Pós-Impressionismo',
        'desc': 'Emoção pura na tela. Usava cores vibrantes e pinceladas grossas e espirais para expressar sua turbulência mental e a beleza intensa que via no mundo.',
        'ano': '1853 - 1890'
    }
}

CLASSES = ['claude_monet', 'leonardo_da_vinci', 'pablo_picasso', 'salvador_dali', 'vincent_van_gogh']

@st.cache_resource
def carregar_modelo():
    return tf.keras.models.load_model('modelo_artes_v2.h5')

def processar_imagem(image):
    img = ImageOps.exif_transpose(image)
    img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array


st.markdown("<h1>CURADOR.IA</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888; margin-top: -15px;'>Reconhecimento de Arte via Inteligência Artificial</p>", unsafe_allow_html=True)
st.markdown("---")

with st.spinner("Preparando os olhos digitais..."):
    model = carregar_modelo()

tab_cam, tab_up = st.tabs(["Câmera", "Upload"])

img_file = None
with tab_cam:
    st.write("Aponte para a tela ou impressão:")
    cam_input = st.camera_input("Capturar", label_visibility="collapsed")
    if cam_input: img_file = cam_input

with tab_up:
    up_input = st.file_uploader("Escolher arquivo", type=["jpg", "png", "jpeg"])
    if up_input: img_file = up_input

if img_file:
    # Processamento
    img_original = Image.open(img_file)
    img_exibicao, img_ia = processar_imagem(img_original)
    
    # Predição
    prediction = model.predict(img_ia)
    indice = np.argmax(prediction)
    confianca = np.max(prediction) * 100
    artista = CLASSES[indice]
    info = INFO_ARTISTAS.get(artista)

    # Layout de Resultado
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 1. Imagem Centralizada
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.image(img_exibicao, caption="Obra Analisada", use_column_width=True)

    # 2. Card de Informação (HTML Puro para controle total do design)
    if confianca > 60:
        html_card = f"""
        <div class="museum-card">
            <div class="artist-name">{info['nome']}</div>
            <div class="art-movement">{info['movimento']} • {info['ano']}</div>
            <div class="art-desc">
                {info['desc']}
            </div>
            <div class="confidence-box">
                🔍 Nível de Certeza da IA: <b>{confianca:.1f}%</b>
            </div>
        </div>
        """
        st.markdown(html_card, unsafe_allow_html=True)
        st.balloons() # Efeito de festa se acertar
        
    else:
        st.error("Identificação Incerta")
        st.markdown(f"""
        <div style="background-color: #332222; padding: 15px; border-radius: 10px; border: 1px solid #AA4444;">
            <p style="color: #FF8888; text-align: center;">
                A IA está confusa. Suspeito que seja <b>{info['nome']}</b> ({confianca:.1f}%), 
                mas a imagem pode estar com reflexos ou muito distante.
            </p>
        </div>
        """, unsafe_allow_html=True)

# Rodapé discreto
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #444; font-size: 12px;'>Desenvolvido com TensorFlow & Streamlit</p>", unsafe_allow_html=True)