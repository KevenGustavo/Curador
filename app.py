import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# --- 1. BASE DE CONHECIMENTO (O "Cérebro" de Texto) ---
INFO_ARTISTAS = {
    'claude_monet': {
        'nome': 'Claude Monet',
        'movimento': 'Impressionismo',
        'desc': 'Focava na luz e na natureza. Famoso por pintar o mesmo jardim várias vezes em horários diferentes.'
    },
    'leonardo_da_vinci': {
        'nome': 'Leonardo da Vinci',
        'movimento': 'Renascimento',
        'desc': 'O homem da Renascença. Mestre do "sfumato" (técnica de suavizar contornos). Pintou a Mona Lisa.'
    },
    'pablo_picasso': {
        'nome': 'Pablo Picasso',
        'movimento': 'Cubismo',
        'desc': 'Desconstruía objetos em formas geométricas. Revolucionou a arte moderna.'
    },
    'salvador_dali': {
        'nome': 'Salvador Dalí',
        'movimento': 'Surrealismo',
        'desc': 'Imagens de sonhos, relógios derretendo e paisagens bizarras. Excêntrico e provocador.'
    },
    'vincent_van_gogh': {
        'nome': 'Vincent van Gogh',
        'movimento': 'Pós-Impressionismo',
        'desc': 'Usava pinceladas grossas e cores vibrantes para expressar emoção. Cortou a própria orelha.'
    }
}

# Se o seu output do Colab mostrou outra ordem, altere aqui.
CLASSES = ['claude_monet', 'leonardo_da_vinci', 'pablo_picasso', 'salvador_dali', 'vincent_van_gogh']

# --- 2. CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Detector de Arte", page_icon="🎨")

st.title("🎨 Reconhecedor de Obras")
st.write("Aponte a câmera para a tela do computador ou impressão.")

# --- 3. CARREGAR MODELO (Cache para não travar) ---
@st.cache_resource
def carregar_modelo():
    # Carrega o modelo que você treinou no Colab
    return tf.keras.models.load_model('modelo_artes_v2.h5')

# Carrega enquanto o usuário lê o título
with st.spinner("A carregar inteligência artificial..."):
    model = carregar_modelo()

# --- 4. PROCESSAMENTO DE IMAGEM (O segredo para funcionar em telas) ---
def processar_imagem(image):
    # Passo A: Corrigir rotação (se o celular mandou a foto deitada)
    img = ImageOps.exif_transpose(image)
    
    # Passo B: Smart Crop (Corte Inteligente)
    # Em vez de esmagar a imagem, cortamos o centro 224x224.
    # Isso remove bordas do monitor e foca na obra.
    img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
    
    # Passo C: Converter para Array e Normalizar (igual ao treino)
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalizar pixels entre 0 e 1
    img_array = np.expand_dims(img_array, axis=0) # Criar lote de 1 imagem
    
    return img, img_array

# --- 5. INTERFACE DA CÂMERA ---
img_file = st.camera_input("Tire uma foto")

if img_file:
    # 1. Abrir imagem
    imagem_original = Image.open(img_file)
    
    # 2. Processar
    img_exibicao, img_para_ia = processar_imagem(imagem_original)
    
    # 3. Previsão
    prediction = model.predict(img_para_ia)
    indice = np.argmax(prediction)      # Qual posição tem o maior número?
    confianca = np.max(prediction) * 100 # Qual a % de certeza?
    
    classe_detectada = CLASSES[indice]
    info = INFO_ARTISTAS.get(classe_detectada)

    # 4. Mostrar Resultados
    st.divider()
    
    # Colunas para organizar (Foto processada na esq, Texto na dir)
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(img_exibicao, caption="O que a IA viu", use_column_width=True)
    
    with col2:
        if confianca > 60: # Só mostra se tiver certeza mínima
            st.success(f"Autor: **{info['nome']}**")
            st.write(f"**Movimento:** {info['movimento']}")
            st.info(info['desc'])
            st.caption(f"Certeza da IA: {confianca:.1f}%")
        else:
            st.warning("Não consegui identificar com clareza.")
            st.write(f"Meu palpite: {info['nome']} ({confianca:.1f}%)")
            st.write("Tente aproximar a câmera ou evitar reflexos na tela.")