import os
import shutil
import random

# --- CONFIGURAÇÕES ---
ORIGEM_DATASET = 'images/images' 
DESTINO_TREINO = 'dataset/train'
DESTINO_VALIDACAO = 'dataset/valid'

ARTISTAS_ALVO = [
    'Pierre_Auguste_Renoir', 
    'Albrecht_Durer',         
    'Pablo_Picasso',
    'Salvador_Dali',
    'Vincent_van_Gogh'
]

MAX_IMAGENS = 400 
SPLIT_VALIDACAO = 0.2

def limpar_pasta_antiga():
    if os.path.exists('dataset'):
        shutil.rmtree('dataset')
        print("Pasta antiga removida para evitar mistura de dados.")

def criar_pastas():
    for artista in ARTISTAS_ALVO:
        os.makedirs(os.path.join(DESTINO_TREINO, artista), exist_ok=True)
        os.makedirs(os.path.join(DESTINO_VALIDACAO, artista), exist_ok=True)

def processar_arquivos():
    for artista in ARTISTAS_ALVO:
        caminho_origem = os.path.join(ORIGEM_DATASET, artista)
        
        if not os.path.exists(caminho_origem) and artista == 'Albrecht_Durer':
             caminho_origem = os.path.join(ORIGEM_DATASET, 'Albrecht_Dürer')

        if not os.path.exists(caminho_origem):
            print(f"ERRO: Pasta não encontrada: {caminho_origem}")
            continue
            
        arquivos = [f for f in os.listdir(caminho_origem) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(arquivos)
        arquivos_selecionados = arquivos[:MAX_IMAGENS]
        
        qtd_validacao = int(len(arquivos_selecionados) * SPLIT_VALIDACAO)
        imgs_validacao = arquivos_selecionados[:qtd_validacao]
        imgs_treino = arquivos_selecionados[qtd_validacao:]
        
        print(f"{artista}: {len(imgs_treino)} treino | {len(imgs_validacao)} validação")
        
        for img in imgs_treino:
            shutil.copy(os.path.join(caminho_origem, img), os.path.join(DESTINO_TREINO, artista))
        for img in imgs_validacao:
            shutil.copy(os.path.join(caminho_origem, img), os.path.join(DESTINO_VALIDACAO, artista))

if __name__ == "__main__":
    limpar_pasta_antiga()
    criar_pastas()
    processar_arquivos()
    print("Dados prontos!")