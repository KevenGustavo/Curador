# **🏛️ Curador.IA \- Reconhecimento de Arte**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg) ![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg) ![License](https://img.shields.io/badge/License-MIT-green.svg)

**Sistema de Visão Computacional capaz de identificar artistas e movimentos artísticos a partir de fotografias de obras (telas ou impressões), utilizando Deep Learning e técnicas robustas de pré-processamento.**

## **Sobre o Projeto**

Este projeto foi desenvolvido no contexto da disciplina de **Processamento de Imagens (PDI)**. O desafio proposto foi criar uma solução capaz de reconhecer obras de arte em condições não controladas — especificamente capturas de câmeras de celular apontadas para **monitores, projetores ou livros**, onde reflexos e distorções de perspectiva são comuns.

#### **Artistas Suportados**

O modelo (CNN) foi treinado para distinguir os traços visuais de 5 grandes mestres:

1. **Claude Monet** (Impressionismo)  
2. **Leonardo da Vinci** (Renascimento)  
3. **Pablo Picasso** (Cubismo)  
4. **Salvador Dalí** (Surrealismo)  
5. **Vincent van Gogh** (Pós-Impressionismo)

## **Estrutura do Projeto**

O repositório está organizado para separar a lógica da aplicação, o treinamento científico e os artefatos de teste:

```text
curador-ia/  
├── app/                   
│   ├── app.py                  # Código frontend e backend
│   ├── dados.py                # Base de Informações sobre os Artistas  
│   └── modelo\_artes\_v2.h5    # Modelo de Deep Learning treinado  
│  
├── training/              
│   ├── preparar\_dados.py         # Script de limpeza e organização do dataset  
│   └── Colab_Treino_Modelo.ipynb  # Jupyter Notebook (Treino do Modelo)  
│  
├── samples/               
│   └── print\_app.png   # Imagens para demonstração  
│  
├── requirements.txt     # Dependências do projeto  
└── README.md            # Documentação

```

## **Fluxo de Desenvolvimento**

O projeto seguiu um pipeline de desenvolvimento estruturado em quatro etapas, focando na integridade dos dados e na robustez do modelo para inferência em tempo real.

### 1. Preparação do Dataset
* **Fonte:** [Best Artworks of All Time (Kaggle)](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time).
* **Curadoria:** Execução do script `training/preparar_dados.py` para filtrar o dataset original (50 classes) para os 5 artistas alvo.
* **Balanceamento:** Aplicação de *down-sampling* para equalizar as classes em ~400 imagens cada, prevenindo viés (bias) nas predições.
* **Particionamento:** Divisão estratificada automática em conjuntos de Treino (80%) e Validação (20%).

### 2. Arquitetura do Modelo
Utilizamos uma abordagem de **Transfer Learning** para otimizar o desempenho com poucos dados:
* **Backbone:** `MobileNetV2` (pré-treinado no ImageNet) com pesos congelados para extração de características (feature extraction).
* **Custom Head:** Adição de camadas `GlobalAveragePooling2D` e `Dense` (Softmax) customizadas para a classificação final das 5 classes.

### 3. Estratégia de Treino
O treinamento foi realizado via Google Colab, com foco na adaptação de domínio (Imagens Digitais $\to$ Fotos de Tela):
* **Data Augmentation:** Configuração agressiva do `ImageDataGenerator` (rotação, zoom, variação de brilho e cisalhamento) para simular artefatos de captura reais.
* **Otimização:** Uso do otimizador `Adam` e função de perda Categorical Crossentropy.

### 4. Pipeline de Inferência
O fluxo de execução no frontend (`app/app.py`) aplica três conceitos fundamentais de Processamento de Imagens depois da captura da imagem e antes da visualização dos resultados:

1.  **Transformação Geométrica (Correção de Orientação):**
    * Leitura de metadados EXIF para identificar a orientação da captura.
    * Aplicação de matriz de rotação (90°, 180°, 270°) para garantir o alinhamento espacial correto da obra.
2.  **Reamostragem e Interpolação:**
    * Utilização do algoritmo de **Filtro Lanczos** (`Image.Resampling.LANCZOS`) para o redimensionamento (Resize) e corte (Crop) da imagem para 224x224px.
    * *Motivação:* O filtro Lanczos preserva melhor as bordas e detalhes de alta frequência (pinceladas) comparado à interpolação bilinear padrão.
3.  **Transformação de Intensidade:**
    * Normalização da matriz de pixels, convertendo o espaço de cor RGB de inteiros `[0, 255]` para ponto flutuante `[0.0, 1.0]`, adequando a entrada para a convergência da rede neural.

## **Instalação e Configuração**

Siga os passos abaixo para configurar e executar a aplicação na sua máquina.

#### **Pré-requisitos**

Para que o projeto funcione, seu ambiente precisa atender aos seguintes requisitos:

**Sistema e Ferramentas:**
* **Python 3.10 ou superior** - Essencial para compatibilidade com o Streamlit e demais bibliotecas.
* **Git** - Para clonar o repositório.
* **Webcam** (Ou câmera do celular conectada) para testes em tempo real.

**Bibliotecas Python Utilizadas (Instalada via `requirements.txt`):**
* `tensorflow-cpu`: Motor de Inteligência Artificial.
* `streamlit`: Framework da interface web.
* `pillow`: Biblioteca de manipulação de imagens (PIL).
* `numpy` & `pandas`: Processamento matemático e de dados.

#### **1\. Clonar o Repositório**

```text
$ git clone https://github.com/KevenGustavo/Curador.git
$ cd curador-ia
```

#### **2\. Criar Ambiente Virtual (Recomendado)**

#### Windows  
```text
$ python \-m venv venv  
$ .\\venv\\Scripts\\activate
```

#### Linux/Mac 
```text
$ python3 \-m venv venv  
$ source venv/bin/activate
```

#### **3\. Instalar Dependências**

```text
$ pip install \-r requirements.txt
```

#### **4\. Executar o App**

Como o arquivo principal está dentro da pasta `App`, o comando é:

```text
$ streamlit run App/app.py
```

O navegador abrirá automaticamente em: **http://localhost:8501/**.

**Nota**: Ao testar no PC, você precisará de uma webcam. Se estiver acessando pelo celular na mesma rede Wi-Fi, o Streamlit fornecerá um "Network URL".

## Referências e Créditos
As seguintes fontes e documentações foram utilizadas como base para o desenvolvimento deste projeto:

* **Dataset:** [Best Artworks of All Time - Kaggle](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time)
* **Modelo de Arquitetura:** [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
* **Documentações:**
    * [TensorFlow/Keras Documentation](https://www.tensorflow.org/api_docs)
    * [Streamlit Documentation](https://docs.streamlit.io/)
    * [Pillow (PIL) Handbook](https://pillow.readthedocs.io/en/stable/)

## **Autor**

Desenvolvido por Keven Gomes

* **Curso:** Engenharia da Computação 
* **Disciplina:** Processamento de Imagens
