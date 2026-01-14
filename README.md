# **🏛️ Curador.IA \- Reconhecimento de Arte**

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
│   ├── app.py             \# Código frontend e backend
│   ├── dados.py           \# Base de Informações sobre os Artistas  
│   └── modelo\_artes\_v2.h5 \# Modelo de Deep Learning treinado  
│  
├── training/              
│   ├── preparar\_dados.py  \# Script de limpeza e organização do dataset  
│   └── Colab_Treino_Modelo.ipynb     \# Jupyter Notebook (Treino do Modelo)  
│  
├── samples/               
│   └── print\_app.png      \# Imagens para demonstração  
│  
├── requirements.txt       \# Dependências do projeto  
└── README.md              \# Documentação

```

## **Tecnologias e Pipeline**

O sistema opera em um fluxo rigoroso de processamento:

#### **1\. Pré-processamento (PDI)**

* **Correção de Orientação (EXIF):** Utiliza Pillow para garantir que fotos de celular (verticais) sejam rotacionadas corretamente antes da análise.  
* **Smart Crop (Lanczos):** Realiza um corte central inteligente e redimensionamento para 224x224px, removendo bordas irrelevantes (molduras de monitor, fundos) sem distorcer a obra.  
* **Normalização:** Conversão de canais RGB (0-255) para float (0-1).

#### **2\. Inteligência Artificial (Deep Learning)**

* **MobileNetV2 (Transfer Learning):** Arquitetura baseada no ImageNet, otimizada para inferência rápida em CPU.  
* **Data Augmentation Agressivo:** O modelo foi treinado simulando:  
  * Variação de brilho (0.5x a 1.5x) para lidar com telas luminosas.  
  * Rotação e cisalhamento (shear) para lidar com fotos tiradas em ângulo.

#### **3\. Interface (Frontend)**

* **Streamlit:** Renderização da interface.  
* **Embedding Base64:** Técnica utilizada para renderizar as imagens processadas dentro de molduras CSS customizadas.  
* **Pandas:** Visualização gráfica das probabilidades de cada classe.

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
Bash

$ git clone https://github.com/KevenGustavo/Curador.git
$ cd curador-ia
```

#### **2\. Criar Ambiente Virtual (Recomendado)**

#### Windows  
```text
Bash

$ python \-m venv venv  
$ .\\venv\\Scripts\\activate
```

#### Linux/Mac 
```text
Bash

$ python3 \-m venv venv  
$ source venv/bin/activate
```

#### **3\. Instalar Dependências**

```text
Bash

$ pip install \-r requirements.txt
```

#### **4\. Executar o App**

Como o arquivo principal está dentro da pasta `App`, o comando é:

```text
Bash

$ streamlit run App/app.py
```

O navegador abrirá automaticamente em: **http://localhost:8501/**.

**Nota**: Ao testar no PC, você precisará de uma webcam. Se estiver acessando pelo celular na mesma rede Wi-Fi, o Streamlit fornecerá um "Network URL".

## **Autor**

Desenvolvido por Keven Gomes

* **Curso:** Engenharia da Computação 
* **Disciplina:** Processamento de Imagens