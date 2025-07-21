# 📈 Previsão de Preços de Ações com LSTM

Este projeto utiliza redes neurais LSTM (Long Short-Term Memory) para prever o **preço de fechamento ajustado do próximo dia útil** de uma ação da B3 (Bolsa de Valores do Brasil), com base nos últimos 60 dias históricos. O sistema foi implementado tanto em **Jupyter Notebook** para fins acadêmicos quanto em uma aplicação **web com Flask**, pronta para uso prático.

---

## 🧠 Tecnologias Utilizadas

- Python
- TensorFlow / Keras
- Pandas / NumPy / Scikit-Learn
- Matplotlib
- yfinance
- Flask
- HTML / CSS (interface web)

---

## 🎯 Objetivo

Demonstrar, de forma prática, como aplicar modelos LSTM para previsão de séries temporais financeiras, utilizando dados reais do mercado de ações brasileiro.

---

## 🔬 Versão Notebook

A versão acadêmica em Jupyter Notebook demonstra todo o processo:

1. **Coleta de dados históricos** com `yfinance`
2. **Pré-processamento** com normalização `MinMaxScaler`
3. **Criação das janelas temporais** com 60 dias
4. **Construção do modelo LSTM** com duas camadas + Dropout
5. **Treinamento** com Keras
6. **Previsão do próximo valor**
7. **Visualização da curva de erro**

📂 Localização: `notebook/Previsao_Acoes.ipynb`

---

## 🌐 Versão Flask (Aplicação Web)

A versão web oferece uma interface amigável para prever preços de ações de forma automática, incluindo:

- Campo de busca com **autocompletar** de empresas da B3
- **Atualização automática** dos modelos com fine-tuning
- Previsão em tempo real do **próximo fechamento ajustado**
- Armazenamento local dos modelos e scalers
- Estilo moderno com HTML5 + CSS

---

## 🚀 Como Executar Localmente

1. **Clone o repositório**
   ```bash
   git clone https://github.com/rickbamberg/Previsao_Acoes.git
   cd Previsao_Acoes

2. **Crie e ative um ambiente virtual**

No terminal

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate      # Windows


3. **Instale as dependências**

pip install -r requirements.txt

4. **Execute a aplicação**

python app.py

5. **Acesse no navegador:**

http://localhost:5000

### 🗂 Estrutura de Pastas

Previsao_Acoes/  
├── app.py  
├── data/  
│   └── Tickers_B3.csv  
├── models/  
│   ├── petr4.sa_model.keras  
│   └── petr4.sa_scaler.pkl  
├── notebook/  
│   └── Previsao_Acoes.ipynb  
├── static/  
│   └── css/  
│       └── style.css  
├── templates/  
│   └── index.html  
├── venv/  
└── requirements.txt  

