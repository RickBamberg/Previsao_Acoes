# ================================
# PREVISOR DE AÇÕES COM LSTM - FLASK APP
# ================================
import os
# -------------------------------
# CONFIGURAÇÕES INICIAIS
# -------------------------------

# Suprime mensagens de log do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# --- Importações ---
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import joblib
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from flask import Flask, request, render_template
import datetime

# Inicialização do Flask 
app = Flask(__name__)

# Diretório para armazenar modelos
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True) # Garante que o diretório de modelos exista

# --------------------------------------------
# FUNÇÃO: Treinamento ou Atualização do Modelo
# --------------------------------------------
def get_or_update_model(ticker):
    """
    Verifica se um modelo existe e está atualizado.
    Se não, cria ou atualiza o modelo (fine-tuning).
    Retorna o modelo e o scaler prontos para uso.
    """
    model_path = os.path.join(MODEL_DIR, f"{ticker.lower()}_model.keras")
    scaler_path = os.path.join(MODEL_DIR, f"{ticker.lower()}_scaler.pkl")
    
    today_str = datetime.date.today().strftime('%Y-%m-%d')
    tamanho_sequencia = 60
    model_age_days = 0

    if os.path.exists(model_path):
       model_age_days = (datetime.datetime.now() - datetime.datetime.fromtimestamp(os.path.getmtime(model_path))).days

    # CASO 1: O modelo NÃO EXISTE. Treinamento completo.
    #         Ou o modelo não foi atualizado por mais de 90 dias
    if not os.path.exists(model_path) or model_age_days > 90:
        print(f"Modelo para {ticker} não encontrado ou desatualizado. Treinando do zero...")
        try:
            dados = yf.download(ticker, start='2018-01-01', end=today_str, auto_adjust=True)
            if len(dados) < tamanho_sequencia + 1:
                return None, None, f"Erro: Dados insuficientes para treinar o modelo para {ticker}."

            dados_fechamento = dados['Close'].values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            dados_scaled = scaler.fit_transform(dados_fechamento)

            X_train, y_train = [], []
            for i in range(tamanho_sequencia, len(dados_scaled)):
                X_train.append(dados_scaled[i-tamanho_sequencia:i, 0])
                y_train.append(dados_scaled[i, 0])

            X_train, y_train = np.array(X_train), np.array(y_train)
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

            modelo = Sequential([
                Input(shape=(X_train.shape[1], 1)),
                LSTM(units=50, return_sequences=True), Dropout(0.2),
                LSTM(units=50, return_sequences=False), Dropout(0.2),
                Dense(units=1)
            ])
            modelo.compile(optimizer='adam', loss='mean_squared_error')
            modelo.fit(X_train, y_train, epochs=25, batch_size=32, verbose=0)

            modelo.save(model_path)
            joblib.dump(scaler, scaler_path)
            print(f"Modelo para {ticker} treinado e salvo.")
            return modelo, scaler, None

        except Exception as e:
            return None, None, f"Erro ao treinar modelo inicial para {ticker}: {e}"

    # CASO 2: O modelo EXISTE. Verificar se precisa de atualização.
    else:
        model_mtime = datetime.date.fromtimestamp(os.path.getmtime(model_path))
        
        # Se o modelo já foi treinado/atualizado hoje, apenas carregue.
        if model_mtime >= datetime.date.today():
            print(f"Modelo para {ticker} já está atualizado. Carregando...")
            modelo = load_model(model_path)
            scaler = joblib.load(scaler_path)
            return modelo, scaler, None

        # Se o modelo é de dias anteriores, faça o fine-tuning.
        else:
            print(f"Modelo para {ticker} desatualizado. Fazendo fine-tuning...")
            try:
                # Baixa APENAS os dados novos
                last_train_date_str = (model_mtime + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
                novos_dados = yf.download(ticker, start=last_train_date_str, end=today_str, auto_adjust=True)

                if novos_dados.empty:
                    print("Nenhum dado novo para treinar. Usando modelo existente.")
                    modelo = load_model(model_path)
                    scaler = joblib.load(scaler_path)
                    return modelo, scaler, None
                
                # Carrega o scaler e modelo antigos
                scaler = joblib.load(scaler_path)
                modelo = load_model(model_path)

                # Pega os últimos dados do treino anterior para construir as novas sequências
                dados_antigos = yf.download(ticker, end=model_mtime.strftime('%Y-%m-%d'), auto_adjust=True)
                dados_combinados = pd.concat([dados_antigos.tail(tamanho_sequencia), novos_dados])['Close'].values.reshape(-1,1)

                # ATENÇÃO: usa o scaler ANTIGO para transformar os novos dados
                dados_combinados_scaled = scaler.transform(dados_combinados)

                X_new, y_new = [], []
                for i in range(tamanho_sequencia, len(dados_combinados_scaled)):
                    X_new.append(dados_combinados_scaled[i-tamanho_sequencia:i, 0])
                    y_new.append(dados_combinados_scaled[i, 0])

                X_new, y_new = np.array(X_new), np.array(y_new)
                X_new = np.reshape(X_new, (X_new.shape[0], X_new.shape[1], 1))
                
                # Recompila com taxa de aprendizado baixa para fine-tuning
                modelo.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')
                modelo.fit(X_new, y_new, epochs=10, batch_size=32, verbose=0)
                
                modelo.save(model_path) # Salvar o modelo ATUALIZA a data de modificação!
                print(f"Fine-tuning para {ticker} concluído e modelo salvo.")
                return modelo, scaler, None

            except Exception as e:
                return None, None, f"Erro durante o fine-tuning para {ticker}: {e}"

# -----------------------------------------
# CARREGAMENTO DE EMPRESAS PARA O FRONTEND
# -----------------------------------------
try:
    df_empresas = pd.read_csv('data/Tickers_B3.csv', sep=';')
    lista_para_template = df_empresas.apply(
        lambda row: {
            'ticker': row['Codigo'] + ".SA", # Adiciona o sufixo .SA que o yfinance precisa
            'nome': row['Empresa']
        },
        axis=1
    ).tolist()
    
except FileNotFoundError:
    print("AVISO: Arquivo 'Tickers_B3.csv' não encontrado. O combo de pesquisa não funcionará.")
    lista_para_template = []

# -------------------------------
# ROTA PRINCIPAL DO FLASK
# -------------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    form_data = {'ticker': 'PETR4.SA'}

    if request.method == 'POST':
        ticker = request.form['ticker']
        form_data['ticker'] = ticker

        if not ticker:
            prediction_result = "Erro: Ticker é obrigatório."
        else:
            # 1. Chama a função inteligente para obter o modelo
            modelo, scaler, error_msg = get_or_update_model(ticker)

            if error_msg:
                prediction_result = error_msg
            else:
                # 2. Prepara os dados para a previsão
                ultimos_dados = yf.download(ticker, period='3mo', auto_adjust=True)['Close'].values.reshape(-1, 1)
                ultimos_60_dias = ultimos_dados[-60:]
                ultimos_60_dias_scaled = scaler.transform(ultimos_60_dias)
                
                X_test = np.array([ultimos_60_dias_scaled])
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

                # 3. Faz a previsão
                preco_previsto_scaled = modelo.predict(X_test)
                preco_previsto = scaler.inverse_transform(preco_previsto_scaled)
                prediction_result = f"{ticker}: R$ {preco_previsto[0][0]:.2f}"
                # prediction_result = f"Previsão de Preço Ajustado para:\n {ticker}: R$ {preco_previsto[0][0]:.2f}"
    
    return render_template(
        'index.html', 
        prediction=prediction_result, 
        form_data=form_data, 
        empresas=lista_para_template
    )

# -------------------------------
# EXECUÇÃO DO SERVIDOR FLASK
# -------------------------------
if __name__ == '__main__':
    # Remover o `debug=True` se for para um ambiente de "produção"
    app.run(host='0.0.0.0', port=5000, debug=True)