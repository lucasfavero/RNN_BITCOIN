#Necessária a instalação das bibliotecas: keras, Numpy, Pandas, Matplotlib e Sklearn.

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#Download do dataset em: https://www.kaggle.com/mczielinski/bitcoin-historical-data

#Selecionando a base de dados
base = pd.read_csv('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv')

#Removendo linhas vazias 
base = base.dropna()

#Formatando a base com as coluna necessárias (selecionando apenas 5 features)
base = base[['Timestamp','Open','High','Low','Close']]

#Reduzindo a base só para os ultimos 30 dias: 09-12-2018 até 09-01-2019 
base = base.loc[base['Timestamp'] >= 1544320800]

#Criando uma base de treino e outra para teste 80/20 de maneira ordenada 
base_train, base_test = train_test_split(base, test_size=0.20, shuffle=False)

#Selecionando os valores para fazer as previsões 
#Selecionando a coluna de Open e transformando para o type nArray
base_treinamento  = base_train.iloc[:, 1:2].values

#Normalizando os valores de Open para valores entre 0,1 por uma questão de processamento 
normalizador  = MinMaxScaler(feature_range=(0,1))
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)

#Criando um intervalo de tempo para fazer a previsão de preços.
previsores  = []
preco_real = []

for i in range(90, base_treinamento_normalizada.shape[0]):
    previsores.append(base_treinamento_normalizada[i-90:i, 0])
    preco_real.append(base_treinamento_normalizada[i, 0]) 
previsores, preco_real = np.array(previsores), np.array(preco_real)
previsores = np.reshape(previsores , (previsores.shape[0], previsores.shape[1], 1))

#Criando a estrutura da rede neural (Long short term memory (LSTM))
#Return_sequences = true, serve para passar a informaçao para as camadas adiantes 
#Units = 4, é a quantidade de unidades de "memoria" do neuronio da rede LSTM

regressor = Sequential()
regressor.add(LSTM(units = 4, return_sequences= True, input_shape = (previsores.shape[1], 1)))
regressor.add(Dropout(0.3)) 

regressor.add(LSTM(units = 2)) 
regressor.add(Dropout(0.3))

regressor.add(Dense(units=1, activation='linear')) 

regressor.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mean_absolute_error'])

#Treinando a rede e salvando suas informaçoes na variavel history
history = regressor.fit(previsores, preco_real, epochs=10, batch_size=32) #epocas 10 por uma questão de processamento, batch_size 32 pq é "padrao"

#Salvando a rede treinada dentro da pasta do codigo
regressor.save('model.h5')

#Pegando os preços da base de teste
preco_real_teste = base_test.iloc[:, 1:2].values

#Concatenando as bases para pegar o numero certo de registros
base_completa = pd.concat((base['Open'], base_test['Open']), axis = 0)
entradas = base_completa[len(base_completa) - len(base_test) - 90:].values
entradas = entradas.reshape(-1, 1)
entradas = normalizador.transform(entradas)

#Criando o intervalo temporal para os preços reais de teste
X_teste = []
for i in range(90, entradas.shape[0]):
    X_teste.append(entradas[i-90:i, 0])
X_teste = np.array(X_teste)
X_teste = np.reshape(X_teste, (X_teste.shape[0], X_teste.shape[1], 1))
previsoes = regressor.predict(X_teste)
previsoes = normalizador.inverse_transform(previsoes)

#Validando 
print("Valor médio da previsão: $", previsoes.mean(), "\nValor médio do teste: $", preco_real_teste.mean(), 
      "\nDiferença: $", previsoes.mean()-preco_real_teste.mean())

#Plotando os graficos das bases de treino e teste
plt.plot(preco_real_teste, color = 'red', label = 'Preço real')
plt.plot(previsoes, color = 'blue', label = 'Previsões')
plt.title('Previsão preço do Bitcoin')
plt.xlabel('Tempo')
plt.ylabel('Valor Bitcoin')
plt.legend()
plt.show()

#Plotando curva do loss
loss = history.history['loss']
mean_absolute_error = history.history['mean_absolute_error']
plt.plot(loss, color='red', label='Loss')
plt.title('Grafico de curva do loss')
plt.legend()
plt.show()
