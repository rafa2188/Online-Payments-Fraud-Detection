import pandas as pd
import numpy as np
import plotly.express as px  # Importe o Plotly Express aqui
import time

# Carregar os dados
data = pd.read_csv(r"H:\Projetos Portifolios\dados estudos Python\credit card.csv")

# Exibir as primeiras linhas do DataFrame
print(data.head())

# Verificar a presença de valores nulos
print(data.isnull().sum())

# Explorar a distribuição do tipo de transação
print(data['type'].value_counts())

# Mapear o tipo de transação para valores numéricos
data['type'] = data['type'].map({"CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4, "DEBIT": 5})
data['isFraud'] = data['isFraud'].map({0: "No Fraud", 1: "Fraud"})

# Exibir as primeiras linhas do DataFrame após a modificação
print(data.head())

# Contar o número de transações de cada tipo
type_counts = data['type'].value_counts()

# Mapear os índices para os nomes das transações
type_names = {1: "CASH_OUT", 2: "PAYMENT", 3: "CASH_IN", 4: "TRANSFER", 5: "DEBIT"}
transactions = [type_names[idx] for idx in type_counts.index]

# Criar o gráfico de pizza com legenda
figure = px.pie(data, 
                values=type_counts.values, 
                names=transactions, 
                hole=0.5, 
                title="Distribuição do Tipo de Transação",
                labels={'names': 'Tipo de Transação', 'values': 'Quantidade'})

figure.show()

# Calcular a correlação apenas entre colunas numéricas
numeric_data = data.select_dtypes(include=[np.number])
correlation = numeric_data.corr()
print(correlation)

# Dividir os dados em conjuntos de treinamento e teste
x = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
y = np.array(data[["isFraud"]])

# Dividindo Dados
from sklearn.model_selection import train_test_split
x = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
y = np.array(data[["isFraud"]])



# Treinar um modelo de aprendizado de máquina
start_training = time.time()
from sklearn.tree import DecisionTreeClassifier
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
end_training = time.time()
training_time = end_training - start_training

# Imprimir a precisão do modelo
print(model.score(xtest, ytest))

# Previsão
start_prediction = time.time()
features = np.array([[4, 9000.60, 9000.60, 0.0]])
prediction = model.predict(features)
end_prediction = time.time()
prediction_time = end_prediction - start_prediction

print("Tempo médio de treinamento:", training_time)
print("Tempo médio de previsão:", prediction_time)


