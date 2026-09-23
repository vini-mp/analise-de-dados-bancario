# %%
import pandas as pd
import matplotlib.pyplot as plt

#Organização da apresentação dos dados no terminal
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

#Importando dados do banco
df = pd.read_csv('dataset_risco_credito_500k.csv') 

# %%
#Verificando conteúdo do banco de dados
df.info()
df.head()
df.describe()



# %%
# 1. Estatísticas descritivas de tudo
print(df.describe(include='all').T)

#%%
#Análise de dados duplicados
print("Linhas totalmente duplicadas:", df.duplicated().sum())
print("IDs de cliente duplicados:", df['id_cliente'].duplicated().sum())
#vert=False → deixa a caixa deitada (mais fácil de ler quando tem outliers longe)


# %%
