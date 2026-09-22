import pandas as pd

#Organização da apresentação dos dados no terminal
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

#Importando dados do banco
df = pd.read_csv('dataset_risco_credito_500k.csv') 
df.info()
df.head()
df.describe()