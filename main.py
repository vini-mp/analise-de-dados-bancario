# %%
import pandas as pd
import matplotlib.pyplot as plt

#Organização da apresentação dos dados no terminal
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

#Importando dados do banco
df = pd.read_csv('base_credito_com_categoricas.csv') 

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
#Verificando valores que mais se repetem a fim de detectar algum código implítico
#que pode vir a atrapalhar as análises
df['renda_mensal_brl'].value_counts().head(10)



# %%
#Verificando valores que mais se repetem a fim de detectar algum código implítico
#que pode vir a atrapalhar as análises (TODAS AS COLUNAS)
colunas_para_checar = ['idade', 'tempo_relacionamento_meses', 'limite_credito_brl',
                        'uso_limite_percentual', 'atraso_historico_dias',
                        'score_bureau', 'qtd_consultas_cpf_ultimos_30d']

for col in colunas_para_checar:
    print(f"\n--- {col} ---")
    print(df[col].value_counts().head(3))


#Renda e idades suspeitas - utilizando flags para sinalizá-los
# %%
df['renda_suspeita'] = (df['renda_mensal_brl'] == 1200.00).astype(int)
df['idade_suspeita'] = (df['idade'] == 18).astype(int)

print("Total de renda suspeita:", df['renda_suspeita'].sum())
print("Total de idade suspeita:", df['idade_suspeita'].sum())


#Verificando a repetição de valores categóricos
# %%
print("--- estado ---")
print(df['estado'].value_counts())

print("\n--- profissao ---")
print(df['profissao'].value_counts())

print("\n--- tipo_de_conta ---")
print(df['tipo_de_conta'].value_counts())
# %%


#Análise de outliers
# %%
import matplotlib.pyplot as plt

colunas_numericas = ['idade', 'renda_mensal_brl', 'tempo_relacionamento_meses',
                      'limite_credito_brl', 'uso_limite_percentual',
                      'atraso_historico_dias', 'score_bureau',
                      'qtd_consultas_cpf_ultimos_30d']

fig, axes = plt.subplots(4, 2, figsize=(12, 14))
axes = axes.ravel()

for i, col in enumerate(colunas_numericas):
    axes[i].boxplot(df[col], vert=False)
    axes[i].set_title(col)

plt.tight_layout()
plt.show()
# %%

#Análise da relação da média com a variável inadimplentes (sem padronização)
# %%
colunas_numericas = ['idade', 'renda_mensal_brl', 'tempo_relacionamento_meses',
                      'limite_credito_brl', 'uso_limite_percentual',
                      'atraso_historico_dias', 'score_bureau',
                      'qtd_consultas_cpf_ultimos_30d']

df.groupby('inadimplente')[colunas_numericas].mean().round(2).T



#Análise variáveis numéricas com a variável alvo (padronizado)
#Fórmula de Cohen
# %%
def cohens_d(grupo1, grupo2):
    n1, n2 = len(grupo1), len(grupo2)
    var1, var2 = grupo1.var(), grupo2.var()
    # desvio padrão combinado (pooled)
    pooled_std = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    pooled_std = pooled_std ** 0.5
    return (grupo1.mean() - grupo2.mean()) / pooled_std

for col in colunas_numericas:
    inadimplentes = df[df['inadimplente'] == 1][col]
    adimplentes = df[df['inadimplente'] == 0][col]
    d = cohens_d(inadimplentes, adimplentes)
    print(f"{col}: d = {d:.3f}")


#Análise variáveis categóricas com a variável alvo
# %%
print("--- Taxa de inadimplência por estado ---")
print((df.groupby('estado')['inadimplente'].mean() * 100).round(2).sort_values(ascending=False))

print("\n--- Taxa de inadimplência por profissão ---")
print((df.groupby('profissao')['inadimplente'].mean() * 100).round(2).sort_values(ascending=False))

print("\n--- Taxa de inadimplência por tipo de conta ---")
print((df.groupby('tipo_de_conta')['inadimplente'].mean() * 100).round(2).sort_values(ascending=False))



#Cálculo da correlação das variáveis numéricas
# %%
colunas_numericas = ['idade', 'renda_mensal_brl', 'tempo_relacionamento_meses',
                      'limite_credito_brl', 'uso_limite_percentual',
                      'atraso_historico_dias', 'score_bureau',
                      'qtd_consultas_cpf_ultimos_30d']

matriz_correlacao = df[colunas_numericas].corr()
print(matriz_correlacao.round(2))

