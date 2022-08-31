import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('C:/Users/marconi.adm/PycharmProjects/prever_ajuste_inflação/GSPC.csv')
df.head()

df = df.drop('Date', axis=1)
df[-2::]
amanha = df[-1::]
base = df.drop(df[-1::].index, axis = 0)
base.tail()
base['target'] = base['Close'][1:len(base)].reset_index(drop = True)
prev = base[-1::].drop('target', axis = 1)
treino = base.drop(base[-1::].index, axis = 0)
treino.loc[treino['target'] > treino['Close'], 'target'] = 1
treino.loc[treino['target'] != 1, 'target'] = 0
y = treino['target']
x = treino.drop('target', axis = 1)
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.3)

from sklearn.ensemble import ExtraTreesClassifier

modelo = ExtraTreesClassifier()
modelo.fit(x_treino, y_treino)
resultado = modelo.score(x_teste, y_teste)
print(resultado)
resultado_bolsa = "A bolsa fechara em baixa"
if modelo.predict(prev) == [1]:
    resiltado_bolsa = "A bolsa fechara em alta"

print(resiltado_bolsa)



