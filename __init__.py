import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.neural_network import _multilayer_perceptron as mlp
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SequentialFeatureSelector as sfs
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import numpy as np


# leitura da base
csv_name = '0-Todos-CSV/25.csv'

base = pd.read_csv(csv_name)

# numero de objetos de cada classe
# print(base['0'].value_counts())

#separação das features e classes
X =base.drop(columns=['0'])
y = base.pop('0')

# separação dos conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42,
                                                     stratify=y)

features = X_train.columns.to_list()

X_train = X_train.values
y_train=y_train.values

# -----------------SELECAO DE FEATURES--------------------
n_estimator_list = [5,50,100,250,500,1000,1500,3000]
X_train_new = []
models = []
for n in n_estimator_list:
    clf = ExtraTreesClassifier(n_estimators=n)
    clf = clf.fit(X_train, y_train)
    feature_importances =clf.feature_importances_
    # print(feature_importances_)

    model = SelectFromModel(clf, prefit=True)
    X_train_new.append(model.transform(X_train))
    models.append(model)

for base in X_train_new:
    print(base.shape)

# salvamento das bases de treino e teste após seleção de features

selected_features = models[3].get_support()

aux = []
for i in range(len(features)):
    if selected_features[i]:
        aux.append(features[i])
aux = np.array(aux)

df = pd.DataFrame(X_train_new[3],columns=aux)
df['CLASSE'] = y_train
df.to_csv('./src/base_treino_preprocessada.csv',index=False)

df2 = X_test
df2['CLASSE'] = y_test
df2.to_csv('./src/base_teste.csv',index=False)

# ----------------------------------------------------------------

# ----------------------

# sfs_knn = sfs(knn(n_neighbors=3))
# sfs_knn=sfs_knn.fit(X_train,y_train)
# print(sfs_knn.transform(X_train).shape())
