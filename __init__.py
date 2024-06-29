import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.neural_network import _multilayer_perceptron as mlp
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SequentialFeatureSelector as sfs


# leitura da base
base = pd.read_csv('0-Todos-CSV/25.csv')

# numero de objetos de cada classe
# print(base['0'].value_counts())

#separação das features e classes
X =base.drop(columns=['0'])
y = base.pop('0')

# separação dos conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42,
                                                     stratify=y)

# seleção de features para cada classificador utilizando
# o sequential feature selection

sfs_knn = sfs(knn(n_neighbors=3))
sfs_knn=sfs_knn.fit(X_train,y_train)
print(sfs_knn.transform(X_train).shape())
