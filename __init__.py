import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.neural_network import _multilayer_perceptron as mlp
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import numpy as np

# -----------------SELECAO DE FEATURES--------------------
# # leitura da base
# csv_name = '0-Todos-CSV/25.csv'

# base = pd.read_csv(csv_name)

# # numero de objetos de cada classe
# # print(base['0'].value_counts())

# #separação das features e classes
# X =base.drop(columns=['0'])
# y = base.pop('0')

# # separação dos conjuntos de treino e teste
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42,
#                                                      stratify=y)

# features = X_train.columns.to_list()

# X_train = X_train.values
# y_train=y_train.values

# n_estimator_list = [5,50,100,250,500,1000,1500,3000]
# X_train_new = []
# models = []
# for n in n_estimator_list:
#     clf = ExtraTreesClassifier(n_estimators=n)
#     clf = clf.fit(X_train, y_train)
#     feature_importances =clf.feature_importances_
#     # print(feature_importances_)

#     model = SelectFromModel(clf, prefit=True)
#     X_train_new.append(model.transform(X_train))
#     models.append(model)

# for base in X_train_new:
#     print(base.shape)

# # salvamento das bases de treino e teste após seleção de features

# selected_features = models[3].get_support()

# aux = []
# for i in range(len(features)):
#     if selected_features[i]:
#         aux.append(features[i])
# aux = np.array(aux)

# df = pd.DataFrame(X_train_new[3],columns=aux)
# df['CLASSE'] = y_train
# df.to_csv('./src/base_treino_preprocessada.csv',index=False)

# df2 = X_test
# df2['CLASSE'] = y_test
# df2.to_csv('./src/base_teste.csv',index=False)

# ----------------------------------------------------------------

# ----------------------GRID SEARCH--------------------
base_treino = pd.read_csv('./src/base_treino_preprocessada.csv')

# adequação da base de treino às features presentes na base de teste
base_teste = pd.read_csv('./src/base_teste.csv',usecols=base_treino.columns.to_list())

assert len(base_treino.columns) == len(base_teste.columns), "A base de treino possui features diferentes da base de teste."

# separação das bases de treino e teste para aplicação nos modelos
X_train = base_treino.drop(columns=['CLASSE'])
y_train = base_treino.pop('CLASSE')

X_test = base_teste.drop(columns=['CLASSE'])
y_test = base_teste.pop('CLASSE')

# definição dos hiperparâmetros para o gridsearch
grid_knn = {
    'metric':['euclidean', 'haversine', 'manhattan'],
    'n_neighbors':[3,5,7,9],
    'weights':['uniform','distance'],
}

grid_mlp = {
    'hidden_layer_sizes':[(100,),(50,),(100,100),(100,50),(50,50)],
    'learning_rate':['constant', 'adaptative'],
    'activation':['tanh', 'logistic'],
    'solver':['adam', 'sgd']
}

KNN = knn()
MLP = mlp()
DTC = dtc()

clf_knn = GridSearchCV(KNN, grid_knn,cv=10)
clf_mlp = GridSearchCV(MLP, grid_knn,cv=10)
clf_dtc = GridSearchCV(DTC, grid_knn,cv=10)


