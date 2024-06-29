import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import numpy as np
import matplotlib.pyplot as plt

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

# # remove valores nulos
# X_train = X_train.dropna()
# y_train = y_train.dropna()

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

# ----------------DISTRIBUIÇÃO DE CLASSES--------------------------
base_treino = pd.read_csv('./src/base_treino_preprocessada.csv')

# adequação da base de treino às features presentes na base de teste
base_teste = pd.read_csv('./src/base_teste.csv',usecols=base_treino.columns.to_list())

# printa valores nulos na base de teste (se existir)
for i in base_teste.isna().sum():
    if i!=0:
        print(i)

# -----------PLOT DISTRIBUIÇÃO DE CLASSES----------------------------------------
# elements_per_class = base_treino['CLASSE'].value_counts().sort_index()

# classes = np.arange(elements_per_class.index.min(),elements_per_class.index.max()+1,1)

# plt.bar(classes, elements_per_class)

# for i, data in enumerate(elements_per_class):
#     plt.text(x=i+1 , y =data+0.5 , s=f"{data}")

# plt.yticks([])
# plt.xticks(classes)
# plt.savefig("Elementos_por_classe.png")
# plt.show()
# ----------------------------------------------------------------------------

assert len(base_treino.columns) == len(base_teste.columns), "A base de treino possui features diferentes da base de teste."


# ----------------------GRID SEARCH--------------------
# separação das bases de treino e teste para aplicação nos modelos
X_train = base_treino.drop(columns=['CLASSE'])
y_train = base_treino.pop('CLASSE')

X_test = base_teste.drop(columns=['CLASSE'])
y_test = base_teste.pop('CLASSE')

# definição dos hiperparâmetros para o gridsearch
grid_knn = {
    'metric':['euclidean', 'manhattan'],
    'n_neighbors':[3,5,7,9],
    'weights':['uniform','distance'],
}

grid_mlp = {
    'hidden_layer_sizes':[(100,100,100),(100,100,50),(100,50,50),(50,50,50)],
    'learning_rate':['constant', 'adaptive'],
    'activation':['tanh', 'logistic'],
    'solver':['adam', 'sgd']
}

grid_dtc = {
    'criterion':['gini', 'entropy'],
    'min_samples_split':[2,4,6,8,10],
    'splitter':['best', 'random']
}

KNN = knn()
MLP = mlp(max_iter=2000)
DTC = dtc()

clf_knn = GridSearchCV(KNN, grid_knn,cv=7)
clf_mlp = GridSearchCV(MLP, grid_mlp,cv=7)
clf_dtc = GridSearchCV(DTC, grid_dtc,cv=7)

clf_knn.fit(X_train, y_train)
clf_mlp.fit(X_train, y_train)
clf_dtc.fit(X_train, y_train)

# Encontrar os melhores hiperparâmetros
print("Melhores hiperparâmetros KNN:", clf_knn.best_params_)
print("Melhores hiperparâmetros MLP:", clf_mlp.best_params_)
print("Melhores hiperparâmetros DTC:", clf_dtc.best_params_)



