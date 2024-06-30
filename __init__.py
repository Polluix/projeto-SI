import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, confusion_matrix,ConfusionMatrixDisplay,classification_report
import numpy as np
import matplotlib.pyplot as plt
import json

num_bases=['03','25']

# -----------------------------IMPLEMENTAÇÃO DOS MODELOS -------------------------------------
# for num_base in num_bases:
#     # -----------------SELECAO DE FEATURES--------------------
#     # leitura da base
#     csv_name = '0-Todos-CSV/'+num_base+'.csv'

#     base = pd.read_csv(csv_name)

#     # numero de objetos de cada classe
#     # print(base['0'].value_counts())

#     #separação das features e classes
#     X =base.drop(columns=['0'])
#     y = base.pop('0')

#     # separação dos conjuntos de treino e teste
#     X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42,
#                                                         stratify=y)

#     features = X_train.columns.to_list()

#     # remove valores nulos
#     X_train = X_train.dropna()
#     y_train = y_train.dropna()

#     X_train = X_train.values
#     y_train=y_train.values

#     n_estimator_list = [2,3,4,5,6,10,20]
#     X_train_new = []
#     models = []
#     feature_importances = []
#     for n in n_estimator_list:
#         clf = ExtraTreesClassifier(n_estimators=n)
#         clf = clf.fit(X_train, y_train)
        
#         if n==2:
#             feature_importances.append(clf.feature_importances_)
        

#         model = SelectFromModel(clf, prefit=True)
#         X_train_new.append(model.transform(X_train))
#         models.append(model)

#     print(f'Formato das bases otimizadas (base {num_base}):\n')
#     for base in X_train_new:
#         print(base.shape)

#     # salvamento das bases de treino e teste após seleção de features
#     index_least_dimensions = 0

#     selected_features = models[index_least_dimensions].get_support()
#     indexes = np.where(selected_features==True)[0]

#     xticks = np.arange(1,len(feature_importances[0][indexes])+1,1)

#     feature_importances = np.array(feature_importances)

#     print(f'Importância de todas features otimizadas somada (base {num_base}): {sum(feature_importances[0])}')

#     plt.figure(figsize=(12,6))
#     plt.bar(xticks,feature_importances[0][indexes])

#     for i, data in zip(xticks, feature_importances[0][indexes]):
#         plt.text(x=i-0.45, y=data+0.005, s=f'{data:.3f}')

#     plt.xticks(xticks)
#     plt.yticks([])
#     plt.savefig(f'Feature_importances_base_{num_base}.png')
#     # plt.show()

#     aux = []
#     for i in range(len(features)):
#         if selected_features[i]:
#             aux.append(features[i])
#     aux = np.array(aux)

#     df = pd.DataFrame(X_train_new[index_least_dimensions],columns=aux)
#     df['CLASSE'] = y_train
#     df.to_csv('./src/base_treino_preprocessada_'+num_base+'.csv',index=False)

#     df2 = X_test
#     df2['CLASSE'] = y_test
#     df2.to_csv('./src/base_teste_'+num_base+'.csv',index=False)

#     # ----------------------------------------------------------------

#     # ----------------DISTRIBUIÇÃO DE CLASSES--------------------------
#     base_treino = pd.read_csv('./src/base_treino_preprocessada_'+num_base+'.csv')

#     # adequação da base de treino às features presentes na base de teste
#     base_teste = pd.read_csv('./src/base_teste_'+ num_base+ '.csv',usecols=base_treino.columns.to_list())

#     # printa valores nulos na base de teste (se existir)
#     for i in base_teste.isna().sum():
#         if i!=0:
#             print(i)

#     # -----------PLOT DISTRIBUIÇÃO DE CLASSES----------------------------------------
#     elements_per_class = base_treino['CLASSE'].value_counts().sort_index()

#     classes = np.arange(elements_per_class.index.min(),elements_per_class.index.max()+1,1)

#     plt.figure(figsize=(12,6))
#     plt.bar(classes, elements_per_class)

#     for i, data in enumerate(elements_per_class):
#         plt.text(x=i+1, y=data+0.5, s=f'{data}')

#     plt.yticks([])
#     plt.xticks(classes)
#     plt.savefig(f'Elementos_por_classe_base_{num_base}.png')
#     # plt.show()
#     # ----------------------------------------------------------------------------

#     assert len(base_treino.columns) == len(base_teste.columns), 'A base de treino possui features diferentes da base de teste.'


#     # ----------------------GRID SEARCH--------------------
#     # separação das bases de treino e teste para aplicação nos modelos
#     X_train = base_treino.drop(columns=['CLASSE'])
#     y_train = base_treino.pop('CLASSE')

#     X_test = base_teste.drop(columns=['CLASSE'])
#     y_test = base_teste.pop('CLASSE')

#     # definição dos hiperparâmetros para o gridsearch
#     grid_knn = {
#         'metric':['euclidean', 'manhattan'],
#         'n_neighbors':[1,3,5,7,9,11],
#         'weights':['uniform','distance'],
#     }

#     grid_mlp = {
#         'hidden_layer_sizes':[(100,100,100,100),(100,100,50),(100,50,50),(50,50,50)],
#         'learning_rate':['constant', 'adaptive'],
#         'activation':['tanh', 'logistic'],
#         'solver':['adam', 'sgd']
#     }

#     grid_dtc = {
#         'criterion':['gini', 'entropy'],
#         'min_samples_split':[2,5,10,20,50,100],
#         'splitter':['best', 'random'],
#         'max_features':['sqrt', 'log2']
#     }

#     KNN = knn()
#     MLP = mlp(max_iter=3000)
#     DTC = dtc()

#     clf_knn = GridSearchCV(KNN, grid_knn,cv=7)
#     clf_mlp = GridSearchCV(MLP, grid_mlp,cv=7)
#     clf_dtc = GridSearchCV(DTC, grid_dtc,cv=7)

#     clf_knn.fit(X_train, y_train)
#     clf_mlp.fit(X_train, y_train)
#     clf_dtc.fit(X_train, y_train)

#     print(F'-----------------------BASE {num_base}--------------------------------')
#     print('Melhores hiperparâmetros KNN:', clf_knn.best_params_)
#     print('Melhores hiperparâmetros MLP:', clf_mlp.best_params_)
#     print('Melhores hiperparâmetros DTC:', clf_dtc.best_params_)

#     # faz a classificação dos dados de teste
#     y_knn = clf_knn.best_estimator_.predict(X_test)
#     y_mlp = clf_mlp.best_estimator_.predict(X_test)
#     y_dtc = clf_dtc.best_estimator_.predict(X_test)

#     # Métricas de desempenho
#     accuracy_knn = accuracy_score(y_test, y_knn)
#     accuracy_mlp = accuracy_score(y_test, y_mlp)
#     accuracy_dtc = accuracy_score(y_test, y_dtc)

#     print('Acurácia KNN:', accuracy_knn)
#     print('Acurácia MLP:', accuracy_mlp)
#     print('Acurácia DTC:', accuracy_dtc)

#     confusion_matrix_knn = confusion_matrix(y_test, y_knn)
#     confusion_matrix_mlp = confusion_matrix(y_test, y_mlp)
#     confusion_matrix_dtc = confusion_matrix(y_test, y_dtc)

#     plt.figure()
#     disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_knn,
#                               display_labels=clf.classes_)
#     disp.plot()
#     plt.savefig(f'matriz_confusao_knn_base_{num_base}.png')
#     # plt.show()

#     plt.figure()
#     disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_mlp,
#                               display_labels=clf.classes_)
#     disp.plot()
#     plt.savefig(f'matriz_confusao_mlp_base_{num_base}.png')
#     # plt.show()

#     plt.figure()
#     disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_dtc,
#                               display_labels=clf.classes_)
#     disp.plot()
#     plt.savefig(f'matriz_confusao_dtc_base_{num_base}.png')
#     # plt.show()

#     report_knn = classification_report(y_test, y_knn)
#     report_mlp = classification_report(y_test, y_mlp)
#     report_dtc = classification_report(y_test, y_dtc)

#     with open(f'report_knn_base{num_base}.json', 'w') as rknn:
#         json.dump(report_knn,rknn)
    
#     with open(f'report_mlp_base{num_base}.json', 'w') as rmlp:
#         json.dump(report_mlp,rmlp)

#     with open(f'report_dtc_base{num_base}.json', 'w') as rdtc:
#         json.dump(report_dtc,rdtc)


#     print(F'-----------------------FIM BASE {num_base}--------------------------------')

# --------------------------------IMPLEMENTAÇÃO BASELINES----------------------------------------
print('começou aqui')
for base in num_bases:
    csv_name = '0-Todos-CSV/'+base+'.csv'

    df= pd.read_csv(csv_name)

    #separação das features e classes
    X =df.drop(columns=['0'])
    y = df.pop('0')

    # separação dos conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42,
                                                        stratify=y)
        
    grid_knn = {
        'metric':['euclidean', 'manhattan'],
        'n_neighbors':[1,3,5,7,9,11],
        'weights':['uniform','distance'],
    }  

    grid_mlp = {
        'hidden_layer_sizes':[(100,100,100,100),(100,100,50),(100,50,50),(50,50,50)],
        'learning_rate':['constant', 'adaptive'],
        'activation':['tanh', 'logistic'],
        'solver':['adam', 'sgd']
    }

    grid_dtc = {
        'criterion':['gini', 'entropy'],
        'min_samples_split':[2,5,10,20,50,100],
        'splitter':['best', 'random'],
        'max_features':['sqrt', 'log2']
    }

    KNN = knn()
    MLP = mlp(max_iter=3000)
    DTC = dtc()

    clf_knn = GridSearchCV(KNN, grid_knn,cv=7)
    clf_mlp = GridSearchCV(MLP, grid_mlp,cv=7)
    clf_dtc = GridSearchCV(DTC, grid_dtc,cv=7)

    clf_knn.fit(X_train, y_train)
    clf_mlp.fit(X_train, y_train)
    clf_dtc.fit(X_train, y_train)

    KNN = knn()

    clf_knn = GridSearchCV(KNN, grid_knn,cv=7)
    clf_mlp = GridSearchCV(MLP, grid_mlp,cv=7)
    clf_dtc = GridSearchCV(DTC, grid_dtc,cv=7)

    clf_knn.fit(X_train, y_train)
    clf_mlp.fit(X_train, y_train)
    clf_dtc.fit(X_train, y_train)

    print(F'-----------------------BASE {base}--------------------------------')
    print(f'Melhores hiperparâmetros baseline KNN base {base}:', clf_knn.best_params_)
    print(f'Melhores hiperparâmetros baseline MLP base {base}:', clf_mlp.best_params_)
    print(f'Melhores hiperparâmetros baseline DTC base {base}:', clf_dtc.best_params_)

    # faz a classificação dos dados de teste
    y_knn = clf_knn.best_estimator_.predict(X_test)
    y_mlp = clf_mlp.best_estimator_.predict(X_test)
    y_dtc = clf_dtc.best_estimator_.predict(X_test)

    # Métricas de desempenho
    accuracy_knn = accuracy_score(y_test, y_knn)
    accuracy_mlp = accuracy_score(y_test, y_mlp)
    accuracy_dtc = accuracy_score(y_test, y_dtc)

    print('Acurácia baseline KNN:', accuracy_knn)
    print('Acurácia baseline MLP:', accuracy_mlp)
    print('Acurácia baseline DTC:', accuracy_dtc)

