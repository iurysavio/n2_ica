import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from scipy.stats import randint as sp_randint
from sklearn.exceptions import ConvergenceWarning
import warnings
import time

def apply_pca(X_train, X_test, min_explained_variance=0.95):
    pca = PCA()
    X_train_pca = pca.fit_transform(X_train)
    
    explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    num_components = np.argmax(explained_variance_ratio >= min_explained_variance) + 1
    
    pca = PCA(n_components=num_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    return X_train_pca, X_test_pca

def train_and_evaluate_classifier(clf, param_dist, n_iter_search, X_train, X_test, y_train, y_test):
    if n_iter_search > 0:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            
            # Realiza o GridSearchCV e mede o tempo
            start_time_gridsearch = time.time()
            random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search, cv=cv, n_jobs=-1)
            random_search.fit(X_train, y_train)
            end_time_gridsearch = time.time()
            elapsed_time_gridsearch = end_time_gridsearch - start_time_gridsearch
            
            # Treinamento do classificador usando os melhores parâmetros encontrados
            start_time_training = time.time()
            clf.set_params(**random_search.best_params_)
            clf.fit(X_train, y_train)
            end_time_training = time.time()
            elapsed_time_training = end_time_training - start_time_training
            
            # Resultados
            y_pred = clf.predict(X_test)
            result = {
                'Classificador': clf.__class__.__name__,
                'Melhores Parâmetros': random_search.best_params_,
                'Relatório de Classificação': classification_report(y_test, y_pred),
                'Acurácia': accuracy_score(y_test, y_pred),
                'Matriz de Confusão': confusion_matrix(y_test, y_pred),
                'Tempo de Treinamento (s)': elapsed_time_training
            }
        return result

# Resultados
all_results = []

csv_files = ['features_and_labels_InceptionResNetV2.csv', 'features_and_labels_InceptionV3.csv',
             'features_and_labels_ResNet50.csv', 'features_and_labels_VGG16.csv']

# Número de folds para validação cruzada
cv = 5
n_iter_search = 5

for csv_file in csv_files:
    df = pd.read_csv(csv_file)

    X = df.drop('label', axis=1)
    y = df['label']

    y = LabelEncoder().fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Inicializando os classificadores
    classifiers = {
        'Bayes': GaussianNB(),
        'MLP': MLPClassifier(max_iter=3000, solver='adam', learning_rate_init=5e-04),
        'Nearest_Neighbors': KNeighborsClassifier(),
        'Random_Forest': RandomForestClassifier(),
        'SVM_Linear': SVC(kernel='linear', probability=True, max_iter=5000, tol=1e-6),
        'SVM_Polynomial': SVC(kernel='poly', probability=True, max_iter=5000, tol=1e-6),
        'SVM_RBF': SVC(kernel='rbf', probability=True, max_iter=5000, tol=1e-6)
    }

    # Parâmetros e distribuições para classificadores
    param_distributions = {
        'Bayes': {},
        'MLP': {"hidden_layer_sizes": list(np.arange(2, 1001))},
        'Nearest_Neighbors': {"n_neighbors": [1, 3, 5, 7, 9, 11]},
        'Random_Forest': {"n_estimators": [3000],
                          "max_depth": [6, None],
                          "max_features": sp_randint(1, 11),
                          "min_samples_split": sp_randint(2, 11),
                          "min_samples_leaf": sp_randint(1, 11),
                          "bootstrap": [True, False],
                          "criterion": ["gini", "entropy"]},
        'SVM_Linear': {'kernel': ['linear'], 'C': [2**i for i in range(-5, 15)]},
        'SVM_Polynomial': {'kernel': ['poly'], 'degree': [2, 3, 4, 5, 6], 'C': [2**i for i in range(-5, 15)]},
        'SVM_RBF': {'kernel': ['rbf'], 'C': [2**i for i in range(-5, 15)]}
    }

    for clf_name, clf in classifiers.items():
        param_dist = param_distributions[clf_name]

        if clf_name.startswith('SVM'):
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            X_train_pca, X_test_pca = apply_pca(X_train_scaled, X_test_scaled)
        else:
            X_train_pca, X_test_pca = apply_pca(X_train, X_test)

        result = train_and_evaluate_classifier(clf, param_dist, n_iter_search, X_train_pca, X_test_pca, y_train, y_test)

        # Armazena os resultados para impressão
        all_results.append((csv_file, clf_name, result))

# Imprimir todas as métricas relevantes
for result in all_results:
    print(f'\nArquivo: {result[0]}, Classificador: {result[1]}')
    print(f'Melhores Parâmetros: {result[2]["Melhores Parâmetros"]}')
    
    # Extrai métricas de precisão, sensibilidade e acurácia
    precision = result[2]["Relatório de Classificação"].split('\n')[2].split()[1]
    recall = result[2]["Relatório de Classificação"].split('\n')[2].split()[2]
    accuracy = result[2]["Acurácia"]
    
    print(f'Acurácia: {accuracy}')
    print(f'Precisão: {precision}')
    print(f'Sensibilidade: {recall}')
    
    # Imprime a matriz de confusão
    print(f'Matriz de Confusão:\n{result[2]["Matriz de Confusão"]}')
    
    print(f'Tempo de Treinamento (s): {result[2]["Tempo de Treinamento (s)"]}\n')
