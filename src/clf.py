import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
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
        # Ignorar avisos de convergência
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search, cv=cv, n_jobs=-1)
            random_search.fit(X_train, y_train)

        # resultados
        y_pred = random_search.predict(X_test)
        result = {
            'Classificador': clf.__class__.__name__,
            'Melhores Parâmetros': random_search.best_params_,
            'Melhor Score': random_search.best_score_,
            'Relatório de Classificação': classification_report(y_test, y_pred),
            'Acurácia': accuracy_score(y_test, y_pred),
            'Matriz de Confusão': confusion_matrix(y_test, y_pred)
        }

        return result

# Melhores resultados
best_results = {}

csv_files = ['features_and_labels_InceptionResNetV2.csv', 'features_and_labels_InceptionV3.csv',
             'features_and_labels_ResNet50.csv', 'features_and_labels_VGG16.csv']

# numero de folds para validação cruzada
cv = 5
n_iter_search = 5

for csv_file in csv_files:
    df = pd.read_csv(csv_file)

    X = df.drop('label', axis=1)
    y = df['label']

    y = LabelEncoder().fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # inicializando os classificadores
    classifiers = {
        'Bayes': GaussianNB(),
        'MLP': MLPClassifier(max_iter=3000, solver='adam', learning_rate_init=5e-04),
        'Nearest_Neighbors': KNeighborsClassifier(),
        'Random_Forest': RandomForestClassifier(),
        'SVM_Linear': SVC(kernel='linear', probability=True, max_iter=5000, tol=1e-6),
        'SVM_Polynomial': SVC(kernel='poly', probability=True, max_iter=5000, tol=1e-6),
        'SVM_RBF': SVC(kernel='rbf', probability=True, max_iter=5000, tol=1e-6)
    }

    # parametros e distribuições para classificadores
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

        # Armazenar os resultados para impressão
        best_results[(csv_file, clf_name)] = result

# Imprimir os melhores resultados
for key, result in sorted(best_results.items(), key=lambda x: x[0][0]):
    print(f'\nArquivo: {key[0]}, Classificador: {key[1]}')
    print(f'Melhores Parâmetros: {result["Melhores Parâmetros"]}')
    print(f'Melhor Score: {result["Melhor Score"]}\n')
    print(f'Relatório de Classificação:\n{result["Relatório de Classificação"]}')
    print(f'Acurácia: {result["Acurácia"]}')
    print(f'Matriz de Confusão:\n{result["Matriz de Confusão"]}\n')