import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from scipy.stats import randint as sp_randint

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
        random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search, cv=cv, n_jobs=-1)
        random_search.fit(X_train, y_train)

        # Print best parameters and score
        print(f'Classifier: {clf.__class__.__name__}')
        print(f'Best Parameters: {random_search.best_params_}')
        print(f'Best Score: {random_search.best_score_}\n')

        # Evaluate on test set
        y_pred = random_search.predict(X_test)
        print(f'Classification Report:\n{classification_report(y_test, y_pred)}')
        print(f'Accuracy Score: {accuracy_score(y_test, y_pred)}')
        print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n')

csv_files = ['features_and_labels_InceptionResNetV2.csv', 'features_and_labels_InceptionV3.csv',
             'features_and_labels_ResNet50.csv', 'features_and_labels_VGG16.csv']

for csv_file in csv_files:
    df = pd.read_csv(csv_file)

    X = df.drop('label', axis=1)
    y = df['label']

    y = LabelEncoder().fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    #Aplicando PCA
    X_train_pca, X_test_pca = apply_pca(X_train, X_test)

    #Inicializando os classificadores
    classifiers = {
        'Bayes': GaussianNB(),
        'MLP': MLPClassifier(max_iter=1000, solver='adam', learning_rate_init=5e-04),
        'Nearest_Neighbors': KNeighborsClassifier(),
        'Random_Forest': RandomForestClassifier(),
        'SVM_Linear': SVC(kernel='linear', probability=True, max_iter=3000, tol=1e-3),
        'SVM_Polynomial': SVC(kernel='poly', probability=True, max_iter=3000, tol=1e-3),
        'SVM_RBF': SVC(kernel='rbf', probability=True, max_iter=3000, tol=1e-3)
    }

    #Parâmetros e distribuições para classificadores
    param_distributions = {
        'Bayes': None,
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
       
    }