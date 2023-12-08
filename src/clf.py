import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report

df = pd.read_csv('features_and_labels.csv')

X = df.drop('label', axis=1)
y = df['label']

y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

pca = PCA()
X_train_pca = pca.fit_transform(X_train)

#Variancia explicada cumulativa
explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

#Definindo a variancia minima desejada
min_explained_variance = 0.95

#Numero de componentes para atingir a variancia minima
num_components = np.argmax(explained_variance_ratio >= min_explained_variance) + 1

print(f'Número de componentes para {min_explained_variance * 100}% de variância explicada: {num_components}')

pca = PCA(n_components=num_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

#FAZER GridSearchCV

#Classificadorers
rf_classifier = RandomForestClassifier(random_state=42)
mlp_classifier = MLPClassifier(random_state=42)
nb_classifier = GaussianNB()
svm_classifier = SVC(kernel='linear', random_state=42)
svm_clf_poly = SVC(kernel='poly', random_state=42)
svm_clf_rbf = SVC(kernel='rbf', random_state=42)

rf_classifier.fit(X_train_pca, y_train)
mlp_classifier.fit(X_train_pca, y_train)
nb_classifier.fit(X_train_pca, y_train)
svm_classifier.fit(X_train_pca, y_train)
svm_clf_poly.fit(X_train_pca, y_train)
svm_clf_rbf.fit(X_train_pca, y_train)

y_pred_rf = rf_classifier.predict(X_test_pca)
y_pred_mlp = mlp_classifier.predict(X_test_pca)
y_pred_nb = nb_classifier.predict(X_test_pca)
y_pred_svm = svm_classifier.predict(X_test_pca)
y_pred_svmpoly = svm_clf_poly.predict(X_test_pca)
y_pred_svmrbf = svm_clf_rbf.predict(X_test_pca)

print("Random Forest - Relatório de Classificação:")
print(classification_report(y_test, y_pred_rf))

print("MLP - Relatório de Classificação:")
print(classification_report(y_test, y_pred_mlp))

print("Naive Bayes - Relatório de Classificação:")
print(classification_report(y_test, y_pred_nb))

print("SVM Linear - Relatório de Classificação:")
print(classification_report(y_test, y_pred_svm))

print("SVM Polinomial - Relatório de Classificação:")
print(classification_report(y_test, y_pred_svmpoly))

print("SVM RBF - Relatório de Classificação:")
print(classification_report(y_test, y_pred_svmrbf))
