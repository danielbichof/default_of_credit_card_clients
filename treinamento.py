# treinamento de classificador de risco de default

import gzip
from pickle import dump
from pprint import pprint

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from credit_data import (
    MODEL_PATH,
    carregar_dados,
    preparar_atributos,
    separar_atributos_classes,
)


dados = carregar_dados()

dados_atributos, dados_classes, coluna_classe = separar_atributos_classes(dados)
dados_atributos = preparar_atributos(dados_atributos)

print("Coluna classe:", coluna_classe)
print("Total de instancias:", dados.shape[0])
print("Total de atributos apos preparacao:", dados_atributos.shape[1])
print("Frequencia das classes:")
print(dados_classes.value_counts().sort_index())

# separar treino e teste antes do balanceamento
atributos_train, atributos_test, classes_train, classes_test = train_test_split(
    dados_atributos,
    dados_classes,
    test_size=0.3,
    stratify=dados_classes,
    random_state=42,
)

# balancear somente os dados de treino
balancer = SMOTE(random_state=42)
atributos_train, classes_train = balancer.fit_resample(atributos_train, classes_train)

print("Frequencia das classes balanceadas no treino:")
print(classes_train.value_counts().sort_index())

# hiperparametrizacao do Random Forest
n_estimators = [int(x) for x in np.linspace(start=50, stop=250, num=5)]
criterion = ["gini", "entropy", "log_loss"]
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
max_depth = [None] + [int(x) for x in np.linspace(start=10, stop=80, num=8)]
max_features = ["sqrt", "log2"]

rf_grid = {
    "n_estimators": n_estimators,
    "criterion": criterion,
    "min_samples_split": min_samples_split,
    "min_samples_leaf": min_samples_leaf,
    "max_depth": max_depth,
    "max_features": max_features,
}

rf = RandomForestClassifier(random_state=42)
rf_hiperparameters = RandomizedSearchCV(
    estimator=rf,
    param_distributions=rf_grid,
    n_iter=12,
    scoring="f1_macro",
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1,
)

rf_hiperparameters.fit(atributos_train, classes_train)

print("Melhores hiperparametros:")
pprint(rf_hiperparameters.best_params_)

# selecionar o melhor modelo encontrado
default_rf = rf_hiperparameters.best_estimator_
classes_preditas = default_rf.predict(atributos_test)

print("Matriz de confusao:")
print(confusion_matrix(classes_test, classes_preditas))

print("Acuracia:", accuracy_score(classes_test, classes_preditas))
print("Precisao macro:", precision_score(classes_test, classes_preditas, average="macro"))
print("Recall macro:", recall_score(classes_test, classes_preditas, average="macro"))
print("F1 macro:", f1_score(classes_test, classes_preditas, average="macro"))

print("Relatorio de classificacao:")
print(classification_report(classes_test, classes_preditas))

artefato = {
    "modelo": default_rf,
    "coluna_classe": coluna_classe,
    "colunas_treinamento": list(dados_atributos.columns),
    "classes": list(default_rf.classes_),
}

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
dump(artefato, gzip.open(MODEL_PATH, "wb"))

print("Modelo salvo em:", MODEL_PATH)
