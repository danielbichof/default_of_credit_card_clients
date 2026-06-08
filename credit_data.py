from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "default_of_credit_card_clients.csv"
MODEL_PATH = BASE_DIR / "models" / "default_random_forest.pkl"


def carregar_dados():
    return pd.read_csv(DATASET_PATH, sep=";")


def encontrar_coluna_classe(dados):
    colunas_default = [
        coluna
        for coluna in dados.columns
        if "default" in coluna.lower() and dados[coluna].nunique() <= 2
    ]

    if colunas_default:
        return colunas_default[0]

    return dados.columns[-1]


def limpar_categorias(dados):
    dados = dados.copy()
    colunas_categoricas = dados.select_dtypes(include=["object", "string"]).columns

    for coluna in colunas_categoricas:
        dados[coluna] = dados[coluna].astype(str).str.strip()

    return dados


def separar_atributos_classes(dados):
    dados = limpar_categorias(dados)
    coluna_classe = encontrar_coluna_classe(dados)

    dados_atributos = dados.drop(columns=[coluna_classe])
    if "ID" in dados_atributos.columns:
        dados_atributos = dados_atributos.drop(columns=["ID"])

    dados_classes = dados[coluna_classe]
    return dados_atributos, dados_classes, coluna_classe


def preparar_atributos(dados_atributos):
    return pd.get_dummies(dados_atributos, dtype=int)


def preparar_nova_instancia(nova_instancia, colunas_treinamento):
    nova_instancia = limpar_categorias(nova_instancia)

    if "ID" in nova_instancia.columns:
        nova_instancia = nova_instancia.drop(columns=["ID"])

    nova_instancia = preparar_atributos(nova_instancia)
    return nova_instancia.reindex(columns=colunas_treinamento, fill_value=0)
