import gzip
import argparse
import json
from pickle import load

import pandas as pd

from credit_data import MODEL_PATH, preparar_nova_instancia


DEFAULT_INSTANCE = {
    "LIMIT_BAL": 50000,
    "SEX": "F",
    "EDUCATION": "Middle School",
    "MARRIAGE": "Married",
    "AGE": 35,
    "PAY_0": 0,
    "PAY_2": 0,
    "PAY_3": 0,
    "PAY_4": 0,
    "PAY_5": 0,
    "PAY_6": 0,
    "BILL_AMT1": 10000,
    "BILL_AMT2": 9000,
    "BILL_AMT3": 8000,
    "BILL_AMT4": 7000,
    "BILL_AMT5": 6000,
    "BILL_AMT6": 5000,
    "PAY_AMT1": 1000,
    "PAY_AMT2": 1000,
    "PAY_AMT3": 1000,
    "PAY_AMT4": 1000,
    "PAY_AMT5": 1000,
    "PAY_AMT6": 1000,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Inferencia de risco de default.")
    parser.add_argument(
        "--instance",
        help="Objeto JSON com os atributos do cliente.",
    )
    return parser.parse_args()


def carregar_instancia(args):
    if args.instance:
        return json.loads(args.instance)

    return DEFAULT_INSTANCE


args = parse_args()

# Abrir modelo
artefato = load(gzip.open(MODEL_PATH, "rb"))
default_model = artefato["modelo"]

# Coletar os dados da nova instancia
nova_instancia = pd.DataFrame([carregar_instancia(args)])
nova_instancia = preparar_nova_instancia(
    nova_instancia,
    artefato["colunas_treinamento"],
)

classe_predita = default_model.predict(nova_instancia)[0]
probabilidades = default_model.predict_proba(nova_instancia)[0]

print("Classe predita:", classe_predita)
print("Distribuicao de probabilidade:")

score_default = None
for classe, probabilidade in zip(artefato["classes"], probabilidades):
    print(f"Classe {classe}: {probabilidade:.4f}")
    if classe == 1:
        score_default = probabilidade

print("Score de risco de default:", f"{score_default:.4f}")
