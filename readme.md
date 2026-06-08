# Sistema Inteligente de Risco de Default

Este projeto implementa um sistema inteligente para o Banco AgoraVai determinar se
um cliente de cartao de credito tem risco de se tornar inadimplente, ou seja,
entrar em `default` no proximo mes.

A base utilizada e `default_of_credit_card_clients.csv`.

## Objetivo

Classificar clientes em duas classes:

- `0`: cliente sem default no proximo mes
- `1`: cliente com default no proximo mes

Alem da classe predita, o sistema informa a distribuicao de probabilidade do
modelo. O score de risco e a probabilidade associada a classe `1`.

## Estrutura

| Arquivo | Funcao |
|---|---|
| `default_of_credit_card_clients.csv` | Base de dados usada no treinamento |
| `credit_data.py` | Funcoes auxiliares para leitura, limpeza e preparacao dos dados |
| `treinamento.py` | Treina, seleciona o melhor RandomForest e salva o modelo |
| `inferencia.py` | Carrega o modelo salvo e calcula o risco de uma nova instancia |
| `models/default_random_forest.pkl` | Modelo treinado salvo com `pickle` comprimido |

## Entendimento da Base

A base possui 30.000 registros e 25 colunas.

A coluna alvo e:

```text
default payment next month
```

A coluna `ID` e apenas identificadora e nao e usada como atributo de treinamento.

Colunas categoricas:

- `SEX`
- `EDUCATION`
- `MARRIAGE`

Essas colunas sao limpas com `strip()` para remover espacos extras e depois
transformadas com `pd.get_dummies`.

Colunas numericas:

- `LIMIT_BAL`
- `AGE`
- `PAY_0`, `PAY_2`, `PAY_3`, `PAY_4`, `PAY_5`, `PAY_6`
- `BILL_AMT1` ate `BILL_AMT6`
- `PAY_AMT1` ate `PAY_AMT6`

Diferente da base de diabetes, valores zero nesta base nao foram tratados como
ausentes, pois podem representar ausencia de pagamento ou fatura zerada.

## Pipeline de Treinamento

O treinamento segue um estilo semelhante ao exemplo `fertility`, com etapas
explicitas:

1. Leitura da base.
2. Separacao entre atributos e classe.
3. Remocao da coluna `ID`.
4. Limpeza das colunas categoricas.
5. Transformacao das categorias com `get_dummies`.
6. Separacao em treino e teste com `train_test_split`.
7. Balanceamento do conjunto de treino com `SMOTE`.
8. Busca automatica de hiperparametros com `RandomizedSearchCV`.
9. Avaliacao do melhor modelo.
10. Salvamento do artefato treinado.

O modelo utilizado e:

```text
RandomForestClassifier
```

A selecao dos hiperparametros e feita por:

```python
RandomizedSearchCV(
    estimator=rf,
    param_distributions=rf_grid,
    n_iter=12,
    scoring="f1_macro",
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1,
)
```

Neste projeto, a selecao do melhor modelo ocorre dentro da familia
`RandomForestClassifier`: o `RandomizedSearchCV` treina varias configuracoes,
compara os resultados por validacao cruzada e retorna automaticamente a melhor
configuracao encontrada em `best_estimator_`. Esse estimador final e o modelo
usado na avaliacao, salvo em disco e carregado pelo modulo de inferencia.

Foram testadas 12 combinacoes de parametros. Como a validacao cruzada usa
`cv=3`, o processo realiza 36 treinamentos internos.

A metrica usada na escolha do melhor modelo e `f1_macro`, pois a base e
desbalanceada e essa metrica considera o desempenho nas duas classes.

## Resultados Obtidos

Melhores hiperparametros encontrados:

```text
criterion: log_loss
max_depth: 60
max_features: sqrt
min_samples_leaf: 1
min_samples_split: 5
n_estimators: 250
```

Metricas no conjunto de teste:

| Metrica | Valor |
|---|---:|
| Acuracia | 0.8072 |
| Precisao macro | 0.7187 |
| Recall macro | 0.6731 |
| F1 macro | 0.6894 |

Matriz de confusao:

```text
[[6404  605]
 [1130  861]]
```

## Como Executar

Entre na pasta da base:

```bash
cd default_of_credit_card_clients
```

Execute o treinamento:

```bash
../.venv/bin/python treinamento.py
```

Execute a inferencia com uma instancia padrao:

```bash
../.venv/bin/python inferencia.py
```

Exemplo de saida:

```text
Classe predita: 0
Distribuicao de probabilidade:
Classe 0: 0.6480
Classe 1: 0.3520
Score de risco de default: 0.3520
```

## Inferencia com Nova Instancia

Tambem e possivel informar os dados de um cliente em JSON:

```bash
../.venv/bin/python inferencia.py --instance '{"LIMIT_BAL": 20000, "SEX": "M", "EDUCATION": "Middle School", "MARRIAGE": "Married", "AGE": 24, "PAY_0": 2, "PAY_2": 2, "PAY_3": -1, "PAY_4": -1, "PAY_5": -2, "PAY_6": -2, "BILL_AMT1": 3913, "BILL_AMT2": 3102, "BILL_AMT3": 689, "BILL_AMT4": 0, "BILL_AMT5": 0, "BILL_AMT6": 0, "PAY_AMT1": 0, "PAY_AMT2": 689, "PAY_AMT3": 0, "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0}'
```

Saida validada:

```text
Classe predita: 1
Distribuicao de probabilidade:
Classe 0: 0.1545
Classe 1: 0.8455
Score de risco de default: 0.8455
```

## Observacoes

- O pipeline salva junto com o modelo a lista de colunas usadas no treinamento.
- Durante a inferencia, a nova instancia e reindexada para manter exatamente o
  mesmo formato usado no treino.
- Categorias nao presentes na instancia recebem valor `0` nas colunas dummy.
- O score de risco e sempre a probabilidade da classe `1`.
- O modelo e salvo com `pickle` usando compressao `gzip` para reduzir o tamanho
  do arquivo.
