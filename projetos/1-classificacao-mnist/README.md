# Projeto 1 — Classificação MNIST

## 💻 O Desafio Técnico

Desenvolva um **modelo de Visão Computacional** capaz de **classificar dígitos manuscritos (0-9)**, e posteriormente **otimize-o para execução em dispositivos Edge**.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**treinamento → validação → salvamento → conversão → otimização**

## 🎯 Conjunto de Dados

Dataset **MNIST**, disponível diretamente via `tf.keras.datasets.mnist` (não é necessário download manual).

## ✅ Requisitos Obrigatórios

### Etapa 1 — Treinamento do Modelo (`train_model.py`)

Implemente:

- Carregamento do dataset MNIST via TensorFlow
- **Split explícito treino/validação** (ex: `validation_split` ou um split manual)
- Construção de uma CNN com:
  - **3 a 4 blocos convolucionais** (`Conv2D` + `BatchNormalization` + `MaxPooling2D`)
  - Camada de `Dropout` antes da saída, para regularização
- Treinamento com **early stopping** baseado na perda de validação (`EarlyStopping`)
- Exibição da **acurácia de validação final** no terminal
- Salvamento do modelo treinado em formato Keras (`model.h5`)

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.h5` treinado
- Conversão para **TensorFlow Lite** (`model.tflite`)
- Aplicação de uma técnica de otimização (ex: **Dynamic Range Quantization**)

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não o `model.h5`) usando `tf.lite.Interpreter`
- Execução de inferência em pelo menos **5 amostras** do conjunto de teste
- Exibição no terminal, para cada amostra, da classe **predita** vs. a classe **real**

> 💡 Essa etapa existe porque uma métrica agregada (accuracy) pode esconder problemas que só aparecem olhando exemplos individuais. Também é o teste mais próximo do uso real em produção: carregar o artefato de edge e classificar uma entrada por vez.

**Objetivo:** reduzir o tamanho do modelo, mantendo desempenho adequado para aplicações de Edge AI.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos.

```
projetos/1-classificacao-mnist/
├── train_model.py         # ✏️ Treinamento do modelo
├── optimize_model.py      # ✏️ Conversão e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.h5               # 🤖 Gerado por você — deve ser commitado
├── model.tflite           # ⚡ Gerado por você — deve ser commitado
└── README.md              # 📝 Este arquivo (também usado como relatório)
```

## ⚠️ Restrições e Considerações de Engenharia

- Entrada do modelo: imagens 28x28, 1 canal (grayscale), normalizadas em [0, 1]
- CNN simples — evite arquiteturas muito profundas
- Não utilize modelos pré-treinados
- Número de épocas limitado (ex: até 15, com early stopping)
- Treinamento apenas em CPU

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`
- **Qualidade do modelo** — acurácia de validação consistente com o esperado para o dataset
- **Edge AI** — conversão correta para `.tflite` com técnica de otimização aplicada
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo:** Hailton Gabriel de Souza Conceição

### 1️⃣ Resumo da Arquitetura do Modelo

Foi desenvolvida uma Rede Neural Convolucional (CNN) composta por blocos convolucionais (`Conv2D` + `MaxPooling2D`), seguida por uma camada de `Flatten`, `Dense` com ativação ReLU e `Dropout` para regularização e prevenção de overfitting. A camada de saída conta com 10 neurônios com ativação Softmax para classificação dos dígitos de 0 a 9. O treinamento utilizou o otimizador Adam, perda `sparse_categorical_crossentropy` e parada antecipada com `EarlyStopping`.

### 2️⃣ Bibliotecas Utilizadas

- Python 3.11+
- TensorFlow / `tf_keras`
- NumPy

### 3️⃣ Técnica de Otimização do Modelo

A otimização em `optimize_model.py` foi realizada utilizando a técnica de **Dynamic Range Quantization** via `tf.lite.TFLiteConverter`. Essa técnica reduz a precisão dos pesos do modelo para inteiros de 8 bits (int8) durante o armazenamento, mantendo a inferência rápida e diminuindo drasticamente o consumo de memória em dispositivos Edge.

### 4️⃣ Resultados Obtidos

- **Acurácia de Validação:** ~98%
- **Tamanho do arquivo `model.h5`:** 4.1 MB
- **Tamanho do arquivo `model.tflite`:** 346 KB (redução de aproximadamente 91% no tamanho)

### 5️⃣ Comentários Adicionais (Opcional)

Durante o desenvolvimento do pipeline, o ponto mais relevante foi garantir a compatibilidade de conversão e desserialização Keras entre versões do ambiente de desenvolvimento (Codespaces com Keras 3) e a esteira de validação automatizada em CI/CD no GitHub Actions (Keras 2 / legacy format).

### 6️⃣ Exemplo de Inferência

Saída gerada no terminal pela execução do script `run_inference.py`:

```text
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
Rodando inferencia em 5 amostras usando model.tflite:

Amostra 1: predito=7 | real=7
Amostra 2: predito=2 | real=2
Amostra 3: predito=1 | real=1
Amostra 4: predito=0 | real=0
Amostra 5: predito=4 | real=4

