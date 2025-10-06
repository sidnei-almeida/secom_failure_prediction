# SECOM Failure Prediction - Sistema de Detecção de Anomalias

Sistema avançado de detecção de anomalias em manufatura de semicondutores utilizando Autoencoder Neural Network.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 Sobre o Projeto

O **SECOM Failure Prediction** é um sistema de detecção de anomalias desenvolvido para identificar falhas em processos de manufatura de semicondutores. Utilizando um **Autoencoder Neural Network**, o sistema aprende padrões de operação normal e detecta desvios que podem indicar potenciais falhas.

### Características Principais

- 🧠 **Autoencoder Neural Network** com arquitetura 558 → 128 → 64 → 32 (bottleneck) → 64 → 128 → 558
- 📊 **Dashboard Interativo** desenvolvido com Streamlit
- 🎯 **Dois Thresholds de Detecção**: Balanced (0.45) e Conservative (0.50)
- 📈 **Visualizações Avançadas** com Plotly para análise de dados e resultados
- 🎨 **Design Dark Premium** com paleta de cores quente (industrial/fogo)
- ⚡ **Performance Otimizada** utilizando TensorFlow CPU

### Métricas do Modelo

- **Recall (Anomalias)**: 35.6%
- **Precision (Anomalias)**: 44.6%
- **F1-Score**: 0.396
- **Accuracy Geral**: 71.5%

## 🚀 Como Executar

### Pré-requisitos

- Python 3.10 ou superior
- pip (gerenciador de pacotes Python)

### Instalação

1. **Clone o repositório**
```bash
git clone https://github.com/sidnei-almeida/secom_failure_prediction.git
cd secom_failure_prediction
```

2. **Crie e ative um ambiente virtual** (recomendado)
```bash
python -m venv venv

# No Linux/Mac:
source venv/bin/activate

# No Windows:
venv\Scripts\activate
```

3. **Instale as dependências**
```bash
pip install -r requirements.txt
```

### Executar o Aplicativo

```bash
streamlit run app.py
```

O aplicativo será aberto automaticamente no seu navegador em `http://localhost:8501`

## 📂 Estrutura do Projeto

```
secom_failure_prediction/
├── app.py                          # Aplicação Streamlit principal
├── requirements.txt                # Dependências do projeto
├── README.md                       # Este arquivo
├── data/
│   └── secom_cleaned_dataset.csv  # Dataset limpo (1567 registros, 558 features)
├── models/
│   └── secom_autoencoder_model.keras  # Modelo treinado
├── training/
│   └── secom_autoencoder_metadata.json  # Metadados do treinamento
└── notebooks/
    ├── 1_Data_Analysis_and_Manipulation.ipynb
    ├── 2_Deep_Learning_Models_Classification.ipynb
    └── 3_Anomaly_Detection.ipynb
```

## 🎯 Funcionalidades do App

### 1. **Home**
- Visão geral do projeto e métricas principais
- Distribuição de classes (Normal vs Falhas)
- Principais insights sobre o dataset e metodologia

### 2. **Análise de Dados**
- Estatísticas descritivas das features
- Visualização de distribuições
- Matriz de correlação
- Exploração interativa do dataset SECOM

### 3. **Modelo**
- Explicação detalhada da arquitetura do Autoencoder
- Visualização interativa da rede neural
- Descrição do processo de detecção de anomalias
- Especificações técnicas completas

### 4. **Treinamento**
- Histórico completo do treinamento
- Gráficos de evolução da loss (training e validation)
- Métricas de performance final
- Configurações e hiperparâmetros utilizados

### 5. **Teste**
- Upload de arquivos CSV para teste
- Seleção de threshold (Balanced ou Conservative)
- Análise em tempo real com visualizações
- Distribuição de erros de reconstrução
- Matriz de confusão (quando labels estão disponíveis)
- Download dos resultados em CSV

## 🧪 Testando o Sistema

Você pode testar o sistema usando o próprio dataset do projeto:

1. Vá para a página **Teste**
2. Faça upload do arquivo `data/secom_cleaned_dataset.csv`
3. Selecione o threshold desejado
4. Clique em **Analisar Dados**
5. Visualize os resultados e baixe o relatório

## 🛠️ Tecnologias Utilizadas

- **TensorFlow/Keras**: Framework de Deep Learning
- **Streamlit**: Framework para criação do dashboard
- **Plotly**: Biblioteca de visualização interativa
- **Pandas & NumPy**: Manipulação e análise de dados
- **Scikit-learn**: Pré-processamento e métricas

## 📊 Dataset SECOM

O dataset SECOM contém dados de sensores de um processo de fabricação de semicondutores:

- **Total de Registros**: 1567
- **Features**: 558 (após limpeza e remoção de features com >40% de valores ausentes)
- **Classes**: Binário (Normal: -1, Falha: 1)
- **Desbalanceamento**: ~93% Normal vs ~7% Falhas

## 🎓 Metodologia

1. **Pré-processamento**: Limpeza de dados, remoção de features com excesso de valores nulos, imputação pela mediana
2. **Arquitetura**: Autoencoder simétrico com bottleneck de 32 dimensões
3. **Treinamento**: Apenas com dados normais (1170 amostras)
4. **Detecção**: Erro de reconstrução (MAE) > threshold = anomalia
5. **Thresholds**: 
   - **Balanced (0.45)**: Melhor equilíbrio precision-recall
   - **Conservative (0.50)**: Menos falsos positivos

## 📝 Licença

Este projeto está sob a licença MIT.

## 👨‍💻 Autor

Desenvolvido com ❤️ para análise avançada de anomalias em processos industriais.

---

**Nota**: Este é um projeto acadêmico/profissional desenvolvido para demonstração de técnicas de Deep Learning aplicadas à detecção de anomalias em ambientes industriais.
