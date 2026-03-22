# Inteligência Educacional: Preditor de Risco de Defasagem

Este projeto utiliza Ciência de Dados e Machine Learning para identificar precocemente alunos em risco de defasagem acadêmica na **Associação Passos Mágicos**. Através da análise de indicadores psicopedagógicos e socioemocionais, o modelo auxilia a equipe pedagógica na tomada de decisão preventiva.

---

## 🚀 Acesse o Dashboard Online
O projeto está publicado e pode ser acessado pelo link abaixo:
👉 **https://modelopaappsmagicos-ne3dtc3tckjbss2sckkd2a.streamlit.app/**

---

## 📊 Estrutura do Projeto
O repositório está organizado da seguinte forma:
* **`data/processed/`**: Base de dados consolidada e limpa utilizada na análise.
* **`models/`**: Modelos preditivos treinados (`Random Forest`) e lista de atributos (features).
* **`notebook/`**: Notebooks com a Análise Exploratória (EDA), Limpeza e Treinamento do Modelo.
* **`src/app.py`**: Código fonte da interface interativa em Streamlit.

## 🧠 O Modelo de IA
O preditor foi treinado utilizando o algoritmo **Random Forest**, alcançando alta precisão na identificação de padrões que levam à queda de desempenho.
* **Principais Variáveis (Features):** Engajamento (IEG), Social (IPS), Psicopedagógico (IPP), Autoavaliação (IAA) e Ponto de Virada (IPV).
* **Objetivo:** Gerar uma probabilidade de 0 a 100% de o aluno apresentar defasagem no próximo ciclo.

## 🛠️ Tecnologias Utilizadas
* **Python 3.12.7**
* **Streamlit**: Para a criação do dashboard interativo.
* **Pandas & Numpy**: Manipulação e tratamento de dados.
* **Scikit-Learn**: Desenvolvimento do modelo de Machine Learning.
* **Plotly**: Gráficos dinâmicos e interativos.
