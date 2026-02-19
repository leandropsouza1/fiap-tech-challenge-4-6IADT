# FIAP Tech Challenge – Fase 4 (6IADT)

## Introdução

Este repositório contém o projeto desenvolvido para o Tech Challenge da Fase 4 da pós-graduação em Inteligência Artificial da FIAP (6IADT). O objetivo do trabalho é aplicar, de forma prática, técnicas de Visão Computacional e Inteligência Artificial para análise automatizada de vídeos, demonstrando a integração entre processamento de mídia, detecção facial, análise emocional e análise comportamental.

O projeto foi concebido como uma solução acadêmica experimental, com foco na construção de um pipeline de IA capaz de processar vídeos frame a frame, extrair informações relevantes sobre rostos, emoções e atividades humanas, e gerar relatórios analíticos estruturados. Ele evidencia a aplicação prática de conceitos fundamentais de IA, como inferência em visão computacional, processamento multimodal e análise comportamental automatizada.

Do ponto de vista técnico, o sistema utiliza bibliotecas consolidadas do ecossistema Python para visão computacional e análise de emoções, além de um modelo ONNX para detecção facial, garantindo reprodutibilidade e execução local offline.

## Objetivo do Projeto

O objetivo principal do projeto é desenvolver um pipeline de Inteligência Artificial capaz de:

- Processar vídeos automaticamente
- Detectar rostos em tempo real
- Analisar emoções faciais
- Identificar padrões comportamentais e movimentação
- Gerar saídas analíticas estruturadas (relatórios, eventos e vídeo anotado)

Este projeto atende aos requisitos acadêmicos da FIAP ao demonstrar a aplicação prática de IA em um cenário real de análise de comportamento humano em vídeo.

## O que o Projeto Faz

O sistema implementa um pipeline completo de análise de vídeo utilizando Visão Computacional e Inteligência Artificial. Ele recebe um vídeo como entrada e executa múltiplas etapas de processamento automatizado.

### 1. Processamento de Vídeo Frame a Frame

O vídeo de entrada é carregado e decomposto em frames individuais, permitindo análises detalhadas ao longo do tempo.

### 2. Detecção Facial com Modelo YuNet (ONNX)

O projeto utiliza um modelo ONNX de detecção facial (YuNet) para identificar rostos presentes em cada frame do vídeo.

### 3. Análise de Emoções

Após a detecção facial, o sistema aplica análise emocional utilizando bibliotecas especializadas, permitindo identificar emoções como felicidade, tristeza, raiva, surpresa e neutro.

### 4. Análise de Movimento e Atividades

O pipeline também realiza análise comportamental baseada em movimentação e mudanças entre frames, possibilitando identificar atividades e eventos relevantes no vídeo.

### 5. Geração de Resultados Analíticos

Ao final do processamento, o sistema gera:

- Vídeo anotado com detecções e emoções
- Relatórios estruturados
- Eventos frame a frame em CSV
- Dados analíticos em JSON

## Estrutura do Repositório

```
fiap-tech-challenge-4-6IADT/
│
├── assets/
│   ├── Unlocking_Facial_Recognition_ Diverse_Activities_Analysis.mp4
│   └── face_detection_yunet_2023mar.onnx
│
├── outputs/
│   └── (arquivos gerados automaticamente)
│
├── src/
│   └── main.py
│
├── requirements.txt
└── README.md
```

## Pré-requisitos

- Python 3.9 a 3.11 (recomendado)
- pip
- Git
- RAM recomendada: 8GB ou superior

## Instalação (Passo a Passo)

### 1. Clonar o repositório

```bash
git clone https://github.com/ackeley/fiap-tech-challenge-4-6IADT.git
cd fiap-tech-challenge-4-6IADT
```

Ou via ZIP:

```bash
unzip fiap-tech-challenge-4-6IADT-main.zip
cd fiap-tech-challenge-4-6IADT-main
```

### 2. Criar ambiente virtual (recomendado)

#### Windows (PowerShell)

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Linux / MacOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

## Execução do Projeto

O entrypoint correto do sistema é:

```
src/main.py
```

Para executar:

```bash
python src/main.py
```

Durante a execução o sistema:

1. Carrega o vídeo em assets/
2. Carrega o modelo YuNet (.onnx)
3. Processa o vídeo frame a frame
4. Detecta faces e emoções
5. Gera relatórios e saídas automáticas

## Configuração do Vídeo de Entrada

Por padrão:

```
assets/Unlocking_Facial_Recognition_ Diverse_Activities_Analysis.mp4
```

Para usar outro vídeo:

1. Coloque o vídeo na pasta `assets/`
2. Edite no `src/main.py`:

```python
input_path = "assets/seu_video.mp4"
```

## Arquivos de Saída

Após a execução, a pasta `outputs/` será criada automaticamente com:

| Arquivo       | Descrição                     |
| ------------- | ----------------------------- |
| annotated.mp4 | Vídeo com detecções e emoções |
| report.txt    | Relatório textual da análise  |
| report.json   | Relatório estruturado em JSON |
| events.csv    | Eventos frame a frame         |

## Arquivos Obrigatórios

O projeto exige obrigatoriamente:

```
assets/face_detection_yunet_2023mar.onnx
assets/Unlocking_Facial_Recognition_ Diverse_Activities_Analysis.mp4
```

Sem esses arquivos o sistema não executará corretamente.

## Troubleshooting

### Erro de vídeo não encontrado

Verifique se o vídeo está dentro da pasta `assets/` e com o nome correto.

### Erro de modelo ONNX não encontrado

Confirme a existência do arquivo:

```
assets/face_detection_yunet_2023mar.onnx
```

### Primeira execução lenta

Comportamento esperado devido ao carregamento inicial dos modelos de IA (DeepFace/FER).
