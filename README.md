# Laboratório 3: Implementando o Decoder de um Transformer

## Google Colab

Voce pode executar este laboratório diretamente no Google Colab através do seguinte link:

[Laboratório 3 - Implementando o Decoder no Google Colab](https://colab.research.google.com/drive/1x2XmDdA9heAaVkTkFmEiPzZlY1jUIMJY?usp=sharing)

Este repositório contém a implementação dos principais componentes do Decoder de um modelo Transformer, conforme explorado no Laboratório 3 da disciplina. O objetivo é demonstrar o funcionamento da máscara causal (Look-Ahead Mask), do mecanismo de Cross-Attention e do loop de inferência auto-regressivo.

## Conteúdo
- `laboratorio3_decoder.py`: O script Python contendo as implementações das três tarefas.
- `Relatorio_Laboratorio3.md`: Um relatório detalhado sobre as implementações e os resultados obtidos.

## Tarefas Implementadas

### 1. Máscara Causal (Look-Ahead Mask)

Implementação da máscara causal para garantir que o Decoder não tenha acesso a informações futuras durante a geração de sequências. Isso é crucial para modelos auto-regressivos, onde a previsão de um token deve depender apenas dos tokens anteriores.

**Funcionalidade:**
- Cria uma matriz de máscara onde os elementos do triângulo superior são definidos como `-infinito`, efetivamente zerando as probabilidades de atenção para tokens futuros após a aplicação da função Softmax.

### 2. Cross-Attention (Ponte Encoder-Decoder)

Implementação do mecanismo de Cross-Attention, que permite ao Decoder focar em partes relevantes da saída do Encoder. Este é o elo que conecta as representações do Encoder com o processo de decodificação.

**Funcionalidade:**
- Calcula os pesos de atenção entre as *Queries* (Q) do estado atual do Decoder e as *Keys* (K) e *Values* (V) da saída do Encoder.
- Permite que o Decoder 
acesso as informações mais relevantes do contexto fornecido pelo Encoder para gerar o próximo token.

### 3. Loop de Inferência Auto-Regressivo

Simulação do processo de inferência do Decoder, onde os tokens são gerados sequencialmente, um por um. Cada token gerado é então realimentado como entrada para a próxima etapa de geração, até que um token de fim de sequência (`<EOS>`) seja produzido.

**Funcionalidade:**
- Demonstra como o Decoder constrói uma sequência de saída, token por token, utilizando as probabilidades geradas a cada passo.

## Como Executar

Para executar o script e replicar os resultados, siga os passos abaixo:

1.  **Clone o repositório** (ou baixe os arquivos `laboratorio3_decoder.py` e `Relatorio_Laboratorio3.md`).
2.  **Certifique-se de ter o `numpy` instalado**:
    ```bash
    pip install numpy
    ```
3.  **Execute o script Python**:
    ```bash
    python laboratorio3_decoder.py
    ```

Os resultados da execução serão impressos no console, demonstrando o funcionamento de cada tarefa.

## Resultados Esperados

Ao executar o script, você verá a saída para cada uma das três tarefas, incluindo:

-   A matriz da máscara causal e os pesos de atenção resultantes, confirmando que as probabilidades para tokens futuros são zero.
-   As dimensões das entradas e saídas da função de Cross-Attention, validando a forma dos tensores.
-   Uma frase gerada pelo loop de inferência, como por exemplo: `<START> o rato roeu a roupa do rei de roma <EOS>`.


## Creditos

Manus AI
Gemini (para tirar algumas dúvidas e criar o README)
