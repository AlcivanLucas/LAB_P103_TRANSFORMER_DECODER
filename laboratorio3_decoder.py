import numpy as np


def softmax(x):
	e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
	return e_x / e_x.sum(axis=-1, keepdims=True)


# --- Tarefa 1: Implementando a Máscara Causa---


def create_causal_mask(seq_len):
	"""Cria uma máscara causal para atenção auto-regressiva.

	A parte triangular superior (tokens futuros) contém -infinito e a parte
	inferior (incluindo a diagonal) contém zeros.
	"""
	mask = np.triu(np.ones((seq_len, seq_len)), k=1)
	mask[mask == 1] = -np.inf
	return mask


def proof_task_1():
	print("--- Tarefa 1: Máscara Causal ---")
	seq_len = 5
	mask = create_causal_mask(seq_len)

	# Matrizes fictícias Q e K (seq_len, d_k)
	d_k = 8
	Q = np.random.randn(seq_len, d_k)
	K = np.random.randn(seq_len, d_k)

	# Produto escalar QK^T
	scaled_dot_product = np.matmul(Q, K.T) / np.sqrt(d_k)

	# Adicionando a máscara
	masked_logits = scaled_dot_product + mask

	# Aplicando Softmax
	attention_weights = softmax(masked_logits)

	print("Máscara M:")
	print(mask)
	print("\nPesos de Atenção (após Softmax):")
	print(np.round(attention_weights, 4))
	print("\nVerificação: As probabilidades para palavras futuras (triângulo superior) são 0.0?")
	print(np.allclose(np.triu(attention_weights, k=1), 0.0))
	print("-" * 30)


# --- Tarefa 2: A Ponte Encoder-Decoder (Cross-Attention) ---


def cross_attention(encoder_output, decoder_state):
	"""Calcula Cross-Attention entre a saída do encoder e o estado do decoder."""
	d_model = encoder_output.shape[-1]
	d_k = d_model  # Simplificação

	# Pesos arbitrários para projeção
	Wq = np.random.randn(d_model, d_k)
	Wk = np.random.randn(d_model, d_k)
	Wv = np.random.randn(d_model, d_k)

	# Projeções
	# Decoder fornece a Query (Q)
	Q = np.matmul(decoder_state, Wq)  # [batch, seq_len_en, d_k]

	# Encoder fornece Keys (K) e Values (V)
	K = np.matmul(encoder_output, Wk)  # [batch, seq_len_fr, d_k]
	V = np.matmul(encoder_output, Wv)  # [batch, seq_len_fr, d_k]

	# Scaled Dot-Product Attention
	# Q: [batch, seq_len_en, d_k], K^T: [batch, d_k, seq_len_fr]
	scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)
	weights = softmax(scores)

	output = np.matmul(weights, V)
	return output, weights


def run_task_2():
	print("\n--- Tarefa 2: Cross-Attention ---")
	batch_size = 1
	seq_len_fr = 10
	seq_len_en = 4
	d_model = 512

	encoder_output = np.random.randn(batch_size, seq_len_fr, d_model)
	decoder_state = np.random.randn(batch_size, seq_len_en, d_model)

	output, weights = cross_attention(encoder_output, decoder_state)

	print(f"Dimensões do encoder_output: {encoder_output.shape}")
	print(f"Dimensões do decoder_state: {decoder_state.shape}")
	print(f"Dimensões da saída da Cross-Attention: {output.shape}")
	print(f"Dimensões dos pesos de atenção: {weights.shape}")
	print("-" * 30)


# --- Tarefa 3: Simulando o Loop de Inferência Auto-Regressivo ---


# Vocabulário fictício para demonstração
VOCAB = ["<PAD>", "<START>", "o", "rato", "roeu", "a", "roupa", "do", "rei", "de", "roma", "<EOS>"]
VOCAB_SIZE = 10000


def generate_next_token(current_sequence, encoder_output):
	"""Simula a passagem pelo decoder e retorna um vetor de probabilidades."""
	# Mock: Retorna um vetor de probabilidades de tamanho VOCAB_SIZE
	probs = np.random.rand(VOCAB_SIZE)

	# Para tornar a simulação "realista", vamos dar alta probabilidade
	# para a próxima palavra lógica da nossa frase mock.
	next_idx_in_logic = len(current_sequence)
	if next_idx_in_logic < len(VOCAB):
		# Mapeia a palavra lógica para um índice no vocabulário de 10k
		# Aqui apenas usamos o índice da lista VOCAB como índice real
		probs[next_idx_in_logic] = 10.0

	return softmax(probs)

