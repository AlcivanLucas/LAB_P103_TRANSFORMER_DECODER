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

