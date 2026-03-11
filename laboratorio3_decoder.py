import numpy as np


def softmax(x):
	e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
	return e_x / e_x.sum(axis=-1, keepdims=True)


# --- Tarefa 1: Implementando a Máscara Causal (Look-Ahead Mask) ---


def create_causal_mask(seq_len):
	"""Cria uma máscara causal para atenção auto-regressiva.

	A parte triangular superior (tokens futuros) contém -infinito e a parte
	inferior (incluindo a diagonal) contém zeros.
	"""
	mask = np.triu(np.ones((seq_len, seq_len)), k=1)
	mask[mask == 1] = -np.inf
	return mask

