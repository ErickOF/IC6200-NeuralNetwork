import numpy as np


def softmax_crossentropy_with_logits(logits: np.array,
                                     reference_answers: np.array) -> np.array:
    """Compute crossentropy from logits[batch,n_classes] and ids of correct
    answers
    """
    logits_for_answers: np.array = logits[np.arange(len(logits)),
                                          reference_answers]

    xentropy: np.array = -logits_for_answers + np.log(np.sum(np.exp(logits),
                                                             axis=-1))

    return xentropy


def grad_softmax_crossentropy_with_logits(logits: np.array,
                                          reference_answers: np.array) -> np.array:
    """Compute crossentropy gradient from logits[batch,n_classes] and ids of
    correct answers"""
    ones_for_answers: np.array = np.zeros_like(logits)
    ones_for_answers[np.arange(len(logits)), reference_answers] = 1

    softmax: np.array = np.exp(logits) / np.exp(logits).sum(axis=-1,
                                                            keepdims=True)

    return (-ones_for_answers + softmax) / logits.shape[0]
