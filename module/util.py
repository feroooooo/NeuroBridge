from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def topk(matrix, k):
    sorted_indices = np.argsort(-matrix, axis=1)
    rankings = np.argsort(sorted_indices, axis=1)
    diagonal_ranks = np.diag(rankings) + 1

    count_k = 0
    count_1  = 0
    for i in range(sorted_indices.shape[0]):
        if diagonal_ranks[i] <= k:
            count_k += 1
        if diagonal_ranks[i] == 1:
            count_1 += 1
    return count_k, count_1

def retrieve_all(eeg_features, image_features, average:bool):
    similarity_matrix = cosine_similarity(eeg_features, image_features)
    count_5, count_1 = topk(similarity_matrix, 5)
    return count_5, count_1, eeg_features.shape[0]