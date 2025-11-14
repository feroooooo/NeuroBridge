import json

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# A custom JSON dumper that pretty-prints dictionaries with indentation, while keeping lists in a compact format.
def dump_pretty(obj, fp, indent=4, ensure_ascii=False):
    def _serialize(o, level):
        if isinstance(o, dict):
            if not o:
                return "{}"
            items = []
            indent_str = " " * (indent * level)
            child_indent_str = " " * (indent * (level + 1))

            items.append("{\n")
            kvs = list(o.items())
            for i, (k, v) in enumerate(kvs):
                key = json.dumps(k, ensure_ascii=ensure_ascii)
                value = _serialize(v, level + 1)
                items.append(f"{child_indent_str}{key}: {value}")
                if i < len(kvs) - 1:
                    items.append(",")
                items.append("\n")
            items.append(indent_str + "}")
            return "".join(items)

        elif isinstance(o, list):
            inner = ", ".join(_serialize(v, level) for v in o)
            return "[" + inner + "]"

        else:
            return json.dumps(o, ensure_ascii=ensure_ascii)

    fp.write(_serialize(obj, 0))

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