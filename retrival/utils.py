import numpy as np
import numba as nb
from numba import njit
from typing import Tuple


@njit(cache=True)
def join_sorted(a1: np.array, a2: np.array):
    """

    递归排序两个数组的元素大小

    Parameters:
    ----------
    a1: np.array,

    a2: np.array,

    Returns:
    -------
    result: np.array

    """
    result = np.empty(len(a1) + len(a2), dtype=np.int32)
    i = 0
    j = 0
    k = 0

    while i < len(a1) and j < len(a2):
        if a1[i] < a2[j]:
            result[k] = a1[i]
            i += 1
        elif a1[i] > a2[j]:
            result[k] = a2[j]
            j += 1
        else:  # a1[i] == a2[j]
            result[k] = a1[i]
            i += 1
            j += 1
        k += 1

    result = result[:k]

    if i < len(a1):
        result = np.concatenate((result, a1[i:]))
    elif j < len(a2):
        result = np.concatenate((result, a2[j:]))

    return result


@njit(cache=True)
def join_sorted_multi(arrays):
    """
    for训练调用递归排序两个数组的元素大小

    Parameters:
    ----------
    arrays: n个的嵌套数组

    Returns:
    -------


    """
    a = arrays[0]

    for i in range(1, len(arrays)):
        a = join_sorted(a, arrays[i])

    return a


@njit(cache=True)
def join_sorted_multi_recursive(arrays):
    """

    递归排序两个数组的元素大小

    Parameters:
    ----------
    arrays: n个的嵌套数组

    Returns:
    -------
    元素排序后的数组，


    """
    if len(arrays) == 1:
        return arrays[0]
    elif len(arrays) == 2:
        return join_sorted(arrays[0], arrays[1])
    else:
        return join_sorted(join_sorted_multi(arrays[:2]), join_sorted_multi(arrays[2:]))


@njit(cache=True)
def unsorted_top_k(array: np.ndarray, k: int):
    top_k_values = np.zeros(k, dtype=np.float32)
    top_k_indices = np.zeros(k, dtype=np.int32)

    min_value = 0.0
    min_value_idx = 0

    for i, value in enumerate(array):
        if value > min_value:
            top_k_values[min_value_idx] = value
            top_k_indices[min_value_idx] = i
            min_value_idx = top_k_values.argmin()
            min_value = top_k_values[min_value_idx]

    return top_k_values, top_k_indices


@njit(cache=True)
def bm25(doc_ids: nb.typed.List[np.ndarray],
         term_doc_freqs: nb.typed.List[np.ndarray],
         relative_doc_lens: nb.typed.List[np.ndarray],
         doc_count: int,
         n_top=100,
         b=0.75,
         k1=1.2) -> Tuple[np.ndarray]:
    """

    Parameters:
    ----------
    doc_ids: nb.typed.List[np.ndarray], 命中的doc id嵌套列表

    term_doc_freqs: nb.typed.List[np.ndarray], 命中的doc tfs嵌套列表

    n_top: 返回排序后的N个

    b, k1: float, 都是公式参数

    Returns:
    -------



    """
    # 对嵌套的doc_ids进行排序 -> 一维数组(doc_id1, doc_id2, ...)
    unique_doc_ids = join_sorted_multi_recursive(doc_ids)

    scores = np.empty(doc_count, dtype=np.float32)
    scores[unique_doc_ids] = 0.0

    for i in range(len(term_doc_freqs)):
        indices = doc_ids[i]
        freqs = term_doc_freqs[i]

        df = np.float32(len(indices))
        idf = np.float32(np.log(1.0 + (((doc_count - df) + 0.5) / (df + 0.5))))

        scores[indices] += idf * (
                (freqs * (k1 + 1.0)) / (freqs + k1 * (1.0 - b + (b * relative_doc_lens[indices])))
        )

    scores = scores[unique_doc_ids]

    # 进行倒排
    if n_top < len(scores):
        scores, indices = unsorted_top_k(scores, k=n_top)
        unique_doc_ids = unique_doc_ids[indices]

    indices = np.argsort(-scores)

    return unique_doc_ids[indices], scores[indices]
