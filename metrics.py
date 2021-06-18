from typing import List, Iterable
from nltk.tokenize import word_tokenize


def exact_match(prediction: str, target: str) -> float:
    return 1.0 if prediction == target else 0.0


def levenshtein(s1: Iterable, s2: Iterable) -> int:
    dist = [[0 for _ in range(len(s1) + 1)] for _ in range(len(s2) + 1)]
    dist[0] = range(len(s1) + 1)
    
    for i2, el2 in enumerate(s2):
        dist[i2+1][0] = i2+1
        for i1, el1 in enumerate(s1):
            if el1 == el2:
                dist[i2+1][i1+1] = dist[i2][i1]
            else:
                dist[i2+1][i1+1] = 1 + min(dist[i2][i1], dist[i2+1][i1], dist[i2][i1+1])

    return dist[-1][-1]

def char_error_rate(prediction: str, target: str) -> float:
    return float(levenshtein(prediction, target))/len(target)

def word_error_rate(prediction: str, target: str) -> float:
    prediction = word_tokenize(prediction)
    target = word_tokenize(target)
    return float(levenshtein(prediction, target))/len(target)
