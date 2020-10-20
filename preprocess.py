import numpy as np


def build_vocab(file_path, count_thr=2):
    with open(file_path, 'r', encoding='utf-8') as f:
        counts = {}
        for line in f:
            line = line.strip()
            words = line.split(' ')
            for word in words:
                counts[word] = counts.get(word, 0) + 1
        vocab = [w for w, n in counts.items() if n >= count_thr]
        return vocab


def generate_bow(file_path, ignore_freq=True):
    vocab = build_vocab(file_path)
    bows = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            bow = np.zeros(len(vocab))
            line = line.strip()
            words = line.split(' ')
            if ignore_freq:
                for i, word in enumerate(vocab):
                    if word in words:
                        bow[i] = 1
            else:
                for word in words:
                    for i, w in enumerate(vocab):
                        if w == word:
                            bow[i] += 1
            bows.append(bow)
    return np.vstack(bows)

# print(generate_bow('./data.txt'))