import numpy


def text_to_vec(words: list, model):
    if len(words) == 0:
        return None

    word_vecs = []
    for word in words:
        try:
            word_vecs.append(model[word])
        except KeyError:
            pass

    if len(word_vecs) == 0:
        return None

    text_vec = numpy.zeros(word_vecs[0].shape, dtype=word_vecs[0].dtype)
    for word_vec in word_vecs:
        text_vec = text_vec + word_vec

    return text_vec

