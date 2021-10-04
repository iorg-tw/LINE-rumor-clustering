import numpy as np
import scipy.sparse
import scipy.sparse
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer
import logging

logger = logging.getLogger(__name__)


def sizeof(a):
    if isinstance(a, np.ndarray):
        return a.nbytes
    elif scipy.sparse.issparse(a):
        return a.data.nbytes + a.indptr.nbytes + a.indices.nbytes


def run(tokenList=None, wordVector=None, distance_threshold=0.6, linkage='average'):
    if wordVector is None:
        logger.debug("*** Generating Word Vectors ***")
        wordVector = vectorize(tokenList)

    logger.debug("*** Calculating Distance Matrix ***")
    similarityMatrix = calculate_similarity(wordVector)
    distanceMatrix = 1 - similarityMatrix

    logger.debug(f"distanceMatrix size: {sizeof(distanceMatrix) * (10 ** -9):.2f} GB")
    del similarityMatrix

    model = AgglomerativeClustering(distance_threshold=distance_threshold,
                                    n_clusters=None,
                                    affinity="precomputed",
                                    linkage=linkage)
    labels = model.fit_predict(distanceMatrix)

    return labels


def vectorize(tokenList):
    stringifyToks = [" ".join(t) for t in tokenList]
    cv = CountVectorizer(binary=True, min_df=10, max_df=0.9)
    wordVector_csr = cv.fit_transform(stringifyToks)
    return wordVector_csr


def calculate_similarity(mat):
    if not scipy.sparse.issparse(mat):
        mat = scipy.sparse.csr_matrix(mat)
    mat = mat.astype(float)

    intrsct = mat * mat.T
    logger.debug(f"intrsct size: {sizeof(intrsct) * (10 ** -9):.2f} GB")

    # for rows
    row_sums = mat.getnnz(axis=1)
    nnz_i = np.repeat(row_sums, intrsct.getnnz(axis=1))
    nnz_j = row_sums[intrsct.indices]

    intrsct.data = intrsct.data / np.maximum(nnz_i, nnz_j)
    return intrsct.A


def pairwise_jaccard(mat):
    if not scipy.sparse.issparse(mat):
        mat = scipy.sparse.csr_matrix(mat)
    mat = mat.astype(float)

    intrsct = mat * mat.T
    logger.debug(f"intrsct size: {sizeof(intrsct) * (10 ** -9):.2f} GB")

    # for rows
    row_sums = mat.getnnz(axis=1)
    nnz_i = np.repeat(row_sums, intrsct.getnnz(axis=1))
    union = nnz_i + row_sums[intrsct.indices] - intrsct.data

    logger.debug(f"union size: {sizeof(union) * (10 ** -9):.2f} GB")

    intrsct.data = intrsct.data / union
    return intrsct.A
