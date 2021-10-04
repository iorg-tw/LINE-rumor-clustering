import os
import numpy as np
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

import grouper as g

import logging

logging.basicConfig(
    format="[%(levelname)s] %(asctime)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.getenv("LOG_LEVEL", "INFO"),
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def clustering(data):
    tokenList = np.array([x['tokens'] for x in data.values()], dtype=object)
    wv = g.vectorize(tokenList)
    labels = g.run(wordVector=wv)
    id2label = dict(zip(data.keys(), labels))
    return id2label


def lda(data, n_topics):
    tokenList = [v['tokens'] for v in data.values()]
    dictionary = Dictionary(tokenList)
    corpus = [dictionary.doc2bow(text) for text in tokenList]
    lda_model = LdaMulticore(corpus, num_topics=n_topics, id2word=dictionary, passes=3, workers=2)
    id2label = dict()
    for i, k in enumerate(data.keys()):
        predicted_label = sorted(lda_model[corpus[i]], key=lambda tup: -1 * tup[1])[0][0]
        id2label[k] = predicted_label
    return id2label


def pca_kmeans(data, n_dim, n_topics):
    tokenList = np.array([x['tokens'] for x in data.values()], dtype=object)
    wv = g.vectorize(tokenList)
    svd = TruncatedSVD(n_components=n_dim)
    wvs = svd.fit_transform(wv)
    kmeans = KMeans(n_clusters=n_topics).fit(wvs)
    return dict(zip(data.keys(), kmeans.labels_))


def cluster_classification(data, train_portion, distance_threshold=0.6):
    n_articles = len(data.keys())
    out = np.full(n_articles, -100)  # init

    tokenList = np.array([x['tokens'] for x in data.values()], dtype=object)
    wv = g.vectorize(tokenList)

    training_size = int(n_articles * train_portion)
    uid_train = np.random.choice(n_articles, training_size, replace=False)
    uid_train = np.sort(uid_train)
    uid_test = np.array(list(set(range(n_articles)) - set(uid_train)))

    wv_train = wv[uid_train, :]
    wv_test = wv[uid_test, :]

    logger.debug("** Clustering **")
    labels = g.run(wordVector=wv_train, distance_threshold=distance_threshold)

    countsof = Counter(labels)
    y_train = np.array([x if countsof[x] > 1 else -1 for x in labels])
    out[uid_train] = y_train

    # classification
    logger.debug("** Classification **")
    knn = KNeighborsClassifier(n_neighbors=10, weights='distance')
    knn.fit(wv_train, y_train)

    y_pred = knn.predict(wv_test)
    out[uid_test] = y_pred

    logger.debug("** 2nd Clustering **")
    logger.debug(f"{len(np.where(out == -1)[0])} leftovers")

    uid_leftover = np.where(out == -1)[0]
    wv_leftover = wv[uid_leftover, :]
    labels_leftover = g.run(wordVector=wv_leftover, distance_threshold=distance_threshold)

    labels_leftover = (np.max(out) + 1) + labels_leftover

    out[uid_leftover] = labels_leftover
    id2label = dict(zip(data.keys(), out))

    return id2label
