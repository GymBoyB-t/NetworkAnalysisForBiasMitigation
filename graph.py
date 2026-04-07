# faiss = Fast search similarity library; for the relationship between our nodes
import faiss
import networkx as nx


def normalizeXforSimilarityCalculations(X):
    faiss.normalize_L2(X)
    return


def get_similaritySearchIndex(X_train):
    
    dimension = X_train.shape[1]
    nlist = 100

    quantizer = faiss.IndexFlatIP(dimension)

    index = faiss.IndexIVFFlat(
        quantizer,
        dimension,
        nlist,
        faiss.METRIC_INNER_PRODUCT
    )

    index.train(X_train)
    index.add(X_train)
    index.nprobe = 20

    return index

def build_knn_graph(X, index, k=20):
    similarities, neighbors = index.search(X, k)

    G = nx.Graph()

    for i in range(len(X)):
        for j, sim in zip(neighbors[i][1:], similarities[i][1:]):
            if sim > 0:
                G.add_edge(i, j, weight=float(sim))

    return G



