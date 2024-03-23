import torch
from glob import glob 
import faiss


def find_nearest_neighbours(query_embeddings, corpus_embeddings, k=1):

    """
    Takes list of queries and finds k nearest neighbours among a list of embeddings
    Nearest neighbours and distances are saved.
    """

    # Initialise faiss
    # res = faiss.StandardGpuResources()
    d=len(corpus_embeddings[0])

    # index = faiss.GpuIndexFlatIP(res, d)
    index=faiss.IndexFlatIP(d)
    index.add(corpus_embeddings)

    # Find k nearest neighbours
    dist_list, nn_list = index.search(query_embeddings, k)

    return dist_list, nn_list



