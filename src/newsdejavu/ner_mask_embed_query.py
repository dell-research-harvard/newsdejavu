from typing import List, Dict, Optional, Union, Tuple
from newsdejavu import ner_and_mask, embed, find_nearest_neighbours


def search_same_story(query_sentences: List[str],
                         corpus_sentences: List[str],
                         ner_model: str,
                         sentence_model: str,
                         batch_size: int = 256,
                         k=1,
                         corpus_ner_mask_path: Optional[str] = None,
                         corpus_embed_path: Optional[str] = None,
                         corpus_id_map: Optional[Dict[int, str]] = None) -> List[Tuple[str, str]]:
    """
    Applies Named Entity Recognition (NER) and masking to a list of query and corpus sentences, embeds them using a specified sentence embedding model, and finds the nearest neighbours for each query sentence in the corpus.

    Args:
        query_sentences (List[str]): A list of query sentences.
        corpus_sentences (List[str]): A list of corpus sentences against which to compare the query sentences.
        ner_model (str): The model identifier for the NER model to use.
        sentence_model (str): The model identifier for the sentence embedding model to use.
        batch_size (int, optional): The batch size for processing sentences through the NER model. Defaults to 256.
        k (int, optional): The number of nearest neighbours to find for each query sentence. Defaults to 1.
        corpus_ner_mask_path (Optional[str], optional): If provided, the function will load pre-masked corpus sentences from this path instead of masking them during runtime. Defaults to None.
        corpus_embed_path (Optional[str], optional): If provided, the function will load pre-computed corpus embeddings from this path instead of embedding them during runtime. Defaults to None.
        corpus_id_map (Optional[Dict[int, str]], optional): A dictionary mapping sentence indices to their corresponding raw corpus sentences. If not provided, a map is generated within the function.

    Returns:
        List[Tuple[str, str]]: A list of tuples, each containing a query sentence and its closest matching sentence from the corpus based on semantic similarity.

    This function processes both query and corpus sentences through NER and masking, then embeds them using the specified sentence model. It finds the nearest neighbour for each query sentence within the corpus and returns these pairs along with their original (raw) form.

    Example:
        >>> query_sentences_with_entities = ["Elon Musk's SpaceX is leading the private space industry."]
        >>> corpus_sentences_with_entities = ["Jeff Bezos' Blue Origin competes with SpaceX in the commercial space race."]
        >>> ner_model = 'ner-model-identifier'
        >>> sentence_model = 'sentence-model-identifier'
        >>> print(ner_mask_embed_query(query_sentences_with_entities, corpus_sentences_with_entities, ner_model, sentence_model))
        [("Elon Musk's SpaceX is leading the private space industry.", "Jeff Bezos' Blue Origin competes with SpaceX in the commercial space race.")]
    """


    if not  corpus_ner_mask_path and not corpus_embed_path:
        ner_masked_corpus=ner_and_mask(corpus_sentences, ner_model, batch_size = batch_size)
    if not corpus_embed_path:
        corpus_embeddings=embed(ner_masked_corpus, sentence_model, save_path=None)
    
    ner_masked_queries=ner_and_mask(query_sentences, ner_model, batch_size = batch_size)
    query_embeddings=embed(ner_masked_queries, sentence_model, save_path=None)
    

    dist_list, nn_list=find_nearest_neighbours(query_embeddings, corpus_embeddings, k=k)

    ###Get corresponding raw sentences - for each query, get the nearest neighbour and return the raw sentences from the corpus
    if not corpus_id_map:
        corpus_id_map={i:corpus_sentences[i] for i in range(len(corpus_sentences))}
        
    ##output dict - id: query, neighbor_list, distance_list
    output_dict={}
    for i in range(len(query_sentences)):
        output_dict[i]={"query":query_sentences[i],
                        "neighbor_list":[corpus_id_map[nn] for nn in nn_list[i]],
                        "distance_list":[dist for dist in dist_list[i]]}
    
    return  output_dict
    



query_sentences_with_entities = [
    "Elon Musk's SpaceX is leading the private space industry.",
    "The United Nations addressed climate change at the conference in Paris.",
    "Serena Williams triumphed at the Wimbledon Championships."
]

corpus_sentences_with_entities = [
    "Tesla, founded by Elon Musk, revolutionizes the electric vehicle market.",
    "The Paris Agreement aims to strengthen the global response to the threat of climate change.",
    "The FIFA World Cup is watched by millions of fans worldwide.",
    "Jeff Bezos' Blue Origin competes with SpaceX in the commercial space race.",
    "The World Health Organization plays a crucial role in managing global health crises.",
    "Roger Federer is known for his exceptional achievements in tennis.",
    "The Kyoto Protocol was an earlier international treaty aimed at combating global warming."
]

ner_model="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/thisdayinhistory/models"
sentence_model="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/thisdayinhistory/same_story_model"



print(search_same_story(query_sentences_with_entities,
                           corpus_sentences_with_entities,
                           ner_model,
                           sentence_model,
                           k=2))
    