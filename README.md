# news-deja-vu
Python package for News Deja Vu

News Deja Vu is a novel semantic search tool that leverages transformer large language models and a bi-encoder approach to identify historical news articles that share semantic similarities with modern news queries. News Déjà Vu first recognizes and masks entities, in order to focus on broader parallels rather than the specific named entities being discussed. Then, a contrastively trained, lightweight bi-encoder retrieves historical articles that are most similar semantically to a modern query.

# Example Usage:

```[python]
ner_model = 'dell-research-harvard/historical_newspaper_ner'
same_story_model = 'dell-research-harvard/same-story'

# Download historic news articles
corpus = download('american stories:1840')
# Perform NER inference
ner_output = ner_and_mask(corpus, ner_model, batch_size = batch_size)
# Embed with biencoder
embeddings = embed(ner_output, same_story_model)

# NER inference for query sentences
query_masked_input = ner_and_mask(sample_query_sentences, ner_model, batch_size = batch_size)
# Embed query sentences
query_embeddings = embed(query_masked_input, same_story_model)

# Search for closest matches in historical corpus
dist_list, nn_list = find_nearest_neighbours(query_embeddings, embeddings, k=1)

# Output results
results_dict = {i: {"query": sample_query_sentences[i], "neighbor": dataset[nn_list[i]]} for i in range(len(sample_query_sentences))}
with open('data/test_data/query_results_1840.json', 'w') as f:
    json.dump(results_dict, f, indent = 4, default=str)
```

Or, in much simpler form:

```[python]
corpus = download('american stories:1840')
results = search_same_story(sample_query_sentences, corpus, ner_model, same_story_model, k = 1)
```

Outputs are query texts matched with their nearest matches in the historical corpus. 
