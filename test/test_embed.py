from newsdejavu import embed, find_nearest_neighbours


class TestEmbed:

    def test_embed(self):
        same_story_model = 'dell-research-harvard/same-story'
        corpus_dict = [{"id": 1, "masked_sentence": "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian"},
                   {"id": 2, "masked_sentence": "I am John Doe and I live in New York. I work at Google. I am a Software Engineer. I am a Nigerian"}]

        embeddings = embed(
            corpus_dict,
            model = same_story_model,
            save_path=None
        )

        assert embeddings.shape[0] == len(corpus_dict)