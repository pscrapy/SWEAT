========================================
Sliced Word Embeddings Association Test
========================================

The Sliced Word Embedding Association Test (SWEAT) is a statistical measure of relative semantic polarization for pairs of textual corpora.

The measure relies on aligned distributional representations for the elements of each corpora vocabulary, specifically aligned Word2Vec gensim models are assumed in this implementation.
While any pair of Word2Vec models can be used, alignment is necessary to ensure that the two distributional representations are comparable.

Guide
-----
