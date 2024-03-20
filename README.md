This repository contains the code for a machine learning model that is trained on the `potential_talents` dataset provided by Apziva. The goal of the project is to streamline the first selection round of potential candidates by ranking their fit based on the semantic similarity between their job title and a (series of) specific keywords such as “full-stack software engineer”, “engineering manager”, or “aspiring human resources”. The code of this repo is also able to re-rank the candidate list, so that a candidate that the HR team may find particular fit for the role jump to the first positions regardless of their actual fit score.

## Prerequisites

The main packages required to run the code are:

1. `pandas`, `numpy`, and `math`
2. `nltk` (and sublibraries therein)
3. `scikit-learn`
4. `torchtext`
5. `gensim`
6. `transformers`

You can install these packages by running the following command: `pip install -r requirements.txt`

# Predict

The main goal of the project was to predict how fit each candidate for the position of 'aspiring human resources' based on their job title. In practice, the fit score can be seen as the vector distance between the vector embedding of the job description of a candidate and the vector embedding of the job position offered. So, the fitting here consists of two steps:

1. Get the embedding of the job title of each candidate, and the embedding of the description of the position offered. I will retrieve the embeddings from a different models:
  - Continuous Bag of Words
  - TF-IDF (pre-trained)
  - Word2Vec (pre-trained)
  - FastText (pre-trained)
  - BERT/SBERT (pre-trained)
2. Calculate the distance between the two vectors. For this project, I will use cosine similarity (rather than Euclidean distance). 

Apart from minimal differences embedding methods, the models seem to behave quite similarly. In absence of an objective evaluation method, the above was merely explorative.

An additional goal of the project was to re-rank candidates, so that a specific subset of candidates are moved to the first positions regardless of their fit score. Here I show two ways to do this. 

1. Add the job title of the selected candidate(s) to the keywords
2. Calculate the average of the embeddings of the job title of the selected candidates(s) and of the pre-set keywords.

# Conclusion

In this project, I showed how NLP techniques (i.e., vector embedding calculations and cosine similarity) can be applied to streamline and speed up the selection candidate process. This first, low-level pruning procedure seems quite effective and may provide a valid tool to objectively rank candidates for a given position based on their job title, while still providing recruiters the necessary freedom to manually adjust the ranking if needed. 
