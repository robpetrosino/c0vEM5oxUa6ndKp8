# basics
import pandas as pd
import numpy as np

# embedding models
import transformers #BERT and SBERT
import torch

# metrics
from sklearn.metrics.pairwise import cosine_similarity

# import data
data = read.csv('../data/processed/potential-talents_aspiring-humanresources_seeking-human-resources_preprocessed_minimal.csv')

##### Part 1 -- embedding calculations and fitting #####

# I will draw embeddings from the SBERT model (but any alternative model would also work; see notebook for further info)
sbert = transformers.AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
sbert_tokenizer = transformers.AutoTokenizer.from_pretrained('sentece-transformers/all-MiniLM-L6-v2')

def str_to_sbert_embedding(str):
  ids = sbert_tokenizer.encode_plus(str, add_special_tokens=True, return_tensors='pt')
  out = sbert(**ids)
  embeddings = torch.mean(out.last_hidden_state, dim=1)
  return embeddings

sbert_data = data.copy()
sbert_title_embeddings = [emb.detach().numpy() for emb in sbert_data['job_title'].apply(str_to_sbert_embedding)]
sbert_keywords_embeddings = str_to_sbert_embedding(keywords[0]).detach().numpy()

# calculating the fit score as cosine similarity
sbert_cosine = [cosine_similarity(sbert_keywords_embeddings, sbert_title_embedding).item() for sbert_title_embedding in bert_title_embeddings]
sbert_data['sbert_fit'] = sbert_cosine

# merging the fit score with the dataset
data = data.merge(sbert_data['sbert_fit'], how='left', left_index=True, right_index=True)
data.sort_values('sbert_fit', ascending=False, inplace=True)


######## Part 2 -- reranking ########

# There are two ways to do this (see notebook for more info). Here I will just use the most intuitive method: i.e., add the job title of the selected candidates as actual keywords to calculate similarity of the tile embeddings to.
rerank_data = data.copy()
candidate_id = [None, None] ## <- candidate IDs should be added here
updated_keywords = update_keywords(keywords, candidate_id, rerank_data)

sbert_updated_keywords_embeddings = str_to_sbert_embedding(updated_keywords).detach().numpy()
sbert_cosine_reranked = [cosine_similarity(sbert_updated_keywords_embeddings, sbert_title_embedding).item() for sbert_title_embedding in sbert_title_embeddings]

rerank_data['rerank_sbert_fit'] = sbert_cosine_reranked
rerank_data.sort_values('rerank_sbert_fit', ascending=False, inplace=True)
