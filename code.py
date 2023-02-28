"""Main script for finding duplicates"""
import csv
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import time
import torch

pd.options.mode.chained_assignment = None


def storing_duplicate_info(unique_col, dup_type, list):
    for test_desc in full_dups[unique_col].unique():
        matches = full_dups[full_dups[unique_col] == test_desc]
        first = matches['id'].unique()[0]
        for id in matches['id'].unique()[1:]:
            row = [first, id, dup_type]
            list.append(row)


# setting up the data
with open("wi_dataset.csv", encoding='utf-8') as infile:
    all = list(csv.reader(infile))

data = pd.DataFrame(all[1:])
data.columns = all[0]

# find all full duplicates -- titles and descriptions
# TODO: change the categories used to determine duplicates -- include location, country_name?
data['full_dups'] = data.duplicated(subset=['title', 'description'], keep=False)
data['title_description'] = data['title'] + ". " + data['description']
full_dups = data[data['full_dups'] == True].copy()

# add full duplicate ids into list of all duplicates
duplicate_list = []
storing_duplicate_info('title_description', "FULL", duplicate_list)

## -----------------------------------------------------------------------
# find all semantic duplicates -- only expressed differently in natural language or in different languages
# remove full duplicates from data, only keep unique observations for future duplicate identification
data['duplicate'] = data.duplicated(subset=['title', 'description'], keep='first')
unique_data = data[data['duplicate'] == False].copy()


start = time.time()

transformer = SentenceTransformer("./transformers/sentence-transformers_paraphrase-multilingual-miniLM-L12-v2",
                                  device='cpu')

# use this if you don't have the transformers downloaded yet
# transformer = SentenceTransformer("paraphrase-multilingual-miniLM-L12-v2",
#                                   device='cpu')

transformer.max_seq_length = 512

semantic_data = unique_data.head(10)
strings = semantic_data['title_description'].values.tolist()

embedding = transformer.encode(strings, convert_to_tensor=True, normalize_embeddings=True)
import pickle
with open('embeddings.pkl', "wb") as outfile:
    pickle.dump({'embeddings': embedding}, outfile, protocol=pickle.HIGHEST_PROTOCOL)

# with open('embeddings.pkl', "rb") as infile:
#     embedding = pickle.load(infile)['embeddings']

cos_sim_scores = pd.DataFrame(util.cos_sim(embedding, embedding).numpy())
cos_sim_scores.to_csv("cos_sim_scores.csv")

top_values = min(3, len(strings))
mask = torch.ones(len(strings)).type(torch.bool)

results = []
index = 0
for string in strings:
    out = []
    mask[index] = False
    to_compare = embedding[mask]
    check_embed = transformer.encode(string, convert_to_tensor=True, normalize_embeddings=True)

    # results.append(out)

    mask[index] = True
    index += 1

print(f'it took {time.time() - start} to complete')

# semantic_data[['top_score', 'top_index',
#                'sec_score', 'seco_index',
#                'third_score', 'third_index',
#                'four_score', 'four_index',
#                'fifth_score', 'fifth_index']] = results
# semantic_data[['top_index', 'seco_index', 'third_index', 'four_index', 'fifth_index']] = \
#     semantic_data[['top_index', 'seco_index', 'third_index', 'four_index', 'fifth_index']].astype('int32')
# semantic_data.to_csv("similarities.csv")

# add semantic duplicate ids into list
# TODO: create bag of words and compare percentages of words that are the same in each advertisement
# TODO: those with between XXX and 0.99 similarity, same/diff company name, same dates posted / retrieval dates

## -----------------------------------------------------------------------
# find all temporal duplicates -- semantic duplicates, but also different advertisement retrieval / vacancy expired date
# TODO: find obs in similar range as semantic dups, but with different ad retrieval or dates posted

# add temporal duplicate ids into list


## -----------------------------------------------------------------------
# finding all partial duplicates -- describe same position, but don't contain all the same elements
# TODO: compare the 'matches' of between 0.6-0.8 and between 0.8-0.99... Which better match the given criteria?

# add partial duplicate ids into list


## -----------------------------------------------------------------------
# exporting duplicates to csv
# with open('duplicates.csv', 'w', newline='') as outfile:
#     writer = csv.writer(outfile)
#     for dup in duplicate_list:
#         writer.writerow(dup)
