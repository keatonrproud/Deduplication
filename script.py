"""Main script for finding duplicates"""
import csv
import pandas as pd
from pickle import load, dump, HIGHEST_PROTOCOL
from os import path
from sentence_transformers import SentenceTransformer, util
import time
import torch

pd.options.mode.chained_assignment = None


def storing_duplicate_info(dup_data, unique_col, dup_type, list):
    for test_desc in dup_data[unique_col].unique():
        matches = dup_data[dup_data[unique_col] == test_desc]
        first = matches['id'].unique()[0]
        for id in matches['id'].unique()[1:]:
            row = [first, id, dup_type]
            list.append(row)


def load_or_store_pickle_files(strings_to_embed, obs: object="all"):
    ### for storing / loading embeddings locally
    embeddings_file = f'embeddings/embeddings_{obs}.pkl'
    if path.exists(embeddings_file):
        with open(embeddings_file, "rb") as infile:
            embedding = load(infile)['embeddings']
    else:
        embedding = transformer.encode(strings_to_embed, convert_to_tensor=True, normalize_embeddings=True)
        with open(embeddings_file, "wb") as outfile:
            dump({'embeddings': embedding}, outfile, protocol=HIGHEST_PROTOCOL)
    return embedding


# setting up the data
with open("wi_dataset.csv", encoding='utf-8') as infile:
    all = list(csv.reader(infile))

data = pd.DataFrame(all[1:])
data.columns = all[0]

## ----------------------------------------------------------------------
# CLEAN / EXPLORE
# merge title and description into new column for future analysis
data['title_description'] = data['title'] + ". " + data['description']

# removing 50 most frequent words -- almost exclusively meaningless words like 'the', 'and', etc.
words = [f'(?i) {word[0]} ' for word in list(pd.DataFrame(' '.join(data.title_description).split()).value_counts().index) if
         word not in [" - ", " : ", " ; ", " / ", " . ", " , "]]
to_replace = dict.fromkeys(words[0:50] + ["(?i)\\*", "(?i)https:\\/\\/", "(?i)http:\\/\\/"], " ")

data['title_description'] = \
    data['title_description'].replace(to_replace, regex=True)
pd.DataFrame(' '.join(data.title_description).split()).value_counts().to_csv("WORDCOUNTS.csv")

## ----------------------------------------------------------------------
# find FULL DUPLICATES based on title and description
# TODO: change the categories used to determine duplicates -- include location, country_name?
# create full_dups based on all observations who are or have a duplicate based on selected columns
data['full_dups'] = data.duplicated(subset=['title', 'description'], keep=False)
full_dups = data[data['full_dups'] == True].copy()

# add full duplicate ids into list of all duplicates
duplicate_list = []
storing_duplicate_info(full_dups, 'title_description', "FULL", duplicate_list)

# remove full duplicates from data, only keep unique observations for future duplicate identification
data['duplicate'] = data.duplicated(subset=['title', 'description'], keep='first')
unique_data = data[data['duplicate'] == False].copy()

## -----------------------------------------------------------------------
# find SEMANTIC DUPLICATES -- only expressed differently in natural language or in different languages
start = time.time()

# set up transformer for encoding text and set max length of an allowed sequence of encodings
transformer = SentenceTransformer("./transformers/sentence-transformers_paraphrase-multilingual-miniLM-L12-v2",
                                  device='cpu')
#### use this if you don't have the transformers downloaded yet
# transformer = SentenceTransformer("paraphrase-multilingual-miniLM-L12-v2", device='cpu')
transformer.max_seq_length = 512

# remove unhelpful words to increase valuable info in the encodings and increase accuracy of scores
unique_data.replace(" the ", " ", regex=True, inplace=True)

# build data for text similarity analysis
semantic_data = unique_data.head(10)
strings_to_compare = semantic_data['title_description'].values.tolist()

####### TODO: remove this -- only used for evaluating the effects of replace
semantic_data.to_csv("DELETE_THIS.csv")

# load from local pickle file with embeddings, and if it doesn't exist then create one for future use
rows = len(semantic_data)
embedding = load_or_store_pickle_files(strings_to_compare, rows) if rows != len(unique_data) else load_or_store_pickle_files(strings_to_compare)

# create the cos_sim_scores for every possible pair of the submitted encodings in embedding variable
cos_sim_scores = pd.DataFrame(util.cos_sim(embedding, embedding).numpy())
cos_sim_scores.to_csv("cos_sim_scores.csv")

# create set of all True values to mask the embedding for each round
mask = torch.ones(len(strings_to_compare)).type(torch.bool)

print(f'it took {time.time() - start} to complete')

# add semantic duplicate ids into list
# TODO: create bag of words and compare percentages of words that are the same in each advertisement after translating?
# TODO: those with between ~0.85 and 0.99 similarity, same/diff company name, same dates posted / retrieval dates

## -----------------------------------------------------------------------
# find TEMPORAL DUPLICATES -- semantic duplicates, but also different advertisement retrieval / vacancy expired date
# TODO: find obs in similar range as semantic dups, but with different ad retrieval or dates posted

# add temporal duplicate ids into list


## -----------------------------------------------------------------------
# finding PARTIAL DUPLICATES -- describe same position, but don't contain all the same elements
# TODO: compare the 'matches' of between 0.75 and .85 and between 0.85-0.99... Which better match the given criteria?

# add partial duplicate ids into list


## -----------------------------------------------------------------------
# exporting duplicates to csv
# with open('duplicates.csv', 'w', newline='') as outfile:
#     writer = csv.writer(outfile)
#     for dup in duplicate_list:
#         writer.writerow(dup)
