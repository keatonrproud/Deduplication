"""Main script for finding duplicates"""
import csv
import numpy as np
import pandas as pd
from pickle import load, dump, HIGHEST_PROTOCOL
from os import path
import time
import cohere
from sentence_transformers import util
from memory_profiler import profile
import gc


class Dedup:
    co = cohere.Client("ho80SX8n3y7ANbv44gATQ61Zfe7KvnYoqqKSp6H5")

    def __init__(self, existing_scores=True, row_num: int = 0):

        # load data for the project
        data = self.set_up_data()

        print('data loaded')

        # build empty list to store all duplicates in prior to writing results to csv
        all_dups = []

        # find and store FULL DUPLICATES -- observations who are or have a duplicate based on the selected columns
        full_dups = self.find_full_dups(data, ['title', 'description'])
        self.store_full_duplicates(full_dups, all_dups)

        print('full dups stored')


        # create df of all unique observations for future analysis (ie remove duplicates)
        unique_data = self.create_unique_df(data)

        print('unique data ready')


        # get texts to compare for similarity analysis
        texts = unique_data['title_description'].values.tolist() if row_num == 0 \
            else unique_data['title_description'].iloc[0:row_num].values.tolist()

        # load from local file with embeddings, and if it doesn't exist then create one for future use
        embeds = self.load_or_store_embeddings(texts, obs=len(texts), length=512)

        print('embeds loaded')


        # if no filename for scores entered, create new scores and store appropriately
        scores = self.store_cos_sim_scores(embeds) if not existing_scores else \
            pd.read_hdf(f'outputs/cos_sim_scores_{len(embeds)}rows.h5')

        # convert diagonal scores to 0 as we don't want to compare against themselves
        scores.values[range(scores.shape[0]), range(scores.shape[0])] = 0

        print(scores)

        del embeds
        del data

        print('scores created/stored or read-in')

        # get top 100 (or 10% of rows) scores for each row
        top_scores_num = min(100, len(scores) // 10)
        top_scores = pd.DataFrame(np.sort(scores, axis=1)[:, ::-1][:, :top_scores_num])
        top_score_indices = pd.DataFrame(scores.columns.to_numpy()
                                         [np.argsort(scores.to_numpy(), axis=1)]
                                         [:, -1:-1*(top_scores_num+1):-1])
        top_scores = pd.concat([top_scores, top_score_indices], axis=1).sort_index(axis=1)

        top_scores.to_csv("top_scores.csv")
        scores.to_csv("raw_scores.csv")

        print(top_scores)
        print(scores)

        print(top_scores)


        # export all_dups list to csv for final results
        self.write_dups_to_csv(all_dups)

    # TODO: account for the few rows that I manually fixed in the original data file?

    @staticmethod
    @profile
    def set_up_data():
        # read in datafile
        with open("wi_dataset.csv", encoding='utf-8') as infile:
            all = list(csv.reader(infile))

        # create dataframe object with column names
        data = pd.DataFrame(all[1:])
        data.columns = all[0]

        # merge title and description into new column for future analysis
        data['title_description'] = data['title'] + ". " + data['description']

        return data

    @staticmethod
    @profile
    def find_full_dups(all_data, dup_cols: list):
        all_data['full_dups'] = all_data.duplicated(subset=dup_cols, keep=False)
        full_dups = all_data[all_data['full_dups'] == True].copy()

        return full_dups

    @staticmethod
    @profile
    def store_full_duplicates(dup_list, container: list):
        for test_desc in dup_list['title_description'].unique():
            # create df of all the matches with a test_description duplicate of the current test_desc value
            matches = dup_list[dup_list['title_description'] == test_desc]

            # get the first (and therefore lowest) id value of the set of all duplicates matching the current test_desc
            lowest_id_dup = matches['id'].unique()[0]

            # store all id's that aren't the lowest using the lowest as the first col, current as second, and "FULL" as type
            for curr_id in matches['id'].unique()[1:]:
                row = [lowest_id_dup, curr_id, "FULL"]
                container.append(row)

    @staticmethod
    @profile
    def create_unique_df(all_data):
        # remove full duplicates from data, only keep unique observations for future duplicate identification
        all_data['duplicate'] = all_data.duplicated(subset=['title', 'description'], keep='first')
        unique_data = all_data[all_data['duplicate'] == False].copy()

        return unique_data

    @classmethod
    @profile
    def load_or_store_embeddings(cls, strings_to_embed, obs: int, filename: str = 'title_desc', length: int = 512):
        ### for storing / loading embeddings locally
        embeddings_file = f'embeddings/cohere_{filename}_{obs}obs_{length}seq.pkl'
        if path.exists(embeddings_file):
            with open(embeddings_file, "rb") as infile:
                embedding = load(infile)['embeddings']
        else:
            if obs < 15000:
                embedding = cls.co.embed(texts=strings_to_embed).embeddings
            else:
                embedding = []
                chunk_val = 9000
                for curr_index in range(0, obs, chunk_val):
                    print(curr_index)
                    end = curr_index + chunk_val if curr_index + chunk_val < obs else obs
                    partial = cls.co.embed(texts=strings_to_embed[curr_index:end]).embeddings
                    embedding += partial
                    del partial
                    gc.collect()
                    time.sleep(40)
            with open(embeddings_file, "wb") as outfile:
                dump({'embeddings': embedding}, outfile, protocol=HIGHEST_PROTOCOL)

        return embedding

    @staticmethod
    @profile
    def store_cos_sim_scores(embeddings, chunk_val: int = 12000):
        row_num = len(embeddings)
        if row_num < chunk_val:
            scores = pd.DataFrame(util.cos_sim(embeddings, embeddings).numpy(), dtype='float32')
            scores.to_hdf(f'outputs/cos_sim_scores_{row_num}rows.h5', key='df', mode='w')
        else:
            scores = pd.DataFrame(dtype='float32')
            for curr_index in range(0, row_num, chunk_val):
                end = curr_index + chunk_val if curr_index + chunk_val < row_num else row_num

                start = time.time()
                print(f'calculating scores from row {curr_index} to row {end}')
                chunk_scores = pd.DataFrame(util.cos_sim(embeddings[curr_index:end], embeddings).numpy(), dtype='float32')
                pd.concat([scores, chunk_scores])
                scoretime = time.time() - start
                print(f'scoring took {scoretime / 60} mins, now on to the csv...')

                hdf_mode = 'w' if curr_index == 0 else 'a'
                chunk_scores.to_hdf(f'outputs/cos_sim_scores_{row_num}rows.h5', key='df', mode=hdf_mode)
                print(f'csv is ready! {(time.time() - scoretime) / 60} mins. Starting the next round...')

                del chunk_scores
                gc.collect()

        return scores

    @staticmethod
    @profile
    def write_dups_to_csv(duplicate_list):
        with open('duplicates.csv', 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            for dup in duplicate_list:
                writer.writerow(dup)


# -----------------------------------------------------------------------
# find SEMANTIC DUPLICATES -- only expressed differently in natural language or in different languages
# TODO: split title/description similarity scores
# TODO: add top similarity scores to dataframe
# TODO: those with between ~0.85 and 0.99 similarity, same/diff company name, same dates posted / retrieval dates
# TODO: there are some high similarity ads where the description is the exact same, but title is different!! need to identify these and figure out where to classify them
# TODO: remove from unique_data the 'full_dups' that were kept in as unique rows before adding to duplicate list for csv

# find TEMPORAL DUPLICATES -- semantic duplicates, but also different advertisement retrieval / vacancy expired date
# TODO: find obs in similar range as semantic dups, but with different ad retrieval or dates posted

# find PARTIAL DUPLICATES -- describe same position, but don't contain all the same elements
# TODO: compare the 'matches' of between 0.75 and .85 and between 0.85-0.99... Which better match the given criteria?

if __name__ == "__main__":
    # build class with Cohere client and init all model functions
    pd.options.mode.chained_assignment = None
    start = time.time()
    d = Dedup(existing_scores=False)
    print(f'{(time.time() - start)/60} mins required')
