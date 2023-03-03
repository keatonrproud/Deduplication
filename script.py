"""Main script for finding duplicates"""
import csv
import pandas as pd
from pickle import load, dump, HIGHEST_PROTOCOL
from os import path
import time
import cohere
from sentence_transformers import util
from memory_profiler import profile
import gc


class Dedup:
    def __init__(self):
        self.co = cohere.Client("ho80SX8n3y7ANbv44gATQ61Zfe7KvnYoqqKSp6H5")

        # load data for the project
        data = self.set_up_data()

        # build empty list to store all duplicates in prior to writing results to csv
        self.all_dups = []

        # find and store FULL DUPLICATES -- observations who are or have a duplicate based on the selected columns
        full_dups = d.find_full_dups(data, ['title', 'description'])
        self.store_full_duplicates(full_dups, self.all_dups)

        # create df of all unique observations for future analysis (ie remove duplicates)
        unique_data = self.create_unique_df(data)

        # get texts to compare for similarity analysis
        texts = unique_data['title_description'].values.tolist()

        # load from local file with embeddings, and if it doesn't exist then create one for future use
        embeds = self.load_or_store_embeddings(texts, obs=len(texts), length=512)

        # create the cos_sim_scores for every possible pair of the submitted encodings in embedding variable
        self.store_cos_sim_scores(embeds)




        # export all_dups list to csv for final results
        self.write_dups_to_csv(self.all_dups)

    # TODO: account for the few rows that I manually fixed in the original data file?

    @profile
    @staticmethod
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


    @profile
    @staticmethod
    def find_full_dups(all_data, dup_cols: list):
        all_data['full_dups'] = all_data.duplicated(subset=dup_cols, keep=False)
        full_dups = all_data[all_data['full_dups'] == True].copy()

        return full_dups

    @profile
    @staticmethod
    def store_full_duplicates(dup_list, container: list):
        for test_desc in dup_list['test_description'].unique():
            # create df of all the matches with a test_description duplicate of the current test_desc value
            matches = dup_list[dup_list['test_description'] == test_desc]

            # get the first (and therefore lowest) id value of the set of all duplicates matching the current test_desc
            lowest_id_dup = matches['id'].unique()[0]

            # store all id's that aren't the lowest using the lowest as the first col, current as second, and "FULL" as type
            for curr_id in matches['id'].unique()[1:]:
                row = [lowest_id_dup, curr_id, "FULL"]
                container.append(row)

    @profile
    @staticmethod
    def create_unique_df(all_data):
        # remove full duplicates from data, only keep unique observations for future duplicate identification
        all_data['duplicate'] = all_data.duplicated(subset=['title', 'description'], keep='first')
        unique_data = all_data[all_data['duplicate'] == False].copy()

        return unique_data

    @profile
    def load_or_store_embeddings(self, strings_to_embed, obs: int, filename: str = 'title_desc', length: int = 512):
        ### for storing / loading embeddings locally
        embeddings_file = f'embeddings/cohere_{filename}_{obs}obs_{length}seq.pkl'
        if path.exists(embeddings_file):
            with open(embeddings_file, "rb") as infile:
                embedding = load(infile)['embeddings']
        else:
            if obs < 15000:
                embedding = self.co.embed(texts=strings_to_embed).embeddings
            else:
                embedding = []
                chunk_val = 9000
                for curr_index in range(0, obs, chunk_val):
                    print(curr_index)
                    end = curr_index + chunk_val if curr_index + chunk_val < obs else obs
                    partial = self.co.embed(texts=strings_to_embed[curr_index:end]).embeddings
                    embedding += partial
                    del partial
                    gc.collect()
                    time.sleep(40)
            with open(embeddings_file, "wb") as outfile:
                dump({'embeddings': embedding}, outfile, protocol=HIGHEST_PROTOCOL)

        return embedding

    @profile
    @staticmethod
    def store_cos_sim_scores(embeddings, chunk_val: int = 12000):
        row_num = len(embeddings)
        if row_num < chunk_val:
            scores = pd.DataFrame(util.cos_sim(embeds, embeds).numpy())
            scores.to_hdf(f'outputs/cos_sim_scores_{row_num}rows.h5', key='df', mode='w')
        else:
            for curr_index in range(0, row_num, chunk_val):
                end = curr_index + chunk_val if curr_index + chunk_val < row_num else row_num

                start = time.time()
                print(f'calculating scores from row {curr_index} to row {end}')
                chunk_scores = pd.DataFrame(util.cos_sim(embeddings[curr_index:end], embeddings).numpy())
                scoretime = time.time() - start
                print(f'scoring took {scoretime / 60} mins, now on to the csv...')

                hdf_mode = 'w' if curr_index == 0 else 'a'
                chunk_scores.to_hdf(f'outputs/cos_sim_scores_{row_num}rows.h5', key='df', mode=hdf_mode)
                print(f'csv is ready! {(time.time() - scoretime) / 60} mins. Starting the next round...')

                del chunk_scores
                gc.collect()

    @profile
    @staticmethod
    def write_dups_to_csv(duplicate_list):
        with open('duplicates.csv', 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            for dup in duplicate_list:
                writer.writerow(dup)


start = time.time()

pd.options.mode.chained_assignment = None


# -----------------------------------------------------------------------
# find SEMANTIC DUPLICATES -- only expressed differently in natural language or in different languages


# TODO: translate, then create bag of words and compare percentages of words that are the same in each description?
# TODO: those with between ~0.85 and 0.99 similarity, same/diff company name, same dates posted / retrieval dates
# TODO: there are some high similarity ads where the description is the exact same, but title is different!! need to identify these and figure out where to classify them
# TODO: remove from unique_data the 'full_dups' that were kept in as unique rows before adding to duplicate list for csv

# find TEMPORAL DUPLICATES -- semantic duplicates, but also different advertisement retrieval / vacancy expired date
# TODO: find obs in similar range as semantic dups, but with different ad retrieval or dates posted

# find PARTIAL DUPLICATES -- describe same position, but don't contain all the same elements
# TODO: compare the 'matches' of between 0.75 and .85 and between 0.85-0.99... Which better match the given criteria?

if __name__ == "__main__":
    # build class with Cohere client and init all model functions
    d = Dedup()
