"""Main script for finding duplicates"""
import csv
import numpy as np
import pandas as pd
from pickle import load, dump, HIGHEST_PROTOCOL
from os import path
import time
import cohere
from sentence_transformers import util

pd.options.mode.chained_assignment = None


class Dedup:
    def __init__(self, new_scores=True, row_num: int = 0):

        # load data for the project
        data = self.set_up_data()

        # build empty list to store all duplicates in prior to writing results to csv
        all_dups = []

        # find and store FULL DUPLICATES -- observations who are or have a duplicate based on the selected columns
        cols_for_full_dups = ['title', 'description']
        full_dups = self.find_full_dups(data, cols_for_full_dups)
        #TODO: check length of full_dups
        self.store_full_duplicates(full_dups, cols_for_full_dups, all_dups)

        # create df of all unique observations for future analysis (ie keep only one of each full duplicate)
        unique_data = self.create_unique_df(data) if row_num == 0 else self.create_unique_df(data).head(row_num)
        #TODO: check length of unique data

        # get texts to compare for similarity analysis
        titles = unique_data['title'].values.tolist()
        descriptions = unique_data['description'].values.tolist()

        # load from local file with embeddings, and if it doesn't exist then create one for future use
        title_embeds = self.load_or_store_embeddings(titles, obs=len(titles), filename='title', length=512)
        desc_embeds = self.load_or_store_embeddings(descriptions, obs=len(descriptions), filename='desc', length=512)

        unique_data_scores, possible_dups = self.get_possible_duplicates([title_embeds, desc_embeds],
                                                                         unique_data=unique_data,
                                                                         new_scores=new_scores)

        # TODO: check between possible_dups to compare dates (diff = temporal, same = semantic), with title > 0.9
        # TODO: only those with same company are duplicates? I think so?
        # TODO: partial -- > 0.9 title, but lower description match? same company still?
        # TODO: check if full duplicates with just title/description is the same as including company name / location
        # TODO: implement removing meaningless / overfrequent words


        possible_dups.to_csv("outputs/possible_dups.csv")


        # export all_dups list to csv for final results
        self.write_dups_to_csv(all_dups)

    # TODO: account for the few rows that I manually fixed in the original data file?

    @staticmethod
    def set_up_data():
        # read in datafile
        with open("wi_dataset.csv", encoding='utf-8') as infile:
            all = list(csv.reader(infile))

        # create dataframe object with column names
        data = pd.DataFrame(all[1:])
        data.columns = all[0]

        return data

    @staticmethod
    def find_full_dups(all_data, dup_cols: list):
        all_data['full_dups'] = all_data.duplicated(subset=dup_cols, keep=False)
        full_dups = all_data[all_data['full_dups'] == True].copy()

        full_dups.sort_values(by=['title']).to_csv("full duplicates.csv")

        return full_dups

    @staticmethod
    def store_full_duplicates(dup_list, dup_cols, container: list):
        dup_list['full_dup_string'] = dup_list[dup_cols[0]]
        for col in dup_cols[1:]:
            dup_list['full_dup_string'] = dup_list['full_dup_string']+dup_list[col]

        for string in dup_list['full_dup_string'].unique():
            # create df of all the matches with a test_description duplicate of the current test_desc value
            matches = dup_list[dup_list['full_dup_string'] == string]

            # # get the first (and therefore lowest) id value of the set of all duplicates matching the current test_desc
            # lowest_id_dup = matches['id'].unique()[0]
            #
            # # store all id's that aren't the lowest using the lowest as the first col, current as second, and "FULL" as type
            # for curr_id in matches['id'].unique()[1:]:
            #     row = [lowest_id_dup, curr_id, "FULL"]
            #     container.append(row)

            # get list of ids of the duplicates
            dup_ids = list(matches['id'].unique())

            if len(dup_ids) % 2 != 0:
                dup_ids.insert(-2, dup_ids[-2])

            id_pairs = [[dup_ids[x], dup_ids[x + 1], "FULL"] for x in range(0, len(dup_ids), 2)]

            for pair in id_pairs:
                container.append(pair)

    @staticmethod
    def create_unique_df(all_data):
        # remove full duplicates from data, only keep unique observations for future duplicate identification
        unique_data = all_data[all_data['full_dups'] == False].copy()

        return unique_data

    @staticmethod
    def load_or_store_embeddings(strings_to_embed, obs: int, filename: str = 'details', length: int = 512):
        ### for storing / loading embeddings locally
        embeddings_file = f'embeddings/cohere_{filename}_{obs}obs_{length}seq.pkl'
        if path.exists(embeddings_file):
            with open(embeddings_file, "rb") as infile:
                embedding = load(infile)['embeddings']
        else:
            co = cohere.Client("ho80SX8n3y7ANbv44gATQ61Zfe7KvnYoqqKSp6H5")
            if obs < 9000:
                embedding = co.embed(texts=strings_to_embed).embeddings
            else:
                embedding = []
                chunk_val = 9000
                for curr_index in range(0, obs, chunk_val):
                    for _ in range(3):
                        # TODO: check last 9000 rows compared with first 9000 -- are they the same ???? wtf
                        try:
                            start_embedding = time.time()
                            end = curr_index + chunk_val if curr_index + chunk_val < obs else obs
                            print(f'currently embedding rows {curr_index} to {end}')
                            partial = co.embed(texts=strings_to_embed[curr_index:end]).embeddings
                            embedding += partial
                            del partial
                            print(f'this embedding took {time.time() - start_embedding}')
                            break
                        except Exception as e:
                            print(e)
                            print("Retrying now after a 1-minute wait.")
                            time.sleep(60)
                    time.sleep(40)
            with open(embeddings_file, "wb") as outfile:
                dump({'embeddings': embedding}, outfile, protocol=HIGHEST_PROTOCOL)

        return embedding

    @staticmethod
    def store_cos_sim_scores(embeddings, info_type: str, chunk_val: int = 12000):
        row_num = len(embeddings)
        if row_num < chunk_val:
            scores = pd.DataFrame(util.cos_sim(embeddings, embeddings).numpy(), dtype='float32')
        else:
            scores = pd.DataFrame(dtype='float32')
            for curr_index in range(0, row_num, chunk_val):
                end = curr_index + chunk_val if curr_index + chunk_val < row_num else row_num

                start = time.time()
                print(f'calculating scores from row {curr_index} to row {end}')
                chunk_scores = pd.DataFrame(util.cos_sim(embeddings[curr_index:end], embeddings).numpy(), dtype='float32')
                scores = pd.concat([scores, chunk_scores])
                print(f'scoring took {round((time.time() - start) / 60, 2)} mins, now on to the csv...')

        # make diagonal of scores matrix 0s to avoid top score being compared with itself
        scores.values[range(scores.shape[0]), range(scores.shape[0])] = 0

        # get top 100 (or 10% of rows) scores for each row
        num_of_scores = len(scores)
        top_scores_num = min(100, num_of_scores // 10)
        top_scores = pd.DataFrame(np.sort(scores, axis=1)[:, ::-1][:, :top_scores_num])
        top_scores_indices = pd.DataFrame(scores.columns.to_numpy()
                                         [np.argsort(scores.to_numpy(), axis=1)]
                                         [:, -1:-1 * (top_scores_num + 1):-1])
        for df in top_scores, top_scores_indices:
            df.columns = (df.columns.astype(int) + 1).astype(str)
        top_scores_indices = top_scores_indices.add_suffix("_index")
        top_scores = pd.concat([top_scores, top_scores_indices], axis=1)

        top_scores.to_hdf(f"outputs/top_scores_{info_type}_{num_of_scores}rows.h5", key='df', mode='w')

        return top_scores

    def get_possible_duplicates(self, embeddings: list, unique_data, new_scores: bool):
        # if no filename for scores entered, create new scores and store appropriately
        top_scores = []
        for embedding in embeddings:
            info_type = 'title' if embedding == embeddings[0] else 'desc'
            filename = f'outputs/top_scores_{info_type}_{len(embedding)}rows.h5'
            if new_scores or not path.exists(filename):
                top_scores.append(self.store_cos_sim_scores(embedding, info_type=info_type))
            else:
                top_scores.append(pd.read_hdf(filename))
            del embedding

        print('scores created/stored or read-in')

        # ensure indices align between dataframes, then combine by column
        top_scores[0] = top_scores[0].add_prefix('title_')
        top_scores[1] = top_scores[1].add_prefix('desc_')

        for df in top_scores:
            df.index = unique_data.index

        unique_data_scores = pd.concat([unique_data, top_scores[0], top_scores[1]], axis=1)

        unique_data_scores.to_csv("outputs/unique_data_scores.csv")

        title_cols = top_scores[0].columns
        desc_cols = top_scores[1].columns

        possible_dups = unique_data_scores[
            (((unique_data_scores[title_cols] < 1) & (unique_data_scores[title_cols] > 0.9)).any(axis=1) &
             ((unique_data_scores[desc_cols] < 1) & (unique_data_scores[desc_cols] > 0.9)).any(axis=1))]

        return unique_data_scores, possible_dups

    @staticmethod
    def write_dups_to_csv(duplicate_list):
        with open('duplicates.csv', 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            for dup in duplicate_list:
                writer.writerow(dup)


# -----------------------------------------------------------------------
# find SEMANTIC DUPLICATES -- only expressed differently in natural language or in different languages

# find TEMPORAL DUPLICATES -- semantic duplicates, but also different advertisement retrieval / vacancy expired date

# find PARTIAL DUPLICATES -- describe same position, but don't contain all the same elements

if __name__ == "__main__":
    # build class with Cohere client and init all model functions
    start = time.time()
    d = Dedup(new_scores=True, row_num=200)
    print(f'{round((time.time() - start)/60, 2)} mins required')
