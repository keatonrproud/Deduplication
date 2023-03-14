"""Main script for finding duplicates"""
import csv
import numpy as np
import pandas as pd
from pickle import load, dump, HIGHEST_PROTOCOL
from os import path
import time
import cohere
from sentence_transformers import util
import gc
from memory_profiler import profile

pd.options.mode.chained_assignment = None


# TODO: account for the few rows that I manually fixed in the original data file

# TODO: get real id of observation before converting to possible_duplicates, otherwise we lose where things are in index
        # convert index column value for a score to the id value of that index in unique_df
# TODO: remove observation from itself -- check why top scores isn't setting 0 for diagonal
        # it is because the scores on diagonal is doing it starting from 0,0 of each chunk's scores matrix, when it should
        # be 0 for every index equal to the column... hard to do
        # maybe run counter of num_chunks and for each num in the chunk, set that row/column to 0?

# TODO: how to classify title/desc the same + company/location/country diff + date diff?   Right now, NON-DUPLICATE

# TODO: use similarity scores to determine possible dups and find temporal/semantic/partial duplicates
    # TODO: check between possible_dups to compare dates (diff = temporal, same = semantic), with title > 0.9
    # TODO: partial -- > 0.9 title, but lower description match? same company still?

class Dedup:
    def __init__(self, new_scores=False, row_num: int = 0):

        # load data for the project
        data = self.set_up_data()

        # build empty list to store all duplicates in prior to writing results to csv
        all_dups = []

        # find and store FULL DUPLICATES -- observations who are or have a duplicate based on the selected columns
        exact_dup_cols = ['title', 'description', 'location', 'country_id', 'company_name', 'retrieval_date']
        lower_title_desc = ['title_lower', 'description_lower']

        for col in range(len(exact_dup_cols)):
            data[f'{exact_dup_cols[col]}_lower'] = data[exact_dup_cols[col]].fillna('').apply(lambda x: x.lower())
            exact_dup_cols[col] = f'{exact_dup_cols[col]}_lower'
        title_desc_match = data[data.duplicated(subset=lower_title_desc, keep=False) == True].copy()

        full_dups = title_desc_match[title_desc_match.duplicated(subset=exact_dup_cols, keep=False) == True].copy()
        almost_full_dups = title_desc_match[title_desc_match.duplicated(subset=exact_dup_cols, keep=False) == False].copy()


        full_dups['full_check'] = full_dups[exact_dup_cols[0]]
        for col in exact_dup_cols:
            full_dups['full_check'] += full_dups[col]
        self.store_ids(full_dups, 'full_check', "FULL", all_dups)


        almost_full_temporal_cols = ['title_lower', 'description_lower', 'location_lower', 'country_id_lower',
                                     'company_name_lower']
        almost_full_dups['temporal_check'] = almost_full_dups[almost_full_temporal_cols[0]]
        for col in almost_full_temporal_cols[1:]:
            almost_full_dups['temporal_check'] += almost_full_dups[col]
        self.store_ids(almost_full_dups, 'temporal_check', "TEMPORAL", all_dups)

        leftovers = almost_full_dups[almost_full_dups['id'].isin(
            almost_full_dups[almost_full_dups.duplicated(subset=['temporal_check'], keep=False) == True]['id']) == False]


        almost_full_partials_cols = ['title_lower', 'description_lower', 'retrieval_date_lower']
        leftovers['partial_check'] = leftovers[almost_full_partials_cols[0]]
        for col in almost_full_partials_cols[1:]:
            leftovers['partial_check'] += leftovers[col]
        self.store_ids(leftovers, 'partial_check', "PARTIAL", all_dups)

        del title_desc_match, full_dups, almost_full_dups, leftovers


        # self.store_full_duplicates(self.find_full_dups(data, exact_dup_cols), exact_dup_cols, all_dups)

        # create df of all unique observations for future analysis (ie keep only one of each full duplicate)
        unique_data = self.create_unique_df(data, lower_title_desc) if row_num == 0 else \
            self.create_unique_df(data, lower_title_desc).head(row_num)
        del data

        print(f'unique data length is {len(unique_data)}')

        print('starting embedding')

        # create top scores from one embedding at a time, then gc.collect() to remove excess data from memory
        top_title_scores = self.convert_embedding_to_top_scores(
            data=unique_data['title_lower'].values.tolist(),
            new_scores=new_scores,
            info_type='title')
        gc.collect()

        print('done title embedding')

        top_desc_scores = self.convert_embedding_to_top_scores(
            data=unique_data['description_lower'].values.tolist(),
            new_scores=new_scores,
            info_type='desc')
        gc.collect()

        print('done desc embedding, now onto finding possible duplicates by score')

        # print unique scores (with data) and all possible duplicates that match the criteria in the function
        unique_data_scores, possible_dups = self.get_possible_duplicates(
            [top_title_scores, top_desc_scores],
            unique_data=unique_data,
            min_title_score=0.9,
            min_desc_score=0.9)

        unique_data_scores.to_csv(f'outputs/unique_data_w_scores_{row_num}.csv')

        possible_dups.to_csv(f'outputs/possible_dups.csv')

        # export all_dups list to csv for final results
        self.write_dups_to_csv(all_dups)


    @staticmethod
    @profile
    def set_up_data():
        # read in datafile
        with open("wi_dataset.csv", encoding='utf-8') as infile:
            all = list(csv.reader(infile))

        # create dataframe object with column names
        data = pd.DataFrame(all[1:])
        data.columns = all[0]

        return data

    @staticmethod
    @profile
    def store_ids(data_to_check, col_to_check, dup_type, container):
        for string in data_to_check[col_to_check].unique():
            matches = data_to_check[data_to_check[col_to_check] == string]

            ids = list(matches['id'].unique())

            for i in range(0, len(ids)):
                for j in range(i + 1, len(ids)):
                    row = [ids[i], ids[j], dup_type]
                    container.append(row)

    @staticmethod
    @profile
    def find_full_dups(all_data, dup_cols: list):
        for col in range(len(dup_cols)):
            all_data[f'{dup_cols[col]}_lower'] = all_data[dup_cols[col]].fillna('').apply(lambda x: x.lower())
            dup_cols[col] = f'{dup_cols[col]}_lower'
        full_dups = all_data[all_data.duplicated(subset=dup_cols, keep=False) == True].copy()

        full_dups.to_csv("outputs/full_duplicates.csv")
        print(f'the number of full duplicates is {len(full_dups)}')

        return full_dups

    @staticmethod
    @profile
    def store_full_duplicates(dup_list, dup_cols, container: list):
        dup_list['full_dup_string'] = dup_list[dup_cols[0]]
        for col in dup_cols[1:]:
            dup_list['full_dup_string'] += dup_list[col]

        for string in dup_list['full_dup_string'].unique():
            # create df of all the matches with a test_description duplicate of the current test_desc value
            matches = dup_list[dup_list['full_dup_string'] == string]

            # get list of ids of the duplicates
            dup_ids = list(matches['id'].unique())

            for i in range(0, len(dup_ids)):
                for j in range(i + 1, len(dup_ids)):
                    row = [dup_ids[i], dup_ids[j], "FULL"]
                    container.append(row)

    @staticmethod
    @profile
    def create_unique_df(all_data, dup_cols):
        # remove full duplicates from data, only keep unique observations for future duplicate identification
        unique = all_data[all_data.duplicated(subset=dup_cols, keep='first') == False].copy()
        all_data.drop(columns=[*dup_cols], inplace=True)

        return unique


    @staticmethod
    @profile
    def load_or_store_embeddings(strings_to_embed, obs: int, filename: str = 'details', length: int = 512):
        ### for storing / loading embeddings locally
        embed_chunksize = 9000
        embeddings_file = f'embeddings/cohere_{filename}_{obs}obs_{length}seq.pkl'
        if path.exists(embeddings_file):
            try:
                with open(embeddings_file, "rb") as infile:
                    return load(infile)['embeddings']
            except:
                print('memory error in embedding. retrying now')
                gc.collect()
                with open(embeddings_file, "rb") as infile:
                    return load(infile)['embeddings']
        else:
            co = cohere.Client("ho80SX8n3y7ANbv44gATQ61Zfe7KvnYoqqKSp6H5")
            if obs < embed_chunksize:
                return co.embed(texts=strings_to_embed).embeddings
            else:
                embedding = []
                chunk_val = embed_chunksize
                for curr_index in range(0, obs, chunk_val):
                    for _ in range(3):
                        try:
                            start_embedding = time.time()
                            end = curr_index + chunk_val if curr_index + chunk_val < obs else obs
                            print(f'currently embedding rows {curr_index} to {end} for the {filename}s')
                            partial = co.embed(texts=strings_to_embed[curr_index:end]).embeddings
                            embedding += partial
                            del partial
                            print(f'this embedding took {round(time.time() - start_embedding)} seconds')
                            break
                        except Exception as e:
                            print(e)
                            print("Retrying now after a 1-minute wait.")
                            time.sleep(60)
                    time.sleep(40)
            with open(embeddings_file, "wb") as outfile:
                dump({'embeddings': embedding}, outfile, protocol=HIGHEST_PROTOCOL)

        gc.collect()
        return embedding


    @staticmethod
    @profile
    def store_cos_sim_scores(embeddings, info_type: str, chunk_val: int = 3000):
        row_num = len(embeddings)
        top_scores_path = f"outputs/top_scores_{info_type}_{row_num}.h5"
        if row_num < chunk_val:
            scores = pd.DataFrame(util.cos_sim(embeddings, embeddings).numpy(),
                                  dtype='float32')

            # make diagonal of scores matrix 0s to avoid top score being compared with itself
            scores.values[range(scores.shape[0]), range(scores.shape[0])] = 0

            # get top 100 (or 10% of rows) scores for each row
            top_scores_num = min(100, row_num // 10) if row_num < chunk_val else 100 // (row_num // chunk_val)
            top_scores = pd.DataFrame(np.sort(scores, axis=1)[:, ::-1][:, :top_scores_num])
            top_scores_indices = pd.DataFrame(scores.columns.to_numpy()
                                              [np.argsort(scores.to_numpy(), axis=1)]
                                              [:, -1:-1 * (top_scores_num + 1):-1])
            for df in top_scores, top_scores_indices:
                df.columns = (df.columns.astype(int) + 1).astype(str)
            top_scores_indices = top_scores_indices.add_suffix("_index")
            top_scores = pd.concat([top_scores, top_scores_indices], axis=1)

        else:
            top_scores = pd.DataFrame()
            for curr_index in range(0, row_num, chunk_val):
                try:
                    end = curr_index + chunk_val if curr_index + chunk_val < row_num else row_num
                    print(f'calculating scores from row {curr_index} to row {end}')
                    start = time.time()
                    scores = pd.DataFrame(util.cos_sim(embeddings[curr_index:end], embeddings).numpy(),
                                   dtype='float32')
                except RuntimeError:
                    print('memory issue in scoring. retrying now')
                    gc.collect()
                    end = curr_index + chunk_val if curr_index + chunk_val < row_num else row_num
                    start = time.time()
                    scores = pd.DataFrame(util.cos_sim(embeddings[curr_index:end], embeddings).numpy(),
                                   dtype='float32')
                    print('it worked this time!')

                print(f'the scoring took {round((time.time() - start) / 60, 2)} mins...')

                # make diagonal of scores matrix 0s to avoid top score being compared with itself
                scores.values[range(scores.shape[0]), range(scores.shape[0])] = 0

                # get top 100 (or 10% of rows) scores for each row
                top_scores_num = min(100, row_num // 10) if row_num < chunk_val else 100 // (row_num // chunk_val)
                curr_top_scores = pd.DataFrame(np.sort(scores, axis=1)[:, ::-1][:, :top_scores_num])
                top_scores_indices = pd.DataFrame(scores.columns.to_numpy()
                                                  [np.argsort(scores.to_numpy(), axis=1)]
                                                  [:, -1:-1 * (top_scores_num + 1):-1])

                del scores

                for df in curr_top_scores, top_scores_indices:
                    df.columns = (df.columns.astype(int) + 1).astype(str)
                top_scores_indices = top_scores_indices.add_suffix("_index")
                curr_top_scores = pd.concat([curr_top_scores, top_scores_indices], axis=1)

                top_scores = pd.concat([top_scores, curr_top_scores])

        top_scores.to_hdf(top_scores_path, key='df', mode='w')

        return top_scores


    @profile
    def convert_embedding_to_top_scores(self, data, new_scores: bool, info_type: str):
        filename = f'outputs/top_scores_{info_type}_{len(data)}.h5'
        if new_scores or not path.exists(filename):
            print('making new scores')
            return self.store_cos_sim_scores(self.load_or_store_embeddings(data, obs=len(data), filename=info_type, length=512),
                                             info_type=info_type)
        else:
            print('reading in scores')
            return pd.read_hdf(filename)

    @staticmethod
    @profile
    def get_possible_duplicates(top_scores: list, unique_data, min_title_score=0.9, min_desc_score=0.9):
        # ensure indices align between dataframes, then combine by column
        top_scores[0] = top_scores[0].add_prefix('title_')
        top_scores[1] = top_scores[1].add_prefix('desc_')

        for df in top_scores:
            df.index = unique_data.index

        unique_data_scores = pd.concat([unique_data, top_scores[0], top_scores[1]], axis=1)

        title_cols = top_scores[0].columns
        desc_cols = top_scores[1].columns

        possible_dups = unique_data_scores[
            (((unique_data_scores[title_cols] < 1) & (unique_data_scores[title_cols] > min_title_score)).any(axis=1) &
             ((unique_data_scores[desc_cols] < 1) & (unique_data_scores[desc_cols] > min_desc_score)).any(axis=1))]

        return unique_data_scores, possible_dups

    @staticmethod
    @profile
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
    d = Dedup(new_scores=False, row_num=200)
    print(f'{round((time.time() - start) / 60, 2)} mins required')
