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

# +
row_num = 0
new_scores = True

# read in datafile
with open("wi_dataset.csv", encoding='utf-8') as infile:
    all = list(csv.reader(infile))

# create dataframe object with column names
data = pd.DataFrame(all[1:])
data.columns = all[0]

data

# +
all_dups = []

exact_dup_cols = ['title', 'description', 'location', 'country_id', 'company_name', 'retrieval_date']

for col in range(len(exact_dup_cols)):
    data[f'{exact_dup_cols[col]}_lower'] = data[exact_dup_cols[col]].fillna('').apply(lambda x: x.lower())
    exact_dup_cols[col] = f'{exact_dup_cols[col]}_lower'

title_desc_match = data[data.duplicated(subset=['title', 'description'], keep=False) == True].copy()

title_desc_match


# +
full_dups = title_desc_match[title_desc_match.duplicated(subset=exact_dup_cols, keep=False) == True].copy()

almost_full_dups = title_desc_match[title_desc_match.duplicated(subset=exact_dup_cols, keep=False) == False].copy()

full_dups.to_csv("outputs/full_duplicates.csv")
almost_full_dups.to_csv('outputs/almost_full_dups.csv')

full_dups.sort_values(by='title')
# -

almost_full_dups.sort_values(by='title')

# +
full_dups['full_check'] = full_dups[exact_dup_cols[0]]

full_dups_list = []

for col in exact_dup_cols:
    full_dups['full_check'] += full_dups[col]

    
for string in full_dups['full_check'].unique():
    matches = full_dups[full_dups['full_check'] == string]
    
    ids = list(matches['id'].unique())

    for i in range(0, len(ids)):
        for j in range(i + 1, len(ids)):
            row = [ids[i], ids[j], "FULL"]
            full_dups_list.append(row)
            
full_dups_list

# +
almost_full_temporal_cols = ['title_lower', 'description_lower', 'location_lower', 'country_id_lower', 'company_name_lower']

almost_full_temporals_list = []

almost_full_dups['temporal_check'] = almost_full_dups[almost_full_temporal_cols[0]]
for col in almost_full_temporal_cols[1:]:
    almost_full_dups['temporal_check'] += almost_full_dups[col]

    
for string in almost_full_dups['temporal_check'].unique():
    matches = almost_full_dups[almost_full_dups['temporal_check'] == string]
        
    ids = list(matches['id'].unique())
    
    for i in range(0, len(ids)):
        for j in range(i + 1, len(ids)):
            row = [ids[i], ids[j], "TEMPORAL"]
            almost_full_temporals_list.append(row)
            
temporals = almost_full_dups[almost_full_dups.duplicated(subset=['temporal_check'], keep=False) == True]
            
almost_full_temporals_list

# +
count_temporal_ids = {i[0] for i in almost_full_temporals_list}

print(len(count_temporal_ids))
# -

# remove temporal matches from almost_full_dups, as temporal is more specific than partial
leftovers = almost_full_dups[almost_full_dups['id'].isin(temporals['id']) == False]

# +
# location, country, or company name is different
almost_full_partials_cols = ['title_lower', 'description_lower', 'retrieval_date_lower']

almost_full_partials_list = []

leftovers['partial_check'] = leftovers[almost_full_partials_cols[0]]

for col in almost_full_partials_cols[1:]:
    leftovers['partial_check'] += leftovers[col]

for string in leftovers['partial_check'].unique():
    matches = leftovers[leftovers['partial_check'] == string]

    ids = list(matches['id'].unique())

    for i in range(0, len(ids)):
        for j in range(i + 1, len(ids)):
            row = [ids[i], ids[j], "PARTIAL"]
            almost_full_partials_list.append(row)
    
partials = leftovers[leftovers.duplicated(subset=['partial_check'], keep=False) == True]

partials.sort_values(by='title')
# -

# ###### get remaining observations that have matching title + description, but have no match for retrieval date or some combination of location / country / company name

# +
removed_ids = list(temporals['id'].unique()) + list(partials['id'].unique())

remaining = almost_full_dups[almost_full_dups['id'].isin(removed_ids) == False]

remaining

# +
print(f'all w/ matching title + description = {len(title_desc_match)}')

print('-------------')

print(f'full duplicates = {len(full_dups)}')

print(f'almost-full duplicates = {len(almost_full_dups)}')

print('-------------')

print(f'temporals = {len(temporals)}')

print(f'partials = {len(partials)}')

print(f'remaining = {len(remaining)}')

print(f'combined total of temporals, partials, and remaining = {len(temporals) + len(partials) + len(remaining)}')

# +
# check which rows are in temporals and partials

temporals_ids = list(temporals['id'])
remaining_ids = list(remaining['id'])
partials_ids = list(partials['id'])

dups = []
for id in partials_ids:
    if id in temporals_ids:
        dups.append(id)
    
temporals[temporals['id'].isin(dups)]
# -

with open('exact_dups_version_output.csv', 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    for item in (full_dups_list, almost_full_temporals_list, almost_full_partials_list):
        for dup in item:
            writer.writerow(dup) 


