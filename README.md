# Mess of a Deduplication Challenge
IDEAS:
1) If we can fix the cos_sim_scores from sentence-transformer and balance for length somehow, we may be on to something. When testing on sentences of similar length, in diff languages, it is able to estimate similarities.
   - WITH THE DEFAULT MAX EMBEDDING SEQUENCE LENGTH, IT WORKS! BUT WITH THE EXTENDED MAX SEQUENCE LENGTH OF 512 (FOR LONG TEXTS), IT DOES NOT WORK AS WELL... Look at the cos_sim_scores.csv for diff seq numbers (indicating diff seq lengths) to see the inflated numbers. I think we should try the below options
   - What if we split the dataset into those with extremely long, medium, and short descriptions then compare within those? Using the assumption that ads in each category are not duplicates. Could even have them overlap (i.e. small is below 100 words, medium is between 80 - 300 words, large is 275 - 400 words and XL is 375+ words?) We would look at distr of total words per description to make the groups.
   - What if we split each description into parts, and compare the similarity of each part to every other part of a description (instead of doing whole vs whole description) and take the average similarity? That way we aren't comparing 1000 words vs 100, but averaging 10 comparisons of 100 words vs 100... Could be much more effective?
   - IT IS WORKING FINE! Could still try the above ideas to see if they help but right now similarity metrics are very effective
3) Translating all the data to a common language -- I am not sure translating is the right path because of the hours it'll take to run the script. When the evaluators run the script on the test data, it may take too long to be feasible. It also says in the competition description the importance of being able to manage crosslingual data.
   - What is the fastest way to translate them all? I have tried googletrans module but it was too low and had too many limitations. Maybe with Text2Text?
   - If we are able to translate them all, can we then create vectors of word counts / frequencies to compare job titles + descriptions?
3) Are there different models / encoding we should be trying?
   - Cohere is our best option --- others I tried were XLM Roberta (tried, maybe not properly), BERT, Text2Text
4) Tried / trying to remove frequent/meaningless words to see how performance changes and to see if it reduces the bias towards longer sentences, which will naturally have more of the frequent/meaningless words.
   - Made a small difference, need to re-implement
5) Use the title as an initial query for all descriptions, then if there's a high similarity match then compare descripition vs description?
   - Titles aren't even similar to their own job descriptions... -- see titles_descriptions_cos_sim_scores.csv for chart of first 50 titles (each row is for a title) and first 100 descriptions (each column is for a description). Most titles are not related to their own descriptions, so don't think they're a good predictor of similarity for other ads

INSTALLATION:
1) Download the data (available from the competition website in 'Participate') -- it is called wi_dataset.csv. Save it to the same folder as the script.

2) Install all required packages based on the imports at the top of the code.

3) After using, I'd suggest making a folder called 'transformers' in the same folder as the code, where you can locally store models that we (may) use for multilingual analysis instead of re-downloading each time.

4) All the existing .csv files are outputs from past/current analysis of the data:
- cos_sim_scores -- similarity score matrix of first 100 job descriptions -- longer texts have higher similarities, this is the main issue
- duplicates -- format and file for the output of duplicates from the code that we will submit to the competition
- lang_detects -- identified language for every observation, stored locally to not have to re-run it each time
- similarities -- test run of 10 obs adding 500 "." to each description to balance for length... Didn't work. Last columns are the similarity scores and the index for each score, across languages.
- test_output -- same as similarities, but for 100 observations. Longer texts naturally have higher scores... 
- para_mining_scores -- tried using the paragraph mining function from sentencetransformers.util, but didn't make any diff - long sentences still were deemed highly similar

Run the script using run_num=200 and new_scores=True parameters for Dedup to create copies of the embeddings and scores using only 200 rows for a quick test. You can remove the row_num parameter and it'll run all rows -- it will take a while, but it will work and will print progress updates for the longest parts. You can see the final outputs file after running it in the unique_data_scores.csv (unique rows with their top similarity scores) file and in the possible_dups.csv (those with over 0.90 similarity) file.
