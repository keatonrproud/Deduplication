# EU_Deduplication_Challenge
IDEAS:
1) If we can fix the cos_sim_scores from sentence-transformer and balance for length somehow, we may be on to something. When testing on sentences of similar length, in diff languages, it is able to estimate similarities.
   - If you view the dummy_cos_sim_scores.csv, you will see 5 strings at the bottom and a matrix of similarity scores. The third and fifth strings are not related whatsoever, and appropriately have a negative similarity score. HOWEVER, the fourth sentence is the same as the fifth sentence repeated over and over to have similar length as the fifth and then their similarity score goes up to a ~0.250. Still not a significant similarity, but significantly inflated from before.
   - What if we split the dataset into those with extremely long, medium, and short descriptions then compare within those? Using the assumption that ads in each category are not duplicates. Could even have them overlap (i.e. small is below 100 words, medium is between 80 - 300 words, large is 275 - 400 words and XL is 375+ words?) We would look at distr of total words per description to make the groups.
   - What if we split each description into parts, and compare the similarity of each part to every other part of a description (instead of doing whole vs whole description) and take the average similarity? That way we aren't comparing 1000 words vs 100, but averaging 10 comparisons of 100 words vs 100... Could be much more effective?
2) Translating all the data to a common language -- I am not sure translating is the right path because of the hours it'll take to run the script. When the evaluators run the script on the test data, it may take too long to be feasible. It also says in the competition description the importance of being able to manage crosslingual data.
   - If we are able to translate them all, can we then create vectors of word counts / frequencies to compare job titles + descriptions?
3) Are there different models / encoding we should be trying?
   - XLM Roberta (tried, maybe not properly), BERT, Text2Text
4) Tried / trying to remove frequent/meaningless words to see how performance changes and to see if it reduces the bias towards longer sentences, which will naturally have more of the frequent/meaningless words.

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
