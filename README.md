# EU_Deduplication_Challenge
 
1) Download the data (available from the competition website in 'Participate') -- it is called wi_dataset.csv. Save it to the same folder as the script.

2) Install all required packages based on the imports at the top of the code.

3) After using, I'd suggest making a folder called 'transformers' in the same folder as the code, where you can locally store models that we (may) use for multilingual analysis instead of re-downloading each time.

4) All the existing .csv files are outputs from past/current analysis of the data:
- cos_sim_scores -- similarity score matrix of first 100 job descriptions -- longer texts have higher similarities, this is the main issue
- duplicates -- format and file for the output of duplicates from the code that we will submit to the competition
- lang_detects -- identified language for every observation, stored locally to not have to re-run it each time
- similarities -- test run of 10 obs adding 500 "." to each description to balance for length... Didn't work. Last columns are the similarity scores and the index for each score, across languages.
- test_output -- same as similarities, but for 100 observations. Longer texts naturally have higher scores...

5) If we can fix the cos_sim_scores and balance for length somehow, we may be on to something. I am not sure translating each is the right path because of the hours it'll take to run the script. When the evaluators run the script on the test data, it may take too long to be feasible.