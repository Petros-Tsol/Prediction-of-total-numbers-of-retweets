# Prediction of total numbers of retweets
Implementation of some algorithms to predict the total number of retweets of a tweet.

## Description
A MATLAB implementation for the prediction of the total number of retweets. The algorithms that were implemented are based on the following papers:
- Gabor Szabo and Bernardo A Huberman.“Predicting the popularity of online content”. In:Communications of the ACM 53.8 (2010), pp. 80–88.
- Shuai Gao, Jun Ma, and Zhumin Chen. “Modeling and predicting retweeting dynamics on microblogging platforms”. In: Proceedings of the Eighth ACM International Conference on Web Search and Data Mining. 2015, pp. 107–116.
- Ryota Kobayashi and Renaud Lambiotte. “Tideh: Time-dependent hawkes process for predicting retweet dynamics”. In: Tenth International AAAI Conference on Web and Social Media. 2016.

## Data
We get the data from http://snap.stanford.edu/seismic/#data.

The test file must be in the an appropriate format like:
- first line \<total number of retweets including the source tweet\> \<average number of followers of the original author and the users who retweeted\>
- second line 0 \<total number of followers of the author of the tweet\>
- third line \<seconds passed to the first retweet\> \<total number of followers of the first user who retweeted the source tweet\>
- forth line \<seconds passed to the second retweet\> \<total number of followers of the second user who retweeted the source tweet\>
- etc...

## Execution
You can set the indicator time (training time up to we can read the data) and the reference time (prediction time up to we want to predict the total number of retweets) in the function retweet_prediction().

We used the mean absolute percentage error to test the performance of the models.

Execute the program by entering retweet_prediction().

## Extra
The code is part of my master thesis on estimation of information diffusion in social media. You can find the complete text here: https://drive.google.com/file/d/1MbrujHhDZGHzuWxzjDyvC2W2jWINLtBQ/ (in Greek).
