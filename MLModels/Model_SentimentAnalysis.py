import PreProcess
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from statistics import mean 

process_reviews = PreProcess.PreProcess()
userReviewsDict = process_reviews.preprocess('./UserWiseReviews.txt', 5)

# TextBlob sentiment score
textblob_sentiment_score = dict()
for key, value in userReviewsDict.items():
    scoreList = []
    for review in value[0]:
        scoreList.append(TextBlob(review).sentiment.polarity)

    textblob_sentiment_score[key] = mean(scoreList) * 100

for key, value in textblob_sentiment_score.items():
    if(value > 95):
        print(key, " = ", value)


print("\n")

# nltk Vader sentiment score
analyser = SentimentIntensityAnalyzer()
vader_sentiment_score = dict()
for key, value in userReviewsDict.items():
    scoreList = []
    for review in value[0]:
        scoreList.append(analyser.polarity_scores(review))

    negative_score = []
    neutral_score = []
    positive_score = []
    for index in range(len(scoreList)):
        score = scoreList[index]
        negative_score.append(score['neg'])
        neutral_score.append(score['neu'])
        positive_score.append(score['pos'])

    average_negative_score = mean(negative_score)*100
    average_neutral_score = mean(neutral_score)*100
    average_positive_score = mean(positive_score)*100

    vader_sentiment_score[key] = {('neg', average_negative_score), ('neu', average_neutral_score), ('pos', average_positive_score)}

    if(average_positive_score > 95):
        print(key, " = ", average_positive_score)

