import PreProcess
from flair.data import Sentence
from flair.nn import Classifier

process_reviews = PreProcess.PreProcess()
userReviewsDict = process_reviews.preprocess('./UserWiseReviews.txt', 5)

# Flair sentiment analysis

# load the NER tagger
tagger = Classifier.load('sentiment')
flair_sentiment_score = dict()
for key, value in userReviewsDict.items():
    scoreList = []
    for review in value[0]:
        if(len(review) < 1):
            continue
        sentence = Sentence(review)
        tagger.predict(sentence)
        print(sentence)
        labels = str(sentence.labels[0]).split(' ')
        score = labels[len(labels)-1]
        sentiment = labels[len(labels)-2]
        score = score[1:]
        score = score[:-1]
        if(sentiment == 'NEGATIVE'):
            score = -(float(score))
        else:
            score = float(score)
        scoreList.append(score)

    flair_sentiment_score[key] = mean(scoreList) * 100

for key, value in flair_sentiment_score.items():
    if(value > 95):
        print(key, " = ", value)