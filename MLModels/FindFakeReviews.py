
with open('./FakeReviewers_Vader.txt') as reviewers:
    fakeReviewers = reviewers.readlines()
reviewers.close()

fakeReviewerList = []
for line in fakeReviewers:
    fakeReviewerList.append(line.strip('\n'))
print(len(fakeReviewerList))

with open('./UserWiseReviews.txt') as reviewFile:
    reviews = reviewFile.readlines()
reviewFile.close()

allReviews = []
with open('./LDAFakeReviews.txt', 'w') as output:
    for line in reviews:
        reviewSplit = line.split("-")
        reviewerId = reviewSplit[0]

        if reviewerId in fakeReviewerList:
            reviewList = reviewSplit[1].strip().split("%#%")
            output.write(reviewerId + ' - \n')
            for review in reviewList:
                if review != "":
                    allReviews.append(review)
                    review = '[' + review + ']\n'
                    output.write(review)
output.close()


import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

print(len(allReviews))
vectorizer = TfidfVectorizer()
tfidf_vector = vectorizer.fit_transform(allReviews)
feature_names = vectorizer.get_feature_names_out()
tfidf_matrix = tfidf_vector.toarray()
total_tfidf_scores = np.sum(tfidf_matrix, axis=0)
vocab_comp = zip(feature_names, total_tfidf_scores)
sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:300]
imp_words_topic = ""
for word in sorted_words:
    imp_words_topic=imp_words_topic+" "+word[0]

wordcloud = WordCloud(width=600, height=400).generate(imp_words_topic)
plt.figure( figsize=(5,5))
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout()
plt.title("TextBlob WordCloud")
plt.show()

import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

reviewDict = {}
verified_count = 0
not_verified_count = 0
product_chunks = pd.read_json('/Users/durgamuralidharan/Desktop/Masters/Assignments/CS267_BigData_with_ML/Project/Dataset/Small/Home_and_Kitchen_5.json', lines=True, chunksize=500)
for c in  product_chunks:
    for index, row in c.iterrows():
        if row["verified"]:
            verified_count = verified_count + 1
        else:
            not_verified_count = not_verified_count + 1

print("Verified = ", verified_count)
print(verified_count*100/len(allReviews))
print("Not verified = ", not_verified_count)
print(not_verified_count*100/len(allReviews))
data = [verified_count*100/len(allReviews), not_verified_count*100/len(allReviews)]
labels = ['Verified', 'Not Verified']
sns.set_style("whitegrid")
plt.figure(figsize=(6,6))
plt.pie(data, labels=labels, autopct='%1.1f%%') # Create pie chart
plt.show()
