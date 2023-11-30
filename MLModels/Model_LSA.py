import numpy as np
import PreProcess
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from statistics import mean
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from tmtoolkit.topicmod.evaluate import metric_coherence_gensim

process_reviews = PreProcess.PreProcess()
productReviewsDict = process_reviews.preprocess("./UserWiseReviews.txt", 40)


vectorizer = TfidfVectorizer()

num_topics = 2
lsa = TruncatedSVD(n_components=num_topics, algorithm='randomized', random_state=42, n_iter=10)

for key, value in productReviewsDict.items():
    #blank_reviews = 0
    for review in value[0]:
        if(review == ""):
            #blank_reviews = blank_reviews + 1
            value[0].remove(review)

    if(len(value[0]) > 0):
        #print(value[0])
        review_vectors = vectorizer.fit_transform(value[0])
        idf = vectorizer.idf_
        topic_dict = dict(zip(vectorizer.get_feature_names_out(), idf))

    # Fit the model to the data
    topics = lsa.fit_transform(review_vectors)


    # Calculate topic similarity
    topic_similarity = []
    for i in range(num_topics):
        for j in range(i + 1, 10):
            topic_1 = lsa.components_[i].reshape(1, -1)
            topic_2 = lsa.components_[j].reshape(1, -1)
            topic_similarity.append(cosine_similarity(topic_1, topic_2))

    average_topic_similarity = np.mean(topic_similarity)

    corpus = vectorizer.get_feature_names_out()
    topic_words = set()

    for index, component in enumerate(lsa.components_):
        corpus_comp = zip(corpus, component)
        sorted_words = sorted(corpus_comp, key = lambda x:x[1], reverse=True)[:15]

        for t in sorted_words:
            topic_words.add(t[0])

    topic_percentage = []
    for review in value[0]:
        count = 0
        word_tokens = word_tokenize(review)
        for word in word_tokens:
            if word in topic_words:
                count += 1

        topic_percentage.append(count/len(topic_words)*100)

    mean_percentage = mean(topic_percentage)
    if(mean_percentage > 95):
        print(key, " = ", mean_percentage)
        print("Average topic similarity:", average_topic_similarity)
