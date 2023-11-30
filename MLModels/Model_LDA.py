import numpy as np
import PreProcess
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from statistics import mean
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt

process_reviews = PreProcess.PreProcess()
productReviewsDict = process_reviews.preprocess("./UserWiseReviews.txt", 5)

def draw_word_cloud(index):
  imp_words_topic=""
  vocab_comp = zip(corpus, lda.components_[index])
  sorted_words = sorted(vocab_comp, key = lambda x:x[1], reverse=True)[:100]
  for word in sorted_words:
    imp_words_topic=imp_words_topic+" "+word[0]
    print(word)
    print(word[0])

  wordcloud = WordCloud(width=600, height=400).generate(imp_words_topic)
  plt.figure( figsize=(5,5))
  plt.imshow(wordcloud)
  plt.axis("off")
  plt.tight_layout()
  plt.show()


vectorizer = TfidfVectorizer()

num_topics = 10
lda = LatentDirichletAllocation(n_components=num_topics, learning_method='online', random_state=42, max_iter=1)
#word_cloud_corpus

for key, value in productReviewsDict.items():
    blank_reviews = 0
    for index in range(len(value[0])):
        if(value[0][index] == ""):
            blank_reviews = blank_reviews + 1
    
    if(blank_reviews < len(value[0])):
        review_vectors = vectorizer.fit_transform(value[0])
        idf = vectorizer.idf_
        topic_dict = dict(zip(vectorizer.get_feature_names_out(), idf))

    # Fit the model to the data
    lda.fit(review_vectors)

    # How well model predicts the words in the dataset
    perplexity = lda.perplexity(review_vectors)
    
    # Calculate topic similarity
    topic_similarity = []
    for i in range(num_topics):
        for j in range(i + 1, 10):
            topic_1 = lda.components_[i].reshape(1, -1)
            topic_2 = lda.components_[j].reshape(1, -1)
            topic_similarity.append(cosine_similarity(topic_1, topic_2))

    average_topic_similarity = np.mean(topic_similarity)

    corpus = vectorizer.get_feature_names_out()
    topic_words = set()

    for index, component in enumerate(lda.components_):
        vocab_comp = zip(corpus, component)
        sorted_words = sorted(vocab_comp, key = lambda x:x[1], reverse=True)[:15]

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
        print(key, " = ", mean_percentage, ", ", perplexity, ", ", average_topic_similarity)
        #if(key == "A240YCM012LJSO"):
        #    draw_word_cloud(3)