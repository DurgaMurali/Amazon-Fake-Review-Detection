import PreProcess
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from statistics import mean

process_reviews = PreProcess.PreProcess()
#productReviewsDict = process_reviews.preprocess("./ProductWiseReviews.txt", 20)
productReviewsDict = process_reviews.preprocess("./UserWiseReviews.txt", 25)

#product_example = dict()
#product_example['7229002036'] = productReviewsDict['7229002036']
#product_example['A14ZUWQKAZUWLX'] = productReviewsDict['A14ZUWQKAZUWLX']

vectorizer = TfidfVectorizer()

num_topics = 10
lda = LatentDirichletAllocation(n_components=num_topics, learning_method='online', random_state=42, max_iter=1)

for key, value in productReviewsDict.items():
    review_vectors = vectorizer.fit_transform(value[0])
    idf = vectorizer.idf_
    topic_dict = dict(zip(vectorizer.get_feature_names_out(), idf))

    # Fit the model to the data
    lda.fit(review_vectors)

    corpus = vectorizer.get_feature_names_out()
    topic_words = set()

    for i, comp in enumerate(lda.components_):
        vocab_comp = zip(corpus, comp)
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
        print(key, " = ", mean_percentage)
