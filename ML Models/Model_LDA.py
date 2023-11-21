import PreProcess
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize 

process_reviews = PreProcess.PreProcess()
productReviewsDict = process_reviews.preprocess()
#print(productReviewsDict['7229002036'])
#product_example = dict()
#product_example['7229002036'] = productReviewsDict['7229002036']

vectorizer = TfidfVectorizer()

num_topics = 8
lda = LatentDirichletAllocation(n_components=num_topics, learning_method='online', random_state=42, max_iter=1)

for key, value in productReviewsDict.items():
    print(type(value))
    print(value[0])
    review_vectors = vectorizer.fit_transform(value[0])
    idf = vectorizer.idf_
    topic_dict = dict(zip(vectorizer.get_feature_names_out(), idf))
    #print(topic_dict)

    # Fit the model to the data
    lda.fit(review_vectors)

    corpus = vectorizer.get_feature_names_out()
    topic_words = set()

    for i, comp in enumerate(lda.components_):
        vocab_comp = zip(corpus, comp)
        sorted_words = sorted(vocab_comp, key = lambda x:x[1], reverse=True)[:10]
        #print("Topic " + str(i) + ": ")
        for t in sorted_words:
            #print(t[0],end=" ")
            topic_words.add(t[0])
        #print("\n")

    topic_percentage = []
    for review in value[0]:
        count = 0
        word_tokens = word_tokenize(review)
        for word in word_tokens:
            if word in topic_words:
                count += 1

        topic_percentage.append(count/len(topic_words)*100)
    print(topic_percentage)
