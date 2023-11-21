import pandas as pd
import contractions # pip install contractions to fix contractions like you're to you are
import re
import unicodedata

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag


class PreProcess:

    def remove_non_ascii_words(self, text_data, suicide_text_removed_non_ascii):

        for text in text_data:
            ascii_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            suicide_text_removed_non_ascii.append(ascii_text)


    def remove_long_numbers(self, text_data, suicide_text_removed_long_numbers):
        for text in text_data:
            text = re.sub(r'[0-9]{10,}', '', text)
            suicide_text_removed_long_numbers.append(text)


    def fix_contractions(self, text_data, suicide_text_fixed_contractions):
        contractions_fixed = []
        for text in text_data:
            sentences = sent_tokenize(text)
            contractions_fixed.clear()
            for line in sentences:
                contractions_fixed.append(contractions.fix(line))
            suicide_text_fixed_contractions.append("".join(contractions_fixed))


    def convert_to_lower_case(self, text_data, suicide_text_lower):
        for text in text_data:
            suicide_text_lower.append(text.lower())


    def remove_punctuation(self, text_data, suicide_text_no_punctuation):
        for text in text_data:
            text = re.sub(r'[^\w\s]', ' ', text)
            suicide_text_no_punctuation.append(text)


    def tokenize_text(self, text_data, suicide_tokenize_text):
        for text in text_data:
            sentences = sent_tokenize(text)

            sentence_tokens = []
            for line in sentences:
                tokens = (word_tokenize(line))
                sentence_tokens.extend(tokens)

            suicide_tokenize_text.append(sentence_tokens)


    def remove_stop_words(self, text_data, suicide_removed_stop_words):
        stop_words=set(stopwords.words('english'))
        for text in text_data:
            post = []
            for word in text:
                if word not in stop_words:
                    post.append(word)

            suicide_removed_stop_words.append(post)


    def lemmatize_verbs(self, text_data, suicide_text_lemmatized):
        lemmatizer = WordNetLemmatizer()
        for text in text_data:
            pos_dict = pos_tag(text)
            #print(pos_dict)
            post = []
            for tag in pos_dict:
                pos_identifier = ""
                tag_identifier = tag[1]
                word = tag[0]

                match tag_identifier:
                    case 'NN': pos_identifier = 'n'
                    case 'NNS': pos_identifier = 'n'
                    case 'NNP': pos_identifier = 'n'
                    case 'NNPS': pos_identifier = 'n'
                    
                    case 'VB': pos_identifier = 'v'
                    case 'VBG': pos_identifier = 'v'
                    case 'VBD': pos_identifier = 'v'

                    case 'RB': pos_identifier = 'r'
                    case 'RBR': pos_identifier = 'r'
                    case 'RBS': pos_identifier = 'r'

                    case 'JJ': pos_identifier = 'a'
                    case 'JJR': pos_identifier = 'a'
                    case 'JJS': pos_identifier = 'a'
            
                if(pos_identifier != ""):
                    lemmatized_word = lemmatizer.lemmatize(word, pos=pos_identifier)
                else:
                    lemmatized_word = word

                post.append(lemmatized_word)

            suicide_text_lemmatized.append(" ".join(post))
    

    def preprocess_text(self, productReviewsDict):
        for key, value in productReviewsDict.items():
            productReviews = []

            review_text_removed_non_ascii = []
            self.remove_non_ascii_words(value, review_text_removed_non_ascii)

            review_text_removed_long_numbers = []
            self.remove_long_numbers(review_text_removed_non_ascii, review_text_removed_long_numbers)

            review_text_fixed_contractions = []
            self.fix_contractions(review_text_removed_long_numbers, review_text_fixed_contractions)

            review_text_lower = []
            self.convert_to_lower_case(review_text_fixed_contractions, review_text_lower)

            review_text_no_punctuation = []
            self.remove_punctuation(review_text_lower, review_text_no_punctuation)

            # Tokenize words
            review_tokenize_text = []
            self.tokenize_text(review_text_no_punctuation, review_tokenize_text)

            review_removed_stop_words = []
            self.remove_stop_words(review_tokenize_text, review_removed_stop_words)

            review_text_lemmatized = []
            self.lemmatize_verbs(review_removed_stop_words, review_text_lemmatized)

            productReviews.append(review_text_lemmatized)

            productReviewsDict[key] = productReviews


    def readReviewsandFilter(self):
        with open("./ProductWiseReviews.txt") as reviewFile:
            productReviews = reviewFile.readlines()

        productReviewsDict = {}
        for line in productReviews:
            reviewSplit = line.split("-")
            productId = reviewSplit[0]
            reviewList = reviewSplit[1].strip().split("%#%")

            # Consider products that have more than 20 reviews
            if(len(reviewList) > 20):
                productReviewsDict[productId] = reviewList

        reviewFile.close()
        return productReviewsDict

    def preprocess(self):
        productReviewsDict =  self.readReviewsandFilter()  
        print(len(productReviewsDict))

        #print(productReviewsDict['7229002036'])
        self.preprocess_text(productReviewsDict)
        return productReviewsDict
