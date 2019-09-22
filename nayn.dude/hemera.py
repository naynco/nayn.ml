import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from TurkishStemmer import TurkishStemmer
import string
from sklearn import model_selection, naive_bayes, svm
import nltk
from sklearn.externals import joblib
import os.path


class AutoTagging:
    def __init__(self):
        self.stop_words = set(stopwords.words('turkish'))
        self.data = pd.read_csv("https://raw.githubusercontent.com/naynco/nayn.data/master/classification_clean.csv")

    def download_nltk_models(self):
        download_list = ["stopwords","wordnet"]
        for download in download_list:
            nltk.download(download)

    def load_data_with_preprocessing(self):
        main_categories = ['DÜNYA', 'SPOR','SANAT','Teknoloji']
        filter = self.data["Categories"].isin(main_categories)
        data = self.data[filter]

        return data

    def cleaning(self,doc):
        stemmer = TurkishStemmer()
        doc = doc.lower()
        filter_punch = str.maketrans('', '', string.punctuation)
        stripped = doc.translate(filter_punch)

        clean_text = []
        for i in stripped.split():
            if i not in self.stop_words:
                clean_text.append(stemmer.stem(i))

        return ' '.join(clean_text)

    def model(self):
        txt_list = []
        data = self.load_data_with_preprocessing()
        for doc in data["Title"]:
            txt_list.append(self.cleaning(doc))


        data["Clean"] = txt_list

        Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(data['Clean'],
                                                                            data['Categories'],
                                                                            test_size=0.3)


        #### TF-IDF ####
        Tfidf_vect = TfidfVectorizer(max_features=5000)
        Tfidf_vect.fit(data['Clean'])
        Train_X_Tfidf = Tfidf_vect.transform(Train_X.ravel())
        Test_X_Tfidf = Tfidf_vect.transform(Test_X.ravel())
        joblib.dump(Tfidf_vect, open("vectorizer.pickle", 'wb'))

        #### Naive Bayes ####
        self.naive_bayes(Train_X_Tfidf, Train_Y,Test_Y,Test_X_Tfidf)

        #### SVM #####
        self.svm(Train_X_Tfidf, Train_Y,Test_X_Tfidf,Test_Y)

    def naive_bayes(self,Train_X_Tfidf, Train_Y,Test_Y,Test_X_Tfidf):
        Naive = naive_bayes.MultinomialNB()
        Naive.fit(Train_X_Tfidf, Train_Y)
        predictions_NB = Naive.predict(Test_X_Tfidf)
        print("Naive Bayes Accuracy:  ", accuracy_score(predictions_NB, Test_Y) * 100)
        filename = 'predictions_NB.pkl'
        joblib.dump(Naive, open(filename, 'wb'))

    def svm(self,Train_X_Tfidf, Train_Y,Test_X_Tfidf,Test_Y):
        SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
        SVM.fit(Train_X_Tfidf, Train_Y)

        predictions_SVM = SVM.predict(Test_X_Tfidf)
        print("SVM Accuracy: ", accuracy_score(predictions_SVM, Test_Y) * 100)
        filename = 'predictions_SVM.pkl'
        joblib.dump(SVM, open(filename, 'wb'))

    def load_model(self,news_title):
        if os.path.isfile("predictions_NB.model"):
            # You can use predictions_NB or predictions_SVM just change this part 'predictions_NB.pkl'
            model_file = open("predictions_NB.pkl", 'rb')
            loaded_model = joblib.load(model_file)

            vectorizer_file = open("vectorizer.pickle", 'rb')
            vectors = joblib.load(vectorizer_file)

            result = loaded_model.predict(vectors.transform([news_title]))
            return result
        else:
            print("Model train is running!")
            self.model()

if __name__ == '__main__':
    tag = AutoTagging().load_model(news_title="Fenerbahçe, Neustadter Transferini Borsaya Bildirdi")
    print(tag)