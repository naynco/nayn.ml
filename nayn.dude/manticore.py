import pandas as pd
from nltk.corpus import stopwords
from TurkishStemmer import TurkishStemmer
import string
import gensim
from nltk.tokenize import word_tokenize
import os

class CorpusSimilarity:
    def __init__(self):
        pass

    def load_data(self):
        df = pd.read_csv("https://raw.githubusercontent.com/naynco/nayn.data/master/nayn.co-2016.csv")
        df = df.dropna().reset_index(drop=True)
        return df

    def cleaning(self,doc):
        stop_words = set(stopwords.words('turkish'))
        stemmer = TurkishStemmer()
        doc = doc.lower()
        filter_punch = str.maketrans('', '', string.punctuation)
        stripped = doc.translate(filter_punch)

        clean_text = []
        for i in stripped.split():
            if i not in stop_words:
                clean_text.append(stemmer.stem(i))

        return ' '.join(clean_text)


    def model(self):
        df = self.load_data()
        raw_documents = list(df["Content"].values)
        print("Number of documents:", len(raw_documents))

        gen_docs = []
        for text in raw_documents:
            word = []
            for w in word_tokenize(self.cleaning(text)):
                word.append(w.lower())

            gen_docs.append(word)

        dictionary = gensim.corpora.Dictionary(gen_docs)
        dictionary.save_as_text("dictionary")

        corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
        gensim.corpora.MmCorpus.serialize('corpus.mm', corpus)

        tf_idf = gensim.models.TfidfModel(corpus)
        tf_idf.save("tfidf.model")

        sims = gensim.similarities.Similarity(os.getcwd(), tf_idf[corpus],
                                              num_features=len(dictionary))

        return sims


    def load_and_predict(self,news):
        if os.path.isfile("tfidf.model"):
            df = self.load_data()
            tf_idf = gensim.models.TfidfModel.load("tfidf.model")
            dictionary = gensim.corpora.Dictionary.load_from_text("dictionary")
            corpus = gensim.corpora.MmCorpus('corpus.mm')

            sims = gensim.similarities.Similarity(os.getcwd(),tf_idf[corpus],
                                                  num_features=len(dictionary))

            query_doc = [w.lower() for w in word_tokenize(self.cleaning(news))]

            query_doc_bow = dictionary.doc2bow(query_doc)
            query_doc_tf_idf = tf_idf[query_doc_bow]
            df_similarty = pd.DataFrame({"Similarty":sims[query_doc_tf_idf]})

            result = pd.concat([df, df_similarty], axis=1, join='inner')
            dfx = result.sort_values("Similarty",ascending=False)[:5]
            print(dfx["Content"])
        else:
            print("Model train is running!")
            self.model()

news = "Katif kentinde nüfusun çoğunluğunu Şii'ler oluşturuyor. Daha önce bölgede Suudi güvenlik güçlerini hedef alan saldırdı"
print(CorpusSimilarity().load_and_predict(news))
