from gensim.models.phrases import Phrases, Phraser
from TurkishStemmer import TurkishStemmer
from gensim.models import Word2Vec
import multiprocessing
import pandas as pd
import string


class WordIntoVectors:
    def __init__(self):
        self.cores = multiprocessing.cpu_count()
        self.url = "https://raw.githubusercontent.com/naynco/nayn.data/master/nayn.co-2016.csv"

    def load_data(self):
        df = pd.read_csv(self.url)
        return df.dropna().reset_index(drop=True)

    def cleaning(self, doc):
        stemmer = TurkishStemmer()
        doc = doc.lower()
        table = str.maketrans('', '', string.punctuation)
        stripped = doc.translate(table)
        clean_text = []
        for i in stripped.split():
            clean_text.append(stemmer.stem(i))

        return ' '.join(clean_text)

    def model(self):
        txt_list = []
        df = self.load_data()
        for doc in df["Content"]:
            txt_list.append(self.cleaning(doc))

        df["clean"] = txt_list

        sent = [row.split() for row in df['clean']]
        phrases = Phrases(sent, min_count=30, progress_per=10000)
        bigram = Phraser(phrases)

        return bigram[sent]

    def w2v_train(self):
        sentences = self.model()

        w2v_model = Word2Vec(min_count=20,
                             window=2,
                             size=300,
                             sample=6e-5,
                             alpha=0.03,
                             min_alpha=0.0007,
                             negative=20,
                             workers=self.cores - 1)
        w2v_model.build_vocab(sentences, progress_per=10000)

        w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
        w2v_model.save("word2vec.model")
        w2v_model.init_sims(replace=True)

    def load_model(self, query):
        import os.path
        if os.path.isfile("word2vec.model"):
            print("Model exists")
            model = Word2Vec.load("word2vec.model")
            return model.wv.most_similar(positive=[query])
        else:
            print("Model train is running!")
            self.w2v_train()


if __name__ == '__main__':
    print(WordIntoVectors().load_model(query="apple"))
