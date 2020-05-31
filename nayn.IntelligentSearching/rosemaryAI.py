import ssl
from haystack import Finder
from haystack.database.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.tfidf import TfidfRetriever
from haystack.reader.farm import FARMReader
from settings import *

ssl._create_default_https_context = ssl._create_unverified_context


class Rosemary:
    def __init__(self, ELASTIC_INDEX):
        self.ELASTIC_INDEX = ELASTIC_INDEX

    def qa(self, question, text_field):
        document_store = ElasticsearchDocumentStore(host=ES_HOST,
                                                    username=ES_USERNAME,
                                                    password=ES_PASSWORD,
                                                    index=self.ELASTIC_INDEX,
                                                    text_field=text_field
                                                    )
        retriever = TfidfRetriever(document_store=document_store)

        reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)

        finder = Finder(reader, retriever)
        prediction = finder.get_answers(question=question,
                                        top_k_retriever=1,
                                        top_k_reader=5)

        return prediction
