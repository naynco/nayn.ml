from elasticsearch_dsl import Search
from elasticsearch_dsl import connections, Q
from ESmodel import Article
from time import time
from settings import *

class ES:
    def __init__(self):
        pass

    def es_connection(self):
        return connections.create_connection(hosts=[ES_HOST],
                                             timeout=30,
                                             port=9200,
                                             http_auth='{}:{}'.format(ES_USERNAME, ES_PASSWORD))

    def insert(self, data, index):
        Article(title=data["Title"],
                name=data["Title"],
                categories=data["Categories"],
                status=data["Status"],
                slug=data["Slug"],
                create_date=time()
                ).save(index=index)

        Article._index.refresh()

    def search(self, query, field, client):
        q = Q("multi_match", query=query, fields=[field], operator="and", tie_breaker=1, type="most_fields")
        s = Search(using=client)
        s = s.query(q)
        return s.execute().to_dict()["hits"]["hits"]

    def find(self, query, client):
        q = Q("match", _id=query)
        s = Search(using=client)
        s = s.query(q)

        return s.execute().to_dict()["hits"]["hits"]

    def delete(self, index, document_id):
        s = Search(index=index).query("match", _id=document_id)
        response = s.delete()
        return response
