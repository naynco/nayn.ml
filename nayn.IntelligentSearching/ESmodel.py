from elasticsearch_dsl import Document, Text, Long, Keyword


class Article(Document):
    title = Text(fields={'keyword': Keyword()})
    name = Text(fields={'keyword': Keyword()})
    categories = Text(fields={'keyword': Keyword()})
    status = Text(fields={'keyword': Keyword()})
    slug = Text(fields={'keyword': Keyword()})
    create_date = Long()

    class Index:
        settings = {
            'number_of_shards': 1,
            'number_of_replicas': 0
        }

