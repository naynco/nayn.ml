import pandas as pd
from ES import ES
from rosemaryAI import Rosemary
from pprint import pprint


def csv_file_save_to_ES():
    ES().es_connection()
    df = pd.read_csv("classification_clean.csv").to_dict(orient="records")
    [ES().insert(item,"news") for item in df]

result  = Rosemary(ELASTIC_INDEX="news").qa(question="Atatürk Havalimanında olan saldırada kaç kişinin tutuklanması istendi ?", text_field="title")


pprint(result)

''' 
#################################### OUTPUT ################################## 

{'answers': [{'answer': 'İlgili 17',
              'context': "Atatürk Havalimanı'ndaki Saldırıyla İlgili 17 "
                         'Kişinin Tutuklanması İstendi',
              'document_id': '4',
              'meta': None,
              'offset_end': 46,
              'offset_end_in_doc': 46,
              'offset_start': 36,
              'offset_start_in_doc': 36,
              'probability': 0.42374577281800946,
              'score': -2.459321975708008}],
 'no_ans_gap': -5.3099236488342285,
 'question': 'Atatürk Havalimanında olan saldırada kaç kişinin tutuklanması '
             'istendi ?'}

#################################################################################
'''