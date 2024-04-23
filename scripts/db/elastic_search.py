import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from elasticsearch import Elasticsearch
from utils.embedding import get_emb


class ElasticSearch:
    def __init__(self, host="http://localhost:9200"):
        self.es = Elasticsearch(host)

    def bm25_search(self, index_name, query, k=5, field="content"):
        body = {
            'query': {
                'match': {
                    field: query
                }
            },
            'size': k
        }
  
        res = self.es.search(
            index=index_name,
            body=body
        )

        return res
    
    def vector_search(self, index_name, query, k=5):
        embeddings = get_emb(query).data[0].embedding
        sem_search_result = self.es.search(
            index=index_name, 
            knn={
                "field": 'embedding',
                "query_vector": embeddings,
                "k": k,
                "num_candidates": 100
            }
        )   

        return sem_search_result
    
    def hybrid_search(self, index_name, query, k=5, ranking_constant=10):
        bm25_result = self.bm25_search(index_name, query, k)
        sem_result = self.vector_search(index_name, query, k)

        # RRF Score 
        rrf_results = {}
        for rank, hit in enumerate(sem_result['hits']['hits']):
            rrf_results[ hit['_id'] ] = {
                "_score": [1.0 / (ranking_constant + rank + 1)],
                "_source": hit['_source'],
            }

        for rank, hit in enumerate(bm25_result['hits']['hits']):
            if hit['_id'] in rrf_results:
                rrf_results[ hit['_id'] ]['_score'].append(1.0 / (ranking_constant + rank + 1))
            else:
                rrf_results[ hit['_id'] ] = {
                    "_score": [1.0 / (ranking_constant + rank + 1)],
                    "_source": hit['_source'],
                }

        for hit in rrf_results.values():
            hit['_score'] = sum(hit['_score'])

        rrf_results = dict(sorted(rrf_results.items(), key=lambda x: x[1]["_score"], reverse=True)[:k])
        return rrf_results
    
    def get_doc(self, index_name, field, value):
        body = {
            'query': {
                'term': {
                    field: value
                }
            },
            'size': 1
        }
        
        res = self.es.search(
            index=index_name,
            body=body,
        )

        return res

if __name__ == "__main__":
    query = "Tội phạm"
    es = ElasticSearch()
    res = es.hybrid_search("law-c-1", query)
    for _id, doc in res.items():
        # print(doc['_source']['metadata']['ref_doc_id'])
        doc_id = doc["_source"]['metadata']["ref_doc_id"]
        doc = es.get_doc("law-p-1", "_id", doc_id)
        print(doc['hits']['hits'][0]['_source']["content"])
        break
        



        
    