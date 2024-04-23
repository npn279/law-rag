import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dotenv
dotenv.load_dotenv()

import time
from pprint import pprint
import argparse
import logging
import cohere

from scripts.db.elastic_search import ElasticSearch
from scripts.utils.GEMINI import GEMINI
from scripts.utils.OpenAI import OPENAI
from scripts.utils.prompt_templates import *


parser = argparse.ArgumentParser()
parser.add_argument('--mode', help='Log mode', default='WARN')
args = parser.parse_args()
log_mode = args.mode
if log_mode.strip().upper() == 'DEBUG':
    logging.basicConfig(level=logging.DEBUG)
elif log_mode.strip().upper() == 'INFO':
    logging.basicConfig(level=logging.INFO)
else:
    logging.basicConfig(level=logging.WARNING)


es = ElasticSearch()
gemini = GEMINI(api_key=os.getenv('GEMINI_API_KEY'))
openai = OPENAI(api_key=os.getenv('OPENAI_API_KEY'))
PARENT_INDEX = os.getenv('PARENT_INDEX')
CHILD_INDEX = os.getenv('CHILD_INDEX')
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
co = cohere.Client(COHERE_API_KEY)


def gen_answer(query, search_method='hybrid', rewrite=True, return_context=False, stream=True):
    try:
        ##########
        ## Rewrite
        ########## 
        if rewrite:
            rewrite_prompt = """User (Original question): {query}\n\nLawie:"""
            rewrite_response = openai.get_response(prompt=rewrite_prompt.format(query=query), system_prompt=REWRITE_TEMPLATE, stream=False, temperature=0)
            rewrite_response = rewrite_response.choices[0].message.content
            if rewrite_response.strip().startswith("queries"):
                queries = [query] + rewrite_response.strip()[8:].split('\n')
                logging.info(f"queries: {queries}")
            elif str(rewrite_response).lower().strip().startswith("response"):
                return {'reponse': rewrite_response, 'context': "", 'type': 'str', 'error': None}
        else:
            queries = [query]


        ##########
        ## Context
        ##########
        contexts, ref_doc_ids = [], []
        search_method = search_method.lower().strip()

        if search_method == 'bm25':
            for sub_query in queries:
                search_results = es.bm25_search(index_name=CHILD_INDEX, query=sub_query, k=5)
                for hit in search_results['hits']['hits']:
                    id_ = hit['_source']['metadata']['ref_doc_id']
                    if id_ not in ref_doc_ids:
                        ref_doc_ids.append(id_)
        elif search_method == 'vector':
            for sub_query in queries:
                search_results = es.vector_search(index_name=CHILD_INDEX, query=sub_query, k=5)
                for hit in search_results['hits']['hits']:
                    id_ = hit['_source']['metadata']['ref_doc_id']
                    if id_ not in ref_doc_ids:
                        ref_doc_ids.append(id_)
        else:
            for sub_query in queries:
                search_results = es.hybrid_search(CHILD_INDEX, query=sub_query, k=5)
                for hit in search_results.values():
                    id_ = hit['_source']['metadata']['ref_doc_id']
                    if id_ not in ref_doc_ids:
                        ref_doc_ids.append(id_)
        
        for id_ in ref_doc_ids:
            doc = es.get_doc(PARENT_INDEX, '_id', id_)
            contexts.append(doc['hits']['hits'][0]['_source']['content'])


        ##########
        ## Re-rank
        ##########
        response = co.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=contexts,
            top_n=5 if len(contexts) > 5 else len(contexts),
        )

        logging.info(f'Rerank response: {response}')

        context = []
        for r in response.results:
            context.append(contexts[r.index])
        context = '\n\n'.join(context)

        ###########
        ## Response
        ###########
        # prompt = GEMINI_ANSWER_TEMPLATE.format(context=context, question=query)
        # logging.info(f"prompt: {prompt}")
        # response = gemini.generate(prompt=prompt, stream=stream)
        prompt = f'''#Context\n{context}\n\n#Question\n{query}\n\n#Answer\n'''
        response = openai.get_response(prompt=prompt, system_prompt=ANSWER_TEMPLATE, stream=stream, temperature=0.9)


        return {
            # 'response': response if stream else response.text, 
            'response': response if stream else response.choices[0].message.content, 
            'context': context if return_context else "", 
            'type': 'stream' if stream else 'str',
            'error': None
        }
    except Exception as e:
        logging.error(f"Error: {e}")
        return {
            'response': None, 
            'context': None, 
            'type': None,
            'error': str(e)
        }


if __name__=='__main__':
    # query = "đụng xe chết người bị phạt thế nào"
    # query = "cách điều chế ma túy là gì"
    query = "người lái xe say xỉn, gây tai nạn giao thông cho người đi xe đạp cùng chiều, người đi xe đạp bị gãy chân thì người gây tai nạn bị phạt như thế nào"
    response = gen_answer(query, rewrite=True, search_method='hybrid', return_context=False)
    print('-'*50)
    # for r in response['response']:
    #     print(r.text, end='', flush=True)
    for r in response['response']:
        print(r.choices[0].delta.content, end='', flush=True)

    # response = gen_answer('đụng xe chết người bị phạt thế nào', stream=False, return_context=False)
    # for r in response:
    #     print(r.choices[0].delta.content, end='', flush=True)
    #     print() 

