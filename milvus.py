import pandas as pd
import torch
from torch.nn.functional import normalize
from simcse import SimCSE
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import psycopg2
import psycopg2.extras as extras
from nltk import word_tokenize, sent_tokenize

def load_db():
    # PostgreSQL 서버에서 df 불러오기 (644800 rows)
    connection_info = "host= dbname=teamdb16 user=team16 password= port="
    conn = psycopg2.connect(connection_info)
    try:
        server_df = pd.read_sql('SELECT * FROM citation_data',conn)

    except psycopg2.Error as e:
        print("DB error: ", e)
        
    finally:
        conn.close()

    # FOS가 Machine learning인 row만 추출 (20000개 샘플링)
    df_ml = server_df[server_df['fos']=='Machine learning'].sample(n=20000, random_state=42)

    return df_ml


def create_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    
    fields = [
    FieldSchema(name='id', dtype=DataType.VARCHAR, descrition='ids', max_length=500, is_primary=True, auto_id=False),
    FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, description='embedding vectors', dim=dim)
    ]
    schema = CollectionSchema(fields=fields, description='Team 16 Project')
    collection = Collection(name=collection_name, schema=schema)

    # create IVF_FLAT index for collection
    index_params = {
        'metric_type':'L2',
        'index_type':"IVF_FLAT",
        'params':{"nlist":2048}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    
    # print("\ncollection created:", collection_name)
    return collection

def search(collection, search_vectors, top_k):
    search_param = {
        "data": search_vectors,
        "anns_field": 'embedding',
        "param": {"metric_type": 'L2', "params": {"nprobe": 16}},
        "limit": top_k}
    results = collection.search(**search_param)
    result_id = [res.id for res in results[0]]
    result_dis = [res.distance for res in results[0]]

    return result_id, result_dis

def connect_milvus(collection_name):

    connections.connect(host='147.47.200.145', port='39530')

    # get an existing collection with name
    collection = Collection(collection_name)

    # flush collection
    collection.flush()

    # load data to memory
    collection.load()

    return collection

def search_similar_abstract(abstract_db, collection, input_abstract, top_k):

    # input_abstract의 임베딩 계산
    model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
    sent_emb_list = model.encode(sent_tokenize(input_abstract), max_length=256)
    query_vector = normalize(torch.mean(sent_emb_list, dim=0), dim=0)

    # search
    result_id, result_dis = search(collection, [query_vector.tolist()], top_k)
    row_list = [abstract_db.loc[abstract_db['id'] == i] for i in result_id]
    df_result = pd.concat(row_list).assign(distance=result_dis)

    return df_result

def disconnect_milvus(collection):
    
    # release memory
    collection.release()

    # drop collection index
    # collection.drop_index()

    # drop collection
    # collection.drop()

    # disconnect connection
    connections.disconnect("default")
