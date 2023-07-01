import streamlit as st
import pandas as pd
import time
import psycopg2
import psycopg2.extras as extras
import fuzzywuzzy
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import ast
from neo4j import GraphDatabase

def fos_date_range_extractor(df, fos, start_date, end_date, k = 10):
    fos_df = df[df['fos'] == fos]
    fos_df.reset_index(inplace = True, drop = True)
    
    fos_year = sorted(fos_df['year'].unique(), reverse = True)
    
    # find range of year
    
    start = -100
    end = -100
    
    for i in range(len(fos_year)):
        if fos_year[i] == start_date:
            start = i
        elif fos_year[i] == end_date:
            end = i

    fos_year = fos_year[end:start + 1]
    
    return fos_year
    
    

def top_k_keyword_extractor(df, fos, fos_year, k = 10):
    fos_df = df[df['fos'] == fos]
    fos_df.reset_index(inplace = True, drop = True)
    
    # create top k keyword dictionary result
    
    result = {}
    
    for y in fos_year:

        temp_df = fos_df[fos_df['year'] == y]
        temp_df.reset_index(inplace = True, drop = True)

        top_keyword_list = []

        for i in range(len(temp_df)):
            top_keyword_list.extend(temp_df['keywords'][i][1:-1].split(','))

        # 빈 문자열 제거
        top_keyword_list = [t for t in top_keyword_list if t]
        empty_idx = []

        for i in range(len(top_keyword_list)):

            # 소문자 처리 및 "", '' 것들 지우기
            top_keyword_list[i] = top_keyword_list[i].upper()
            top_keyword_list[i] = top_keyword_list[i].replace('"','')
            top_keyword_list[i] = top_keyword_list[i].replace("'","")

            # 빈 문자열 제거
            if top_keyword_list[i] == '':
                empty_idx.append(i)
                continue

            # 앞 뒤로 {, } 들 지우기 (원래 dict 인 것들)
            if top_keyword_list[i][0] == ' ':
                top_keyword_list[i] = top_keyword_list[i][1:]
            if top_keyword_list[i][-1] == ' ':
                top_keyword_list[i] = top_keyword_list[i][:-1]

        for i in empty_idx:
            del top_keyword_list[i]

        temp_dict = dict(Counter(top_keyword_list))
        temp = sorted(temp_dict.items(), key=lambda x: x[1],reverse = True)
        
        top_keyword_list = temp[:k] # top k keyword extraction

        temp_result = {y:top_keyword_list}
        result.update(temp_result)
        
    return result

# create result dataframe view 
def output_view_maker(result, fos_year, k):
    
    idx_list = []
    idx = ''

    # top k 
    for i in range(k):
        idx = 'top' + ' ' + str(i+1)
        idx_list.append(idx)

    temp_list = []
    
    # result initialization
    
    y = fos_year[0]
    
    for i in range(len(result[y])):
        temp_temp = list(result[y][i])
        temp_list.append(temp_temp)
        
    result_df = pd.DataFrame(temp_list, columns=[[y,y],['keyword', 'counts']], index = idx_list)


    # make multi column result df
    for y in fos_year[1:]:

        temp_list = []

        for i in range(len(result[y])):
            temp_temp = list(result[y][i])
            temp_list.append(temp_temp)
        temp_df = pd.DataFrame(temp_list, columns=[[y,y],['keyword', 'counts']], index = idx_list)
        result_df = pd.concat([result_df, temp_df], axis=1) # column bind

    return result_df

def query(col, conn):
    try:
        # 테이블을 Pandas.Dataframe으로 추출
        server_df = pd.read_sql(f'SELECT {str(col)} FROM citation_data',conn)

    except psycopg2.Error as e:
        # 데이터베이스 에러 처리
        print("DB error: ", e)

    finally:
        # 데이터베이스 연결 해제 필수!!
        conn.close()
    return (server_df)
def Word_cloud_maker(result, fos_year):
    wc = WordCloud(background_color="white", max_font_size=50)
    
    for y in fos_year:
        cloud = wc.generate_from_frequencies(dict(result[y]))
        plt.figure(figsize=(20, 20))
        plt.axis('off')
        plt.imshow(cloud)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.write('Key Word Cloud in ' + str(y))
        st.pyplot()
        
        #plt.show()
        
# neo4j 드라이버 import 및 함수 정의
class Neo4jConnection:
    
    def __init__(self, uri, user, pwd):
        self.__uri = uri
        self.__user = user
        self.__pwd = pwd
        self.__driver = None
        try:
            self.__driver = GraphDatabase.driver(self.__uri, auth=(self.__user, self.__pwd))
        except Exception as e:
            print("Failed to create the driver:", e)
        
    def close(self):
        if self.__driver is not None:
            self.__driver.close()
        
    def query(self, query, db=None):
        assert self.__driver is not None, "Driver not initialized!"
        session = None
        response = None
        try: 
            session = self.__driver.session(database=db) if db is not None else self.__driver.session() 
            response = list(session.run(query))
        except Exception as e:
            print("Query failed:", e)
        finally: 
            if session is not None:
                session.close()
        return response
    
    
def graph_recommendation(title_list, k=5):
    dbname = "teamdb16"
    uri_param = ""
    user_param = "team16"
    pwd_param = "bdai2023"
    conn = Neo4jConnection(uri=uri_param, user=user_param, pwd=pwd_param)
    
    score_dict = {}
    for title in title_list:
        # subgraph 초기화
        cypher = '''CALL gds.graph.drop('citations', false) YIELD graphName'''
        conn.query(cypher, db=dbname)
        
        # edge 3개 이내 그래프 저장
        cypher = f'''CALL gds.graph.project.cypher("citations","MATCH (:Paper {{title:'{title}'}})-[rel:REF*1..2]-(p:Paper) RETURN id(p) as id","MATCH (p1:Paper {{title:'{title}'}})-[rel:REF*1..2]-(p2:Paper) RETURN id(p1) AS source, id(p2) AS target") YIELD graphName AS graph, nodeQuery, nodeCount AS nodes, relationshipQuery, relationshipCount AS rels;'''
        conn.query(cypher, db=dbname)
        
        # article rank 계산
        cypher = '''CALL gds.articleRank.stream('citations') YIELD nodeId, score RETURN gds.util.asNode(nodeId).title AS title, score ORDER BY score DESC, title ASC'''
        response = conn.query(cypher, db=dbname)
        
        if not response:
            continue
            
        for i in response:
            score_dict[i['title']] = score_dict.get(i['title'], 0) + i['score']
            
    conn.close()     
    df = pd.DataFrame(score_dict.items(), columns=['title','score'])
    df = df[~df['title'].isin(title_list)]
    df = df.sort_values('score', ascending=False, ignore_index=True)
    
    df = df[:k]
    idx_list = []
    idx = ''
    
    # top k 
    for i in range(k):
        idx = 'top' + ' ' + str(i+1)
        idx_list.append(idx)
    df['index'] = idx_list
    df = df.set_index('index')
    return df