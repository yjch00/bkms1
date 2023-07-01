from utils import fos_date_range_extractor, top_k_keyword_extractor, output_view_maker, query, Word_cloud_maker, graph_recommendation
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
from milvus import *
import torch
from torch.nn.functional import normalize
from simcse import SimCSE
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from nltk import word_tokenize, sent_tokenize


# streamlit run project.py  실행
# ____ 이 자리에 네 자리 버전 숫자 적기






#server table 확인
connection_info = "host= dbname=teamdb16 user=team16 password= port="
conn = psycopg2.connect(connection_info)





st.title('Paper Recommendation System')
st.write('2023 BKMS Project Team 16')



option = st.sidebar.selectbox(
    'Select Menu',
     pd.Series(['---SELECT---',"1) Citation-Based Recommendation", "2) Context-Based Recommendation", "3) Field Trend Analysis"]))


if str(option) == '1) Citation-Based Recommendation':    
    st.header(option)
    st.write()
    text = st.text_area(label="Write down the paper titles you are referring to. (ex. Paper1/Paper2/Paper3,...)", value="ex) OLAP for Trajectories/A Data Model for Moving Objects Supporting Aggregation/Spatial aggregation: Data model and implementation")
    text_k = st.text_area(label='Write down top k parameter. (ex. 5)', value = '5')
    if st.button("Search"):
        paper_list = text.split('/')
        st.dataframe(graph_recommendation(paper_list,int(text_k)).style.format({'score' : "{:.3f}"}))
        
        
          

elif str(option) == '2) Context-Based Recommendation':
    st.header(option)
    st.write()
    text = st.text_area(label="Write down the abstract you are working on or referring to.", value="ex) Deep learning allows computational models that are composed of multiple processing layers to learn representations of data with multiple levels of abstraction. These methods have dramatically improved the state-of-the-art in speech recognition, visual object recognition, object detection and many other domains such as drug discovery and genomics. Deep learning discovers intricate structure in large data sets by using the backpropagation algorithm to indicate how a machine should change its internal parameters that are used to compute the representation in each layer from the representation in the previous layer. Deep convolutional nets have brought about breakthroughs in processing images, video, speech and audio, whereas recurrent nets have shone light on sequential data such as text and speech.")
    text_k = st.text_area(label='Write down top k parameter. (ex. 3)', value = '3')
    if st.button("Search"):
        # PostgreSQL 데이터 로드
        df = load_db()
        # Milvus 데이터 로드
        collection = connect_milvus('team16_project_ml')
        
        index = ['top '+str(i+1) for i in range(0,int(text_k))]
        result = search_similar_abstract(df, collection, text, int(text_k))
        result['index'] = index
        result = result.set_index('index')
        st.dataframe(result[['title', 'abstract', 'distance']].style.format({'distance' : "{:.3f}"}))
        
        
    


elif str(option) == '3) Field Trend Analysis': 
    st.header(option)
    fos_list = ['Database', 'Cloud computing', 'Computation', 'Scalability',
       'Multimedia', 'Embedded system', 'The Internet',
       'Machine learning', 'Distributed computing', 'Mathematics']
    select = st.selectbox("Choose the field you are interested in.",fos_list)
    year_range = st.text_area(label="Write down years between 2000 ~ 2019. (ex. 2010~2018)", value="2010~2018")
    k_parameter = st.text_area(label='Write down top k parameter. (ex. 15)', value='15')
    
    start_year = int(year_range[0:4])
    end_year = int(year_range[5:9])
    k = int(k_parameter)
    if st.button("Analyze"):
        server_df = query('*', conn)
        fos_year = fos_date_range_extractor(server_df, str(select), start_year, end_year, k)
        result = top_k_keyword_extractor(server_df, select, fos_year, k)
        st.dataframe(output_view_maker(result, fos_year, k))
        Word_cloud_maker(result, fos_year)
    
     


else:
    st.title('')
    
    st.subheader('1) Citation-Based Recommendation')
    st.write('Recommend similar papers based on citation network')
    st.subheader('2) Context-Based Recommendation')
    st.write('Recommend similar papers based on abstract embedding')
    st.subheader('3) Field Trend Analysis')
    st.write('Analyze and visualize keyword trends for selected fields and years')
    
    st.title('')
    st.title('')
    st.title('')
    st.title('')
    st.title('')
    st.title('')
    st.title('')
    st.title('')
    
    st.write('BDAI LAB : 김현종, 정재원, 정철환, 조영재')

 
# if st.button("Confirm"):
#     con = st.container()
#     # con.caption("Result")
#     # con.write(f"User Name is {str(input_user_name)}")
#     # con.write(str(radio_gender))
#     # con.write(f"agree : {check_1}")
#     con.write(f"memo : {str(radio_gender)}")
