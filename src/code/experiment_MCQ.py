from re import T
import re
import ast
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from time import sleep
import os
import json
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import helpers
import api 

enem_pt = pd.read_csv('/Users/claesia/Documents/Thesis - experiment/thesis-experiments/src/dataset/enem-highschool/enem-highschool_PT.csv')
enem_pt = enem_pt.loc[(enem_pt['year']>=2017) & ((enem_pt['ledor'] == 'Yes') | (enem_pt['image'] == 'No'))].copy()
enem_pt_MCQ = pd.DataFrame(columns=['id', 'area', 'year', 'context', 'response_gpt', 'response_llama', 'response_gpt_4', 'runtime_gpt', 'runtime_llama', 'runtime_gpt_4'])

enem_en = pd.read_csv('/Users/claesia/Documents/Thesis - experiment/thesis-experiments/src/dataset/enem-highschool/enem-highschool_EN.csv')
enem_en = enem_en.loc[(enem_en['year']>=2017) & ((enem_pt['ledor'] == 'Yes') | (enem_pt['image'] == 'No'))].copy()
enem_en_MCQ = pd.DataFrame(columns=['id', 'area', 'year', 'context', 'response_gpt', 'response_llama', 'response_gpt_4', 'runtime_gpt', 'runtime_llama', 'runtime_gpt_4'])

enade_pt = pd.read_csv('/Users/claesia/Documents/Thesis - experiment/thesis-experiments/src/dataset/enade-university/enade-university_PT.csv', sep=';')
enade_pt_MCQ = pd.DataFrame(columns=['id', 'area', 'year', 'context', 'response_gpt', 'response_llama', 'response_gpt_4', 'runtime_gpt', 'runtime_llama', 'runtime_gpt_4'])

enade_en = pd.read_csv('/Users/claesia/Documents/Thesis - experiment/thesis-experiments/src/dataset/enade-university/enade-university_EN.csv')
enade_en_MCQ = pd.DataFrame(columns=['id', 'area', 'year', 'context', 'response_gpt', 'response_llama', 'response_gpt_4', 'runtime_gpt', 'runtime_llama', 'runtime_gpt_4'])

df_err = pd.DataFrame(columns=['id', 'area', 'year', 'context', 'file', 'error'])

gpt_3_5 = api.GPT(model="gpt-3.5-turbo")
gpt_4 = api.GPT(model="gpt-4")
#gpt_3_5_instruct = api.GPT(model="gpt-3.5-turbo-instruct")
llama = api.LLAMA()

date = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

c=0
for index, row in tqdm(enem_pt.iterrows()):
    try:
        c+=1
        system_context, user_prompt = helpers.prompt_builder_PT([[row['area'], row['context']]])
        messages=[{"role":"system","content": system_context},
                {"role":"user","content": user_prompt}]
        
        start_time = time.time()
        res_gpt = ''#gpt_3_5.chat(messages)
        end_time = time.time()
        runtime_gpt = ''#end_time - start_time 

        start_time = time.time()
        res_llama = ''#llama.chat(messages)
        end_time = time.time()
        runtime_llama = ''#end_time - start_time 

        # start_time = time.time()
        # res_gpt_3_5_instruct = gpt_3_5_instruct.chat_instruct(system_context + "\n\n" + user_prompt)
        # end_time = time.time()
        # runtime_gpt_3_5_instruct= end_time - start_time

        start_time = time.time()
        res_gpt_4 = gpt_4.chat(messages)
        end_time = time.time()
        runtime_gpt_4 = end_time - start_time

        dic = {'id': row['id'], 'area': row['area'], 'year': row['year'], 'context': row['context'], 'response_gpt': res_gpt, 'response_llama': res_llama, 'response_gpt_4': res_gpt_4, 'runtime_gpt': runtime_gpt, 'runtime_llama': runtime_llama, 'runtime_gpt_4': runtime_gpt_4}
        enem_pt_MCQ.loc[len(enem_pt_MCQ)] = dic
        print(str(c) +"_"+"enem_pt")
    
    except Exception as e:
        dic_error = {'id': row['id'], 'area': row['area'], 'year': row['year'], 'context': row['context'], 'file': "enem_pt", 'error': str(e)}
        df_err.loc[len(df_err)] = dic_error
        print(e)
        pass
enem_pt_MCQ.to_csv(f'outputs/enem_pt_MCQ_{date}.csv', index=False)

c=0
for index, row in tqdm(enade_pt.iterrows()):
    try:
        c+=1
        system_context, user_prompt = helpers.prompt_builder_PT([[row['area'], row['context']]])
        messages=[{"role":"system","content": system_context},
                {"role":"user","content": user_prompt}]
        
        start_time = time.time()
        res_gpt = ''#gpt_3_5.chat(messages)
        end_time = time.time()
        runtime_gpt = ''#end_time - start_time 

        start_time = time.time()
        res_llama = ''#llama.chat(messages)
        end_time = time.time()
        runtime_llama = ''#end_time - start_time 

        # start_time = time.time()
        # res_gpt_3_5_instruct = gpt_3_5_instruct.chat_instruct(system_context + "\n\n" + user_prompt)
        # end_time = time.time()
        # runtime_gpt_3_5_instruct= end_time - start_time 

        start_time = time.time()
        res_gpt_4 = gpt_4.chat(messages)
        end_time = time.time()
        runtime_gpt_4 = end_time - start_time 

        dic = {'id': row['id'], 'area': row['area'], 'year': row['year'], 'context': row['context'], 'response_gpt': res_gpt, 'response_llama': res_llama, 'response_gpt_4': res_gpt_4, 'runtime_gpt': runtime_gpt, 'runtime_llama': runtime_llama, 'runtime_gpt_4': runtime_gpt_4}
        enade_pt_MCQ.loc[len(enade_pt_MCQ)] = dic
        print(str(c) +"_"+"enade_pt")
    
    except Exception as e:
        dic_error = {'id': row['id'], 'area': row['area'], 'year': row['year'], 'context': row['context'], 'file': "enade_pt", 'error': str(e)}
        df_err.loc[len(df_err)] = dic_error
        pass
enade_pt_MCQ.to_csv(f'outputs/enade_pt_MCQ_{date}.csv', index=False)

c=0
for index, row in tqdm(enem_en.iterrows()):
    try:
        c+=1
        system_context, user_prompt = helpers.prompt_builder([[row['area'], row['context']]])
        messages=[{"role":"system","content": system_context},
                {"role":"user","content": user_prompt}]

        start_time = time.time()
        res_gpt = ''#gpt_3_5.chat(messages)
        end_time = time.time()
        runtime_gpt = ''#end_time - start_time 

        start_time = time.time()
        res_llama = ''#llama.chat(messages)
        end_time = time.time()
        runtime_llama = ''#end_time - start_time 

        # start_time = time.time()
        # res_gpt_3_5_instruct = gpt_3_5_instruct.chat_instruct(system_context + "\n\n" + user_prompt)
        # end_time = time.time()
        # runtime_gpt_3_5_instruct= end_time - start_time 

        start_time = time.time()
        res_gpt_4 = gpt_4.chat(messages)
        end_time = time.time()
        runtime_gpt_4 = end_time - start_time 

        dic = {'id': row['id'], 'area': row['area'], 'year': row['year'], 'context': row['context'], 'response_gpt': res_gpt, 'response_llama': res_llama, 'response_gpt_4': res_gpt_4, 'runtime_gpt': runtime_gpt, 'runtime_llama': runtime_llama, 'runtime_gpt_4': runtime_gpt_4}
        enem_en_MCQ.loc[len(enem_en_MCQ)] = dic
        print(str(c) +"_"+"enem_en")
    
    except Exception as e:
        dic_error = {'id': row['id'], 'area': row['area'], 'year': row['year'], 'context': row['context'], 'file': "enem_en", 'error': str(e)}
        df_err.loc[len(df_err)] = dic_error
        pass
enem_en_MCQ.to_csv(f'/Users/claesia/Documents/Thesis - experiment/thesis-experiments/src/code/outputs/enem_en_MCQ_{date}.csv', index=False)

c=0
for index, row in tqdm(enade_en.iterrows()):
    try:
        c+=1
        system_context, user_prompt = helpers.prompt_builder([[row['area'], row['context']]])
        messages=[{"role":"system","content": system_context},
                {"role":"user","content": user_prompt}]

        start_time = time.time()
        res_gpt = ''#gpt_3_5.chat(messages)
        end_time = time.time()
        runtime_gpt = ''#end_time - start_time 

        start_time = time.time()
        res_llama = ''#llama.chat(messages)
        end_time = time.time()
        runtime_llama = ''#end_time - start_time 

        # start_time = time.time()
        # res_gpt_3_5_instruct = gpt_3_5_instruct.chat_instruct(system_context + "\n\n" + user_prompt)
        # end_time = time.time()
        # runtime_gpt_3_5_instruct= end_time - start_time 

        start_time = time.time()
        res_gpt_4 = gpt_4.chat(messages)
        end_time = time.time()
        runtime_gpt_4 = end_time - start_time 

        dic = {'id': row['id'], 'area': row['area'], 'year': row['year'], 'context': row['context'], 'response_gpt': res_gpt, 'response_llama': res_llama, 'response_gpt_4': res_gpt_4, 'runtime_gpt': runtime_gpt, 'runtime_llama': runtime_llama, 'runtime_gpt_4': runtime_gpt_4}
        enade_en_MCQ.loc[len(enade_en_MCQ)] = dic
        print(str(c) +"_"+"enade_en")
    
    except Exception as e:
        dic_error = {'id': row['id'], 'area': row['area'], 'year': row['year'], 'context': row['context'], 'file': "enade_en", 'error': str(e)}
        df_err.loc[len(df_err)] = dic_error
        pass
enade_en_MCQ.to_csv(f'outputs/enade_en_MCQ_{date}.csv', index=False)


df_err.to_csv(f'outputs/df_err_{date}.csv', index=False)
