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

gpt_3_5 = api.GPT(model="gpt-3.5-turbo")
gpt_4 = api.GPT(model="gpt-4")
llama = api.LLAMA()

enem_pt = pd.read_csv('/Users/claesia/Documents/Thesis - experiment/thesis-experiments/src/dataset/enem-highschool/enem-highschool_PT.csv')
enem_pt = enem_pt.loc[(enem_pt['year']>=2017) & ((enem_pt['ledor'] == 'Yes') | (enem_pt['image'] == 'No'))].copy()

enem_en = pd.read_csv('/Users/claesia/Documents/Thesis - experiment/thesis-experiments/src/dataset/enem-highschool/enem-highschool_EN.csv')
enem_en = enem_en.loc[(enem_en['year']>=2017) & ((enem_pt['ledor'] == 'Yes') | (enem_pt['image'] == 'No'))].copy()

enade_pt = pd.read_csv('/Users/claesia/Documents/Thesis - experiment/thesis-experiments/src/dataset/enade-university/enade-university_PT.csv', sep=';')

enade_en = pd.read_csv('/Users/claesia/Documents/Thesis - experiment/thesis-experiments/src/dataset/enade-university/enade-university_EN.csv')

df_err = pd.DataFrame(columns=['id', 'area', 'year', 'context', 'file', 'error'])

input_dfs = {
    "PT":[{"enem-highschool_PT": enem_pt,
            "enade-university_PT": enade_pt
            }],
    "EN":[{"enem-highschool_EN": enem_en,
            "enade-university_EN": enade_en
    }]
}


def answering(exame, df, lang):
    for index, row in tqdm(df.iterrows()):
        try:
            if lang == 'PT':
                system_context, user_prompt = helpers.solver_builder_PT([row['context'], row['question'], str(row['options'])])
            if lang == 'EN':
                system_context, user_prompt = helpers.solver_builder_EN([row['context'], row['question'], str(row['options'])])
            
            solver_messages = [{"role":"system","content": system_context},
                      {"role":"user","content": user_prompt}]

            start_time = time.time()
            llama_response_solver= llama.chat(solver_messages)
            end_time = time.time()
            df.at[index, 'llama_response_solver'] = llama_response_solver
            df.at[index, 'llama_runtime'] = end_time - start_time

            start_time = time.time()
            gpt_3_5_response_solver= gpt_3_5.chat(solver_messages)
            end_time = time.time()
            df.at[index, 'gpt_3_5_response_solver'] = gpt_3_5_response_solver
            df.at[index, 'gpt_3_5_runtime'] = end_time - start_time

            start_time = time.time()
            gpt_4_response_solver= gpt_4.chat(solver_messages)
            end_time = time.time()
            df.at[index, 'gpt_4_response_solver'] = gpt_4_response_solver
            df.at[index, 'gpt_4_runtime'] = end_time - start_time

            llama_response_solver = json.loads(llama_response_solver)
            gpt_3_5_response_solver = json.loads(gpt_3_5_response_solver)
            gpt_4_response_solver = json.loads(gpt_4_response_solver)
            
        
            df.loc[index, 'system_answer_gpt_3_5'] = '{'+ gpt_3_5_response_solver['answer_letter'] + ': ' + gpt_3_5_response_solver['answer_text']+'}'
            df.loc[index, 'steps_answer_gpt_3_5'] = gpt_3_5_response_solver['steps_answer']

            df.loc[index, 'system_answer_gpt_4'] = '{'+ gpt_4_response_solver['answer_letter'] + ': ' + gpt_4_response_solver['answer_text']+'}'
            df.loc[index, 'steps_answer_gpt_4'] = gpt_4_response_solver['steps_answer']

            df.loc[index, 'system_answer_llama'] = '{'+ llama_response_solver['answer_letter'] + ': ' + llama_response_solver['answer_text']+'}'
            df.loc[index, 'steps_answer_llama'] = llama_response_solver['steps_answer']
    
        except Exception as e:
            dic_error = {'id': row['id'], 'area': row['area'], 'year': row['year'], 'context': row['context'], 'file': exame, 'error': str(e)}
            df_err.loc[len(df_err)] = dic_error
            print(e)
            pass
    return df

for lang, dfs_list in input_dfs.items():
    for dfs in dfs_list:
        for exame, df in dfs.items():
            print(lang)
            print(exame)
            df_res = answering(exame, df, lang)
            df_res.to_csv(f'outputs/{exame}_answered.csv', index=False)

df_err.to_csv(f'outputs/df_err_exams_answering.csv', index=False)


