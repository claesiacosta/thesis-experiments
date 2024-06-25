from re import T
import re
import ast
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

df = pd.read_csv('results/provas_enem_EN.csv')
df_filtered = df.loc[(df['year']>=2017) & (df['image'] == 'No')].sample(100).copy()
df_filtered['steps_answer_gpt'] = "" 
df_filtered['steps_answer_llama'] = "" 
df_filtered['system_answer_llama'] = "" 
df_filtered['system_answer_gpt'] = "" 

c=0
gpt = api.GPT(model="gpt-3.5-turbo")
llama = api.LLAMA()

for index, row in tqdm(df_filtered.iterrows()):
    try:
        c+=1
        print(c)
        system_context, user_prompt = helpers.solve_builder_EN([row['context'], row['question'], row['options']])
        messages=[{"role":"system","content": system_context},
                {"role":"user","content": user_prompt}]

        res_gpt = gpt.chat(messages)
        res_llama = llama.chat(messages)

        response_resposta_EN = json.loads(res_gpt)
        df_filtered.loc[index, 'system_answer_gpt'] = '{'+ response_resposta_EN['answer_letter'] + ': ' + response_resposta_EN['answer_text']+'}'
        df_filtered.loc[index, 'steps_answer_gpt'] = response_resposta_EN['steps_answer']

        response_resposta_EN = json.loads(res_llama)
        df_filtered.loc[index, 'system_answer_llama'] = '{'+ response_resposta_EN['answer_letter'] + ': ' + response_resposta_EN['answer_text']+'}'
        df_filtered.loc[index, 'steps_answer_llama'] = response_resposta_EN['steps_answer']
    
    except Exception as e:
        print(e)
        pass

df_filtered.to_csv(f'results/provas_enem_EN_sample_solved.csv', index=False)