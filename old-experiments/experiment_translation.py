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

df = pd.read_csv('all_provas_enem_areas_sys_solved.csv')
df_EN = pd.DataFrame(columns=df.columns)

c=0
gpt = api.GPT(model="gpt-3.5-turbo")

for index, row in tqdm(df.iterrows()):
    try:
        c+=1
        print(c)
        dic_all_en = {'id': df.loc[index, 'id'], 'year': df.loc[index, 'year'], 'CE': df.loc[index, 'CE'], 
                          'DS': df.loc[index, 'DS'], 'EK': df.loc[index, 'EK'], 'IC': df.loc[index, 'IC'],
                          'MR': df.loc[index, 'MR'], 'TC': df.loc[index, 'TC'],
                         'image': df.loc[index, 'image'], 'IU': df.loc[index, 'IU'], 'ledor': df.loc[index, 'ledor'], 
                          'figures': df.loc[index, 'figures']
                    }
        system_context, user_prompt = helpers.translate_PT_EN([df.loc[index, 'area'], df.loc[index, 'context'], df.loc[index, 'question'], df.loc[index, 'options'], df.loc[index, 'correct_answer'], df.loc[index, 'cor']])
        messages=[{"role":"system","content": system_context},
                {"role":"user","content": user_prompt}]
        response_translation_EN = json.loads(gpt.chat(messages))
        dic_all_en['area'] = response_translation_EN['1']
        dic_all_en['context'] = response_translation_EN['2']
        dic_all_en['question'] = response_translation_EN['3']
        dic_all_en['options'] = response_translation_EN['4']
        dic_all_en['correct_answer'] = response_translation_EN['5']
        dic_all_en['cor'] = response_translation_EN['6']

        df_EN.loc[len(df_EN)] = dic_all_en

    except:
        pass

df_EN.to_csv(f'results/provas_enem_EN.csv', index=False)
