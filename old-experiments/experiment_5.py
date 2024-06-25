from re import T
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from time import sleep
import os
import json
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import helpers
import api 

df = pd.read_csv('all_provas_enem_areas_sys_solved.csv')
df_MCQ = pd.DataFrame(columns=['area', 'year', 'context', 'question', 'options', 'correct_answer', 'bloom_level', 'difficult_level', 'relevance', 'adherence', 'answerability', 'correctness', 'feedback', 'rouge', 'bleu', 'system_answer', 'steps_answer', 'model'])
df_2013 = df.loc[(df['year']==2013) & (df['image'] == 'No')].copy()
df_2013['steps_answer'] = "" 


for model in ["gpt-3.5-turbo"]:#, "gpt-3.5", "gpt-4-32k", "gpt-4"]:
    c=0
    gpt = api.GPT(model=model)
    for index, row in tqdm(df_2013.iterrows()):
        try:
            c+=1
            print(c)
            system_context, user_prompt = helpers.prompt_builder_PT([[row['area'], row['context']]])
            messages=[{"role":"system","content": system_context},
                  {"role":"user","content": user_prompt}]
            res = gpt.chat(messages)
            response = json.loads(res)
    
            for question in response['questoes'][0]['perguntas']:
                evals = helpers.evaluate_generated_mcq(response['questoes'][0]['context'], question['pergunta'])
                dic = {'area': row['area'], 'year': row['year'], 'context':row['context']}
                dic['question'] = question['pergunta']
                options = [{k: v for k, v in dict(option).items() if k != 'correta'} for option in question['options']]
                options =  dict(item for d in options for item in d.items())
                dic['options'] = str(options)
                dic['correct_answer'] = question['correct_answer'][0]
                dic['bloom_level'] = question['bloom_level']
                dic['difficult_level'] = question['difficult_level']
                eval_pt = question
                eval_pt['area'] = response['questoes'][0]['area']
                eval_pt['context'] = response['questoes'][0]['context']

                system_context, user_prompt = helpers.evaluator_builder_PT_(eval_pt)
                messages=[{"role":"system","content": system_context},
                      {"role":"user","content": user_prompt}]
                response_eval = json.loads(gpt.chat(messages))
                dic['relevance'] = response_eval['relevance']
                dic['adherence'] = response_eval['adherence']
                dic['answerability'] = response_eval['answerability']
                dic['correctness'] = response_eval['correctness']
                dic['feedback'] = response_eval['feedback']
            
                dic['rouge'] = str(evals[0]['rougeL'])
                dic['bleu'] = evals[1]

                system_context, user_prompt = helpers.solve_builder_PT_([response['questoes'][0]['context'], question['pergunta'], str(options)])
                messages=[{"role":"system","content": system_context},
                      {"role":"user","content": user_prompt}]
                response_answer = json.loads(re.sub(r"\'(.*?)\'", r'"\1"', gpt.chat(messages)))
                dic['system_answer'] = response_answer[0]
                dic['steps_answer'] = response_answer[1]
                dic['model'] = model
        
                df_MCQ.loc[len(df_MCQ)] = dic
    
            system_context, user_prompt = helpers.solve_builder_PT_([row['context'], row['question'], row['options']])
            messages=[{"role":"system","content": system_context},
                  {"role":"user","content": user_prompt}]
            response_resposta = re.sub(r"\'(.*?)\'", r'"\1"', gpt.chat(messages))
            df_2013.loc[index, 'system_answer'] = response_resposta[0]
            df_2013.loc[index, 'steps_answer'] = response_resposta[1]
        except:
            pass
 
df_MCQ.to_csv('MCQ_solved_3.csv', index=False)
df_2013.to_csv('all_provas_enem_areas_sys_solved_.csv', index=False)

