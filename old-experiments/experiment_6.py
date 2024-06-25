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
df_MCQ = pd.DataFrame(columns=['area', 'year', 'context', 'question', 'options', 'correct_answer', 'bloom_level', 'difficult_level', 'relevance', 'adherence', 'answerability', 'correctness', 'feedback', 'rouge', 'bleu', 'system_answer', 'steps_answer', 'model'])
df_MCQ_EN = pd.DataFrame(columns=['area', 'year', 'context', 'question', 'options', 'correct_answer', 'bloom_level', 'difficult_level', 'relevance', 'adherence', 'answerability', 'correctness', 'feedback', 'rouge', 'bleu', 'system_answer', 'steps_answer', 'model'])
df_2023 = df.loc[(df['year']>=2017) & (df['image'] == 'No')].copy()
df_2023['steps_answer'] = "" 
df_2023_EN = pd.DataFrame(columns=df_2023.columns)

for model in ["gpt-3.5-turbo"]:#, "gpt-3.5", "gpt-4-32k", "gpt-4"]:
    c=0
    gpt = api.GPT(model=model)
    for index, row in tqdm(df_2023.iterrows()):
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
                response_answer = json.loads(gpt.chat(messages))
                dic['system_answer'] = '{'+ response_answer['answer_letter'] + ': ' + response_answer['answer_text']+'}'
                dic['steps_answer'] = response_answer['steps_answer']
                dic['model'] = model
        
                df_MCQ.loc[len(df_MCQ)] = dic
                #############################

                system_context, user_prompt = helpers.translate_PT_EN([row['area'], row['context'], question['pergunta'], str(options), question['correct_answer'][0], question['bloom_level'], question['difficult_level']])
                messages=[{"role":"system","content": system_context},
                      {"role":"user","content": user_prompt}]
                response_translation = json.loads(gpt.chat(messages))
                dic_en = {'area': response_translation['1'], 'year': row['year'], 'context': response_translation['2'], 'question': response_translation['3'], 'options': response_translation['4'], 'correct_answer': response_translation['5'], 'bloom_level': response_translation['6'], 'difficult_level':response_translation['7']}
                
                system_context, user_prompt = helpers.evaluator_builder_EN([response_translation['1'], response_translation['2'], response_translation['3'], response_translation['4'], response_translation['6'], response_translation['7']])
                messages=[{"role":"system","content": system_context},
                      {"role":"user","content": user_prompt}]
                response_eval_EN = json.loads(gpt.chat(messages))
                dic_en['relevance'] = response_eval_EN['relevance']
                dic_en['adherence'] = response_eval_EN['adherence']
                dic_en['answerability'] = response_eval_EN['answerability']
                dic_en['correctness'] = response_eval_EN['correctness']
                dic_en['feedback'] = response_eval_EN['feedback']
                evals_EN = helpers.evaluate_generated_mcq(response_translation['2'], response_translation['3'])
                dic_en['rouge'] = str(evals_EN[0]['rougeL'])
                dic_en['bleu'] = evals_EN[1]

                system_context, user_prompt = helpers.solve_builder_EN([response_translation['2'], response_translation['3'], response_translation['4']])
                messages=[{"role":"system","content": system_context},
                      {"role":"user","content": user_prompt}]
                ans = gpt.chat(messages)
                response_answer_EN = json.loads(gpt.chat(messages))
                dic_en['system_answer'] = '{'+ response_answer_EN['answer_letter'] + ': ' + response_answer_EN['answer_text']+'}'
                dic_en['steps_answer'] = response_answer_EN['steps_answer']
                dic_en['model'] = model
                
                df_MCQ_EN.loc[len(df_MCQ_EN)] = dic_en
                print(c)

    
            system_context, user_prompt = helpers.solve_builder_PT_([row['context'], row['question'], row['options']])
            messages=[{"role":"system","content": system_context},
                  {"role":"user","content": user_prompt}]
            response_resposta = json.loads(gpt.chat(messages))
            df_2023.loc[index, 'system_answer'] = '{'+ response_resposta['answer_letter'] + ': ' + response_resposta['answer_text']+'}'
            df_2023.loc[index, 'steps_answer'] = response_resposta['steps_answer']
            
            ##############################

            dic_all_en = {'id': df_2023.loc[index, 'id'], 'year': df_2023.loc[index, 'year'], 'CE': df_2023.loc[index, 'CE'], 
                          'DS': df_2023.loc[index, 'DS'], 'EK': df_2023.loc[index, 'EK'], 'IC': df_2023.loc[index, 'IC'],
                          'MR': df_2023.loc[index, 'MR'], 'TC': df_2023.loc[index, 'TC'],
                         'image': df_2023.loc[index, 'image'], 'IU': df_2023.loc[index, 'IU'], 'ledor': df_2023.loc[index, 'ledor'], 
                          'figures': df_2023.loc[index, 'figures']
                         }
            system_context, user_prompt = helpers.translate_PT_EN([df_2023.loc[index, 'area'], df_2023.loc[index, 'context'], df_2023.loc[index, 'question'], df_2023.loc[index, 'options'], df_2023.loc[index, 'correct_answer'], df_2023.loc[index, 'cor']])
            messages=[{"role":"system","content": system_context},
                      {"role":"user","content": user_prompt}]
            response_translation_EN = json.loads(gpt.chat(messages))
            dic_all_en['area'] = response_translation_EN['1']
            dic_all_en['context'] = response_translation_EN['2']
            dic_all_en['question'] = response_translation_EN['3']
            dic_all_en['options'] = response_translation_EN['4']
            dic_all_en['correct_answer'] = response_translation_EN['5']
            dic_all_en['cor'] = response_translation_EN['6']
            
            system_context, user_prompt = helpers.solve_builder_EN([dic_all_en['context'], dic_all_en['question'], dic_all_en['options']])
            messages=[{"role":"system","content": system_context},
                  {"role":"user","content": user_prompt}]
            response_resposta_EN = json.loads(gpt.chat(messages))
            dic_all_en['system_answer'] = '{'+ response_resposta_EN['answer_letter'] + ': ' + response_resposta_EN['answer_text']+'}'
            dic_all_en['steps_answer'] = response_resposta_EN['steps_answer']
            
            df_2023_EN.loc[len(df_2023_EN)] = dic_all_en

        except:
            pass

date = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

df_MCQ.to_csv(f'results/MCQ_sample_solved_{date}.csv', index=False)
df_2023.to_csv(f'results/provas_enem_sample_solved_{date}.csv', index=False)
df_MCQ_EN.to_csv(f'results/MCQ_sample_solved_EN_{date}.csv', index=False)
df_2023_EN.to_csv(f'results/provas_enem_sample_solved_EN_{date}.csv', index=False)


