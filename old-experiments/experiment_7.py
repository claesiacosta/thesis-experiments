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
df_filtered = df.loc[(df['year']>=2017) & (df['image'] == 'No')].sample(25).copy()
df_MCQ = pd.DataFrame(columns=['id', 'area', 'year', 'context', 'question', 'options', 'correct_answer', 'bloom_level', 'difficult_level', 'relevance', 'adherence', 'answerability', 'correctness', 'feedback', 'rouge', 'bleu', 'system_answer', 'steps_answer', 'model', 'lang'])
df_MCQ_EN = pd.DataFrame(columns=['id', 'area', 'year', 'context', 'question', 'options', 'correct_answer', 'bloom_level', 'difficult_level', 'relevance', 'adherence', 'answerability', 'correctness', 'feedback', 'rouge', 'bleu', 'system_answer', 'steps_answer', 'model', 'lang'])
df_err = pd.DataFrame(columns=['id', 'area', 'year', 'context', 'response_gpt', 'response_llama'])

c=0
gpt = api.GPT(model="gpt-3.5-turbo")
llama = api.LLAMA()

for index, row in tqdm(df_filtered.iterrows()):
    try:
        c+=1
        print(c)
        system_context, user_prompt = helpers.prompt_builder_PT([[row['area'], row['context']]])
        messages=[{"role":"system","content": system_context},
                  {"role":"user","content": user_prompt}]
        res_gpt = gpt.chat(messages)
        res_llama = llama.chat(messages)

        response = json.loads(res_gpt)
        for question in response['perguntas']:
            evals = helpers.evaluate_generated_mcq(row['context'], question['pergunta'])
            dic = {'area': row['area'], 'year': row['year'], 'context': row['context']}
            dic['question'] = question['pergunta']
            dic['options'] = str(question['options'])
            dic['correct_answer'] = question['correct_answer']
            dic['bloom_level'] = question['bloom_level']
            dic['difficult_level'] = question['difficult_level']
            
            eval_pt = question
            eval_pt['area'] = row['area']
            eval_pt['context'] = row['context']
            system_context, user_prompt = helpers.evaluator_builder_PT(eval_pt)
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

            system_context, user_prompt = helpers.solve_builder_PT([row['context'], question['pergunta'], str(question['options'])])
            messages=[{"role":"system","content": system_context},
                      {"role":"user","content": user_prompt}]
            response_answer = json.loads(gpt.chat(messages))
            dic['system_answer'] = '{'+ response_answer['answer_letter'] + ': ' + response_answer['answer_text']+'}'
            dic['steps_answer'] = response_answer['steps_answer']
            dic['model'] = "gpt-3.5-turbo"
            dic['id'] = str(index)+'_'+str(row['year'])
            dic['lang'] = 'PT'

            df_MCQ.loc[len(df_MCQ)] = dic

            #############################

            system_context, user_prompt = helpers.translate_PT_EN([row['area'], row['context'], question['pergunta'], str(question['options']), question['correct_answer'], question['bloom_level'], question['difficult_level']])
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
            response_answer_EN = json.loads(gpt.chat(messages))
            dic_en['system_answer'] = '{'+ response_answer_EN['answer_letter'] + ': ' + response_answer_EN['answer_text']+'}'
            dic_en['steps_answer'] = response_answer_EN['steps_answer']
            dic_en['model'] = "gpt-3.5-turbo"
            dic_en['id'] = str(index)+'_'+str(row['year'])
            dic_en['lang'] = 'EN'
                
            df_MCQ_EN.loc[len(df_MCQ_EN)] = dic_en

            print(str(c) +"_"+"gpt")
        
        response = json.loads(res_llama)
        for question in response['perguntas']:
            evals = helpers.evaluate_generated_mcq(row['context'], question['pergunta'])
            dic = {'area': row['area'], 'year': row['year'], 'context': row['context']}
            dic['question'] = question['pergunta']
            dic['options'] = str(question['options'])
            dic['correct_answer'] = question['correct_answer']
            dic['bloom_level'] = question['bloom_level']
            dic['difficult_level'] = question['difficult_level']
            
            eval_pt = question
            eval_pt['area'] = row['area']
            eval_pt['context'] = row['context']
            system_context, user_prompt = helpers.evaluator_builder_PT(eval_pt)
            messages=[{"role":"system","content": system_context},
                      {"role":"user","content": user_prompt}]
            response_eval = json.loads(llama.chat(messages))
            dic['relevance'] = response_eval['relevance']
            dic['adherence'] = response_eval['adherence']
            dic['answerability'] = response_eval['answerability']
            dic['correctness'] = response_eval['correctness']
            dic['feedback'] = response_eval['feedback']
            dic['rouge'] = str(evals[0]['rougeL'])
            dic['bleu'] = evals[1]

            system_context, user_prompt = helpers.solve_builder_PT([row['context'], question['pergunta'], str(question['options'])])
            messages=[{"role":"system","content": system_context},
                      {"role":"user","content": user_prompt}]
            response_answer = json.loads(llama.chat(messages))
            dic['system_answer'] = '{'+ response_answer['answer_letter'] + ': ' + response_answer['answer_text']+'}'
            dic['steps_answer'] = response_answer['steps_answer']
            dic['model'] = "llama"
            dic['id'] = str(index)+'_'+str(row['year'])
            dic['lang'] = 'PT'

            df_MCQ.loc[len(df_MCQ)] = dic

            #############################

            system_context, user_prompt = helpers.translate_PT_EN([row['area'], row['context'], question['pergunta'], str(question['options']), question['correct_answer'], question['bloom_level'], question['difficult_level']])
            messages=[{"role":"system","content": system_context},
                      {"role":"user","content": user_prompt}]
            response_translation = json.loads(llama.chat(messages))
            dic_en = {'area': response_translation['1'], 'year': row['year'], 'context': response_translation['2'], 'question': response_translation['3'], 'options': response_translation['4'], 'correct_answer': response_translation['5'], 'bloom_level': response_translation['6'], 'difficult_level':response_translation['7']}
                
            system_context, user_prompt = helpers.evaluator_builder_EN([response_translation['1'], response_translation['2'], response_translation['3'], response_translation['4'], response_translation['6'], response_translation['7']])
            messages=[{"role":"system","content": system_context},
                      {"role":"user","content": user_prompt}]
            response_eval_EN = json.loads(llama.chat(messages))
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
            response_answer_EN = json.loads(llama.chat(messages))
            dic_en['system_answer'] = '{'+ response_answer_EN['answer_letter'] + ': ' + response_answer_EN['answer_text']+'}'
            dic_en['steps_answer'] = response_answer_EN['steps_answer']
            dic_en['model'] = "llama"
            dic_en['id'] = str(index)+'_'+str(row['year'])
            dic_en['lang'] = 'EN'
                
            df_MCQ_EN.loc[len(df_MCQ_EN)] = dic_en

            print(str(c) +"_"+"llama")

    except Exception as e:
        dic_error = {'id': row['id'], 'area': row['area'], 'year': row['year'], 'context': row['context']}
        dic_error['response_gpt'] = res_gpt
        dic_error['response_llama'] = res_llama
        df_err.loc[len(df_err)] = dic_error
        print(e)
        pass

date = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

df_MCQ.to_csv(f'results/MCQ_sample_solved_{date}.csv', index=False)
df_MCQ_EN.to_csv(f'results/MCQ_EN_sample_solved_{date}.csv', index=False)

df_err.to_csv(f'results/MCQ_sample_error_{date}.csv', index=False)

