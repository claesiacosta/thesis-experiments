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

date = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

input_dfs_EN = {
    "enem":[{"gpt_3_5": pd.read_csv("outputs/MCQ-preprocessed/EN/GPT_3_5/enem_en_MCQ_gpt_3_5_preprocessed.csv"),
            "llama": pd.read_csv("outputs/MCQ-preprocessed/EN/LLAMA/enem_en_MCQ_llama_preprocessed.csv"),
            "gpt_4": pd.read_csv("outputs/MCQ-preprocessed/EN/GPT_4/enem_en_MCQ_gpt_4_preprocessed.csv")
            }],
    "enade":[{"gpt_3_5": pd.read_csv("outputs/MCQ-preprocessed/EN/GPT_3_5/enade_en_MCQ_gpt_3_5_preprocessed.csv"),
            "llama": pd.read_csv("outputs/MCQ-preprocessed/EN/LLAMA/enade_en_MCQ_llama_preprocessed.csv"),
            "gpt_4": pd.read_csv("outputs/MCQ-preprocessed/EN/GPT_4/enade_en_MCQ_gpt_4_preprocessed.csv")
    }]
}

input_dfs_PT = {
    "enem":[{"gpt_3_5": pd.read_csv("outputs/MCQ-preprocessed/PT/GPT_3_5/enem_pt_MCQ_gpt_3_5_preprocessed.csv"),
            "llama": pd.read_csv("outputs/MCQ-preprocessed/PT/LLAMA/enem_pt_MCQ_llama_preprocessed.csv"),
            "gpt_4": pd.read_csv("outputs/MCQ-preprocessed/PT/GPT_4/enem_pt_MCQ_gpt_4_preprocessed.csv")
            }],
    "enade":[{"gpt_3_5": pd.read_csv("outputs/MCQ-preprocessed/PT/GPT_3_5/enade_pt_MCQ_gpt_3_5_preprocessed.csv"),
            "llama": pd.read_csv("outputs/MCQ-preprocessed/PT/LLAMA/enade_pt_MCQ_llama_preprocessed.csv"),
            "gpt_4": pd.read_csv("outputs/MCQ-preprocessed/PT/GPT_4/enade_pt_MCQ_gpt_4_preprocessed.csv")
    }]
}


df_err_pt = pd.DataFrame(columns=['id','area','year','context','question','bloom_level','difficult_level','options','correct_answer','correct_answer_letter','correct_answer_text','response_json','relevance','adherence','answerability','correctness','feedback','rouge','bleu','system_answer','steps_answer','response_evaluator','response_solver','system_answer_letter','system_answer_text', 'model', 'exame', 'error']) 
df_err_en = pd.DataFrame(columns=['id','area','year','context','question','bloom_level','difficult_level','options','correct_answer','correct_answer_letter','correct_answer_text','response_json','relevance','adherence','answerability','correctness','feedback','rouge','bleu','system_answer','steps_answer','response_evaluator','response_solver','system_answer_letter','system_answer_text', 'model', 'exame', 'error'])    

def pt_eval(model, exame, df):
    for index, row in tqdm(df.iterrows()):
        print(len(df)-index)
        try:
            evals_1 = helpers.evaluate_generated_mcq(row['context'], row['question'])
            df.at[index, 'rouge'] = str(evals_1[0]['rougeL'])
            df.at[index, 'bleu'] = evals_1[1]

            evals_2 = row['response_json'].replace("\'", "\"")
            evals_2 = str(evals_2) + "\n" + str({"area":row['area'], "context": row['context']})

            system_context, user_prompt = helpers.evaluator_builder_PT(evals_2)
            evaluator_messages = [{"role":"system","content": system_context},
                      {"role":"user","content": user_prompt}]
            
            system_context, user_prompt = helpers.solver_builder_PT([row['context'], row['question'], str(row['options'])])
            solver_messages = [{"role":"system","content": system_context},
                      {"role":"user","content": user_prompt}]

            if model == 'llama':
                response_evaluator = llama.chat(evaluator_messages)
                response_solver= llama.chat(solver_messages)

            if model == 'gpt_3_5':
                response_evaluator = gpt_3_5.chat(evaluator_messages)
                response_solver= gpt_3_5.chat(solver_messages)
                
            if model == 'gpt_4':
                response_evaluator = gpt_4.chat(evaluator_messages)
                response_solver= gpt_4.chat(solver_messages)
            
            df.at[index, 'response_evaluator'] = response_evaluator
            df.at[index, 'response_solver'] = response_solver

            response_evaluator = json.loads(response_evaluator)
            df.at[index, 'relevance'] = response_evaluator['relevance']
            df.at[index, 'adherence'] = response_evaluator['adherence']
            df.at[index, 'answerability'] = response_evaluator['answerability']
            df.at[index, 'correctness'] = response_evaluator['correctness']
            df.at[index, 'feedback'] = response_evaluator['feedback']

            response_solver = json.loads(response_solver)
            df.at[index, 'system_answer'] = '{'+ response_solver['answer_letter'] + ': ' + response_solver['answer_text']+'}'
            df.at[index, 'steps_answer'] = response_solver['steps_answer']
            df.at[index, 'system_answer_letter'] = response_solver['answer_letter']
            df.at[index, 'system_answer_text'] = response_solver['answer_text']
    
        except Exception as e:
            row_dict = df.loc[index].to_dict()
            row_dict['error'] = str(e)
            row_dict['model'] = model
            row_dict['exame'] = exame
            df_err_pt.loc[len(df_err_pt)] = row_dict
            pass
    return df

def en_eval(model, exame, df):
    for index, row in tqdm(df.iterrows()):
        print(len(df)-index)
        try:
            evals_1 = helpers.evaluate_generated_mcq(row['context'], row['question'])
            df.at[index, 'rouge'] = str(evals_1[0]['rougeL'])
            df.at[index, 'bleu'] = evals_1[1]

            evals_2 = row['response_json'].replace("\'", "\"")
            evals_2 = str(evals_2) + "\n" + str({"area":row['area'], "context": row['context']})

            system_context, user_prompt = helpers.evaluator_builder_EN(evals_2)
            evaluator_messages = [{"role":"system","content": system_context},
                      {"role":"user","content": user_prompt}]
            
            system_context, user_prompt = helpers.solver_builder_EN([row['context'], row['question'], str(row['options'])])
            solver_messages = [{"role":"system","content": system_context},
                      {"role":"user","content": user_prompt}]

            if model == 'llama':
                response_evaluator = llama.chat(evaluator_messages)
                response_solver= llama.chat(solver_messages)

            if model == 'gpt_3_5':
                response_evaluator = gpt_3_5.chat(evaluator_messages)
                response_solver= gpt_3_5.chat(solver_messages)
                
            if model == 'gpt_4':
                response_evaluator = gpt_4.chat(evaluator_messages)
                response_solver= gpt_4.chat(solver_messages)
            
            df.at[index, 'response_evaluator'] = response_evaluator
            df.at[index, 'response_solver'] = response_solver
            
            response_evaluator = json.loads(response_evaluator)
            df.at[index, 'relevance'] = response_evaluator['relevance']
            df.at[index, 'adherence'] = response_evaluator['adherence']
            df.at[index, 'answerability'] = response_evaluator['answerability']
            df.at[index, 'correctness'] = response_evaluator['correctness']
            df.at[index, 'feedback'] = response_evaluator['feedback']
            
            response_solver = json.loads(response_solver)
            df.at[index, 'system_answer'] = '{'+ response_solver['answer_letter'] + ': ' + response_solver['answer_text']+'}'
            df.at[index, 'steps_answer'] = response_solver['steps_answer']
            df.at[index, 'system_answer_letter'] = response_solver['answer_letter']
            df.at[index, 'system_answer_text'] = response_solver['answer_text']
    
        except Exception as e:
            row_dict = df.loc[index].to_dict()
            row_dict['error'] = str(e)
            row_dict['model'] = model
            row_dict['exame'] = exame
            df_err_en.loc[len(df_err_en)] = row_dict
            pass

    return df


for exam, models_list in input_dfs_EN.items():
    for models in models_list:
        for model_name, df in models.items():
            df_res = en_eval(model_name, exam, df)
            df_res.to_csv(f'outputs/{exam}_en_MCQ_{model_name}_answered_evaluated.csv', index=False)


for exam, models_list in input_dfs_PT.items():
    for models in models_list:
        for model_name, df in models.items():
            df_res = pt_eval(model_name, exam, df)
            df_res.to_csv(f'outputs/{exam}_pt_MCQ_{model_name}_answered_evaluated.csv', index=False)      

df_err_en.to_csv(f'outputs/df_errors_EN_answered_evaluated.csv', index=False)
df_err_pt.to_csv(f'outputs/df_errors_PT_answered_evaluated.csv', index=False)
