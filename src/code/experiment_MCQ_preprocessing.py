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

enem_pt = pd.read_csv("outputs/MCQ-raw/enem_pt_MCQ.csv")
enem_en = pd.read_csv("outputs/MCQ-raw/enem_en_MCQ.csv")
enade_pt = pd.read_csv("outputs/MCQ-raw/enade_pt_MCQ.csv")
enade_en = pd.read_csv("outputs/MCQ-raw/enade_en_MCQ.csv")

def json_to_dataframe(data, response_col):
    rows = []
    errors = []
    for i in range(len(data)):
        try:
            response_json = json.loads(data[response_col].iloc[i])
            for question in response_json['questions']:
                row = {
                    'index': i,
                    'id': data['id'].iloc[i],
                    'area': data['area'].iloc[i],
                    'year': data['year'].iloc[i],
                    'context': data['context'].iloc[i],
                    'question': question['question'],
                    'bloom_level': question['bloom_level'],
                    'difficult_level': question.get('difficult_level', question.get('difficulty_level')),
                    'options': question['options'],
                    'correct_answer': question['correct_answer'],
                    'correct_answer_letter': question['correct_answer_letter'],
                    'correct_answer_text': question['correct_answer_text'],
                    'response_json': question,
                    'relevance': None,
                    'adherence': None,
                    'answerability': None,
                    'correctness': None,
                    'feedback': None,
                    'rouge': None,
                    'bleu': None,
                    'system_answer': None,
                    'steps_answer': None
                }
                rows.append(row)
        except KeyError as e:
            error_row = {
                'index': len(errors),
                'id': data['id'].iloc[i],
                'area': data['area'].iloc[i],
                'year': data['year'].iloc[i],
                'context': data['context'].iloc[i],
                response_col: data[response_col].iloc[i],
                'error': str(e)
            }
            errors.append(error_row)
        except json.JSONDecodeError as e:
            error_row = {
                'index': len(errors),
                'id': data['id'].iloc[i],
                'area': data['area'].iloc[i],
                'year': data['year'].iloc[i],
                'context': data['context'].iloc[i],
                response_col: data[response_col].iloc[i],
                'error': str(e)
            }
            errors.append(error_row)

    return pd.DataFrame(rows), pd.DataFrame(errors)

def process_dataframes(dfs_dict):
    results = {}
    errors = {}
    for name, df in dfs_dict.items():
        print(name)
        df_gpt, errors_gpt = json_to_dataframe(df, 'response_gpt')
        df_llama, errors_llama = json_to_dataframe(df, 'response_llama')
        df_gpt_4, errors_gpt_4 = json_to_dataframe(df, 'response_gpt_4')
        
        df_gpt.to_csv(f'outputs/{name}_gpt_3_5_preprocessed.csv', index=False)
        df_llama.to_csv(f'outputs/{name}_llama_preprocessed.csv', index=False)
        df_gpt_4.to_csv(f'outputs/{name}_gpt_4_preprocessed.csv', index=False)

        if not errors_gpt.empty:
            errors_gpt.to_csv(f'outputs/{name}_gpt_3_5_preprocessed_errors.csv', index=False)
        if not errors_llama.empty:
            errors_llama.to_csv(f'outputs/{name}_llama_preprocessed_errors.csv', index=False)
        if not errors_gpt_4.empty:
            errors_gpt_4.to_csv(f'outputs/{name}_gpt_4_preprocessed_errors.csv', index=False)

        results[name] = (df_gpt, df_llama, df_gpt_4)
        errors[name] = (errors_gpt, errors_llama, errors_gpt_4)
    return results, errors

input_dfs = {
    'enem_pt_MCQ': enem_pt,
    'enem_en_MCQ': enem_en,
    'enade_pt_MCQ': enade_pt,
    'enade_en_MCQ': enade_en
}

output_dfs = process_dataframes(input_dfs)