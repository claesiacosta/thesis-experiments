import pandas as pd
import os
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

def prompt_builder(topics):
    system_context = f"""
    You are a teacher specialist in creating educational materials and assessments for students about {', '.join([str(q[0]+' - described as '+q[1]) for q in topics])}. 
    Your role involves on creating top quality multiple-choice questions A multiple-choice question (MCQ) is composed of two parts: a stem that identifies the question or problem, and a set of alternatives or possible answers that contain a key that is the best answer to the question, and a number of distractors that are plausible but incorrect answers to the question.

    Multiple-choice questions must be clear, concise, and grammatically correct. They should be easily understandable for students and free of unnecessary complexity or ambiguity. Here are some guidelines to follow:
    1. Keep sentences short and straightforward.
    2. Use consistent wording when referring to the same item or concept multiple times.
    3. Provide sufficient context for each question, ensuring that all terms and actions are within the scope of the topic.
    4. Ensure that distractors present distinct misconceptions related to the topic.
    5. Avoid providing too many clues in the answer options, prompting students to rely on their knowledge and reasoning skills to select the correct answer.
    
    Difficulty Levels:
    Multiple-choice questions must adhere to specific guidelines to ensure appropriate difficulty levels:
    1. Beginner: Questions should be simple, with obvious correct answers and easily distinguishable distractors.
    2. Intermediate: Questions should be more challenging, with correct answers not overly difficult to determine but distractors prompting second-guessing.
    3. Advanced: Questions should be highly complex, requiring substantial topic knowledge to identify correct answers, with distractors discouraging uninformed guesses.

    Blooms’ Taxonomy:
    Multiple-choice questions should assess skills at the appropriate cognitive level according to Bloom’s Taxonomy. 
    Bloom’s Taxonomy categorizes learning depth and guides action verb selection for objectives. The taxonomy consists of six levels:
    1. Remember: Retrieve, recognize, and recall knowledge from memory.
    2. Understand: Construct meaning from messages through interpreting, exemplifying, summarizing, etc.
    3. Apply: Execute or implement procedures.
    4. Analyze: Break material into parts, determine relationships, and overall structure.
    5. Evaluate: Make judgments based on criteria through checking and critiquing.
    6. Create: Combine elements to form a coherent whole or generate new structures.

    Output Format:
    MCQ generate should cover:
    {'. '.join([str(str(q[2])+' questions about the topic '+q[0]+'. Should be at the '+ q[3]+' level in Bloom´s taxonomy and suitable for '+q[4]+' level students') for q in topics])}. 
    Your response should follow this template in json format and MCQs generated should be in PORTUGUESE LANGUAGE, the same language as the topic and topic description.:
    {{
	    "title": str,
	    "topics": [
		    {{
			    "topic": str,
                "topic_description": str,
                "bloom_level": str,
                "difficult_level": str,
			    "questions": [
				    {{
					    "context": str,
					    "question": str,
					    "options": [
						    {{
							    "option": str,
							    "correct": boolean
						    }},
						    {{
							    "option": str,
							    "correct": boolean
						    }},
						    {{
						    	"option": str,
						    	"correct": boolean
						    }},
						    {{
						    	"option": str,
						    	"correct": boolean
						    }}
					    ]
				    }}
			    ]
		    }},
		    {{
			    "topic": str,
                "topic_description": str,
                "bloom_level": str,
                "difficult_level": str,
			    "questions": [
				    {{
					    "context": str,
					    "question": str,
					    "options": [
						    {{
							    "option": str,
							    "correct": boolean
						    }},
						    {{
							    "option": str,
							    "correct": boolean
						    }},
						    {{
						    	"option": str,
						    	"correct": boolean
						    }},
						    {{
						    	"option": str,
						    	"correct": boolean
						    }}
					    ]
				    }}
			    ]
		    }}
	    ]
    }}  
    """

    user_prompt = f"""Generate a top quality multiple-choice questions that follow this:
    {'. '.join([str(str(q[2])+' questions about the topic '+q[0]+'. Should be at the '+ q[3]+' level in Bloom´s taxonomy and suitable for '+q[4]+' level students') for q in topics])}.
    MCQs generated should be in PORTUGUESE LANGUAGE, the same language as the topic and topic description.
    """

    return system_context, user_prompt

def evaluate_generated_mcq(reference_summary, candidate_summary):
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True) # ROUGE-L measures the longest common subsequence (LCS) between the candidate text and the reference text.
    rouge = rouge.score(reference_summary, candidate_summary)

    bleu = sentence_bleu([reference_summary.split()], candidate_summary.split())

    return rouge, bleu


def evaluator_builder(questions):
    system_context = f"""
    You are an annotator with teaching experience. 
    Your role involves on evaluate top quality generate multiple-choice questions. A multiple-choice question (MCQ) is composed of two parts: a stem that identifies the question or problem, and a set of alternatives or possible answers that contain a key that is the best answer to the question, and a number of distractors that are plausible but incorrect answers to the question.
    
    Your evaluation should focus on four quality metrics:
    1. Relevance(1-5): MCQ is related to context-topic description 
    2. Adherence(1-5): bloom taxonomy level and difficult are assigned appropriated considering the MCQ information.
    3. Grammar(1-5): MCQ i gramatically correct
    4. Answerability(1-5): MCQ provides enough information to arrive at an answer
    5. Correctness(1-5): Option marked as correct is actually correct
    For each metric assign an appropriated score. Also feedback related is welcome.

    Output Format:
    Your response should follow this template in json format:
    {{
	    "title": str,
	    "topics": [
		    {{
			    "topic": str,
                "topic_description": str,
                "bloom_level": str,
                "difficult_level": str,
			    "questions": [
				    {{
					    "context": str,
					    "question": str,
					    "options": [
						    {{
							    "option": str,
							    "correct": boolean
						    }},
						    {{
							    "option": str,
							    "correct": boolean
						    }},
						    {{
						    	"option": str,
						    	"correct": boolean
						    }},
						    {{
						    	"option": str,
						    	"correct": boolean
						    }}
					    ],
                        "relevance": float,
                        "adherence": float,
                        "grammar": float,
                        "answerability": float,
                        "correctness": float,
                        "feedback": str
				    }}
			    ]
		    }},
		    {{
			    "topic": str,
                "topic_description": str,
                "bloom_level": str,
                "difficult_level": str,
			    "questions": [
				    {{
					    "context": str,
					    "question": str,
					    "options": [
						    {{
							    "option": str,
							    "correct": boolean
						    }},
						    {{
							    "option": str,
							    "correct": boolean
						    }},
						    {{
						    	"option": str,
						    	"correct": boolean
						    }},
						    {{
						    	"option": str,
						    	"correct": boolean
						    }}
					    ],
                        "relevance": float,
                        "adherence": float,
                        "grammar": float,
                        "answerability": float,
                        "correctness": float,
                        "feedback": str
				    }}
			    ]
		    }}
	    ]
    }}
    """
    user_prompt = f"""Evaluate a top quality multiple-choice questions related to the topic :
    The data below contains the the topic, topic description, questions and options, also bloom and difficulty level.
    Your output MCQs should be in PORTUGUESE LANGUAGE, the same language as the topic and topic description.
    {questions} 
    """

    return system_context, user_prompt

def solve_builder(questions):
    system_context = f"""
    You are a student and your teacher has asked you to complete all MCQs test. 
    Your goal is to answer these questions to the best of your ability, demonstrating your understanding of the topic.

    Output Format:
    Your response should follow this template in json format:
    {{
	    "title": str,
	    "topics": [
		    {{
			    "questions": [
				    {{
					    "question": str,
					    "options": [
						    {{
							    "option": str
						    }},
						    {{
							    "option": str
						    }},
						    {{
						    	"option": str
						    }},
						    {{
						    	"option": str
						    }}
					    ],
                        "answer": str
				    }}
			    ]
		    }},
		    {{
			    "questions": [
				    {{
					    "question": str,
					    "options": [
						    {{
							    "option": str
						    }},
						    {{
							    "option": str
						    }},
						    {{
						    	"option": str
						    }},
						    {{
						    	"option": str
						    }}
					    ],
                        "answer": str
				    }}
			    ]
		    }}
	    ]
    }}
    """
    user_prompt = f"""Answer all multiple-choice questions, keeping all the information passed:
    The data below contains the multiple-choice questions. Please answer all the following questions. Your responses should be concise and accurate.
    {questions} 
    """

    return system_context, user_prompt