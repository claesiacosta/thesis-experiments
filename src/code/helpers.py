import pandas as pd
import os
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

def prompt_builder_PT(topics):
    system_context = f"""
    Você é um professor especialista em criar materiais educacionais e avaliações para estudantes sobre {', '.join([str('área: '+q[0]+' - cobrindo o seguinte contexto: " '+q[1]+'"; ') for q in topics])}. 
    Seu papel envolve a criação de questões de múltipla escolha de alta qualidade. Uma questão de múltipla escolha (QCM - multiple-choice question (MCQ)) é composta por duas partes: um enunciado que identifica a pergunta ou problema, e um conjunto de alternativas ou respostas possíveis que contêm uma chave que é a melhor resposta para a pergunta, e um número de distratores que são respostas plausíveis, mas incorretas para a pergunta.

    Questões de múltipla escolha devem ser claras, concisas e gramaticalmente corretas. Elas devem ser facilmente compreensíveis para os alunos e livres de complexidade ou ambiguidade desnecessárias. Aqui estão algumas diretrizes a seguir:
    1. Mantenha as frases curtas e diretas.
    2. Use uma terminologia consistente ao se referir ao mesmo item ou conceito várias vezes.
    3. Forneça contexto suficiente para cada pergunta, garantindo que todos os termos e ações estejam dentro do escopo do tópico.
    4. Garanta que os distratores apresentem concepções equivocadas distintas relacionadas ao tópico.
    5. Evite fornecer muitas dicas nas opções de resposta, incentivando os alunos a confiar em seus conhecimentos e habilidades de raciocínio para selecionar a resposta correta.
    
    Níveis de Dificuldade:
    Questões de múltipla escolha devem obedecer a diretrizes específicas para garantir níveis adequados de dificuldade:
    1. Beginner: As questões devem ser simples, com respostas corretas óbvias e distratores facilmente distinguíveis.
    2. Intermediate: As questões devem ser mais desafiadoras, com respostas corretas não sendo excessivamente difíceis de determinar, mas distratores gerando dúvidas.
    3. Advanced: As questões devem ser altamente complexas, exigindo conhecimento substancial do tópico para identificar respostas corretas, com distratores desencorajando suposições sem informações.

    Taxonomia de Bloom:
    Questões de múltipla escolha devem avaliar habilidades no nível cognitivo apropriado de acordo com a Taxonomia de Bloom. 
    A Taxonomia de Bloom categoriza a profundidade de aprendizado e orienta a seleção de verbos de ação para objetivos. A taxonomia consiste em seis níveis:
    1. Remember: Recuperar, reconhecer e recordar conhecimento da memória.
    2. Understand: Construir significado a partir de mensagens por meio de interpretação, exemplificação, sumarização, etc.
    3. Apply: Executar ou implementar procedimentos.
    4. Analyze: Dividir material em partes, determinar relacionamentos e estrutura geral.
    5. Evaluate: Fazer julgamentos com base em critérios por meio de verificação e crítica.
    6. Create: Combinar elementos para formar um todo coerente ou gerar novas estruturas.

    Formato de Saída:
    A geração de QCM deve cobrir: Para cada contexto e área deve abranger perguntas em cada nível de dificuldade e da taxonomia de Bloom - gerando 5 perguntas, cobrindo todos os aspectos relevantes para os alunos, cada pergunta deve ter 5 alternativas. 
    
    Sua resposta deve seguir este modelo no formato JSON e as QCMs geradas devem estar em PORTUGUÊS, o mesmo idioma do tópico e descrição do tópico. Forneça o JSON completo, fechando todas as chaves.
    {{"perguntas":[{{"pergunta": str,"bloom_level": str,"difficult_level": str,"options": [{{"A": str,"B": str,"C": str,"D": str,"E": str}}],"correct_answer":[{{letra_da_opcao_correta: texto_da_opcao_correta}}]}},{{"pergunta": str,"bloom_level": str,"difficult_level": str,"options": [{{"A": str,"B": str,"C": str,"D": str,"E": str}}],"correct_answer":[{{letra_da_opcao_correta: texto_da_opcao_correta}}]}}]}}  
    """

    user_prompt = f"""Gere questões de múltipla escolha de alta qualidade que sigam isso:
    {'. '.join([str('Para o contexto '+q[1]+' da área '+q[0]+' - deve abranger perguntas em cada nível de dificuldade e da taxonomia de Bloom, cobrindo todos os aspectos relevantes para os alunos, cada pergunta deve ter 5 alternativas.') for q in topics])}.
    As questões de múltipla escolha geradas devem estar em PORTUGUÊS, o mesmo idioma do area e do contexto.
    Forneça apenas o JSON completo, fechando todas as chaves, nao forneça nenhuma informação ou nenhum texto introdutorio adicional.
    """

    return system_context, user_prompt


def prompt_builder(topics):
    system_context = f"""
    You are a teacher specialist in creating educational materials and assessments for students about {', '.join([str('area: '+q[0]+' - covering the following context: " '+q[1]+'"; ') for q in topics])}. 
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
    MCQ generate should cover: For each context and area, it must cover questions at each difficulty level and Bloom's taxonomy - in this case, for each context, 18 questions will be generated in a combination of the 3 difficulty levels and 6 bloom taxonomy (3x6 = 18), covering all aspects relevant to students, each question must have 5 alternatives.
    
    Your response must follow this template in JSON format and MCQs generated should be in ENGLISH LANGUAGE, the same language as the topic and topic description. Provide the full JSON, closing all curly braces:
    {{"questions":[{{"question": str,"bloom_level": str,"difficult_level": str,"options": [{{"A": str,"B": str,"C": str,"D": str,"E": str}}],"correct_answer":[{{correct_answer_letter: correct_answer_text}}]}},{{"question": str,"bloom_level": str,"difficult_level": str,"options": [{{"A": str,"B": str,"C": str,"D": str,"E": str}}],"correct_answer":[{{correct_answer_letter: correct_answer_text}}]}}]}}
    """

    user_prompt = f"""Generate a top quality multiple-choice questions that follow this:

    {'. '.join([str('For each context '+q[1]+' of the area '+q[0]+' - it must cover questions at each difficulty level and Bloom''s taxonomy, covering all aspects relevant to students, each question must have 5 alternatives.') for q in topics])}.
    MCQs generated should be in ENGLISH LANGUAGE, the same language as the area and context.
	Provide the full JSON, closing all curly braces, do not provide any additional information or introductory text.
    """

    return system_context, user_prompt

def evaluate_generated_mcq(reference_summary, candidate_summary):
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True) # ROUGE-L measures the longest common subsequence (LCS) between the candidate text and the reference text.
    rouge = rouge.score(reference_summary, candidate_summary)

    bleu = sentence_bleu([reference_summary.split()], candidate_summary.split())

    return rouge, bleu

def evaluator_builder_PT(questions):
    system_context = f"""
    Você é um rígido avaliador com experiência em ensino. 
    Seu papel envolve a avaliação de questões de múltipla escolha (QCM) geradas de alta qualidade. Uma questão de múltipla escolha (QCM) é composta por duas partes: um enunciado que identifica a pergunta ou problema, e um conjunto de alternativas ou possíveis respostas que contêm uma chave que é a melhor resposta para a pergunta, e um número de distratores que são respostas plausíveis, mas incorretas para a pergunta.
    
    Sua avaliação deve se concentrar em cinco métricas de qualidade:
    1. Relevance(1-5): A QCM está relacionada à descrição do contexto-tópico.
    2. Adherence(1-5): Nível de taxonomia de Bloom e dificuldade são atribuídos adequadamente considerando a informação da QCM.
    3. Answerability(1-5): A QCM fornece informações suficientes para chegar a uma resposta.
    4. Correctness(yes-no): A opção marcada como correta é realmente correta.
    Para cada métrica, atribua uma pontuação apropriada. Além disso, feedback relacionado é bem-vindo, explique cada pontuação.

    Formato de Saída:
    Sua resposta deve seguir este modelo no formato JSON. Forneça o JSON completo, fechando todas as chaves:
    {{
        "relevance": float,
        "adherence": float,
        "answerability": float,
        "correctness": str,
        "feedback": str
    }}  
    """
    user_prompt = f"""Avalie questões de múltipla escolha de alta qualidade relacionadas:
    Os dados abaixo contêm a área, contexto, perguntas e opções, também o nível de bloom e dificuldade.
    {questions}
    Forneça o JSON completo, fechando todas as chaves.
    """

    return system_context, user_prompt


def evaluator_builder_EN(questions):
    system_context = f"""
    You are an annotator with teaching experience. 
    Your role involves on evaluate top quality generate multiple-choice questions. A multiple-choice question (MCQ) is composed of two parts: a stem that identifies the question or problem, and a set of alternatives or possible answers that contain a key that is the best answer to the question, and a number of distractors that are plausible but incorrect answers to the question.
    
    Your evaluation should focus on four quality metrics:
    1. Relevance(1-5): MCQ is related to context-topic description 
    2. Adherence(1-5): bloom taxonomy level and difficult are assigned appropriated considering the MCQ information.
    3. Answerability(1-5): MCQ provides enough information to arrive at an answer
    4. Correctness(0-1): Option marked as correct is actually correct, 0 is true, 1 is false.
    For each metric assign an appropriated score. Also feedback related is welcome.

    Output Format:
    Your response should follow this template in json format. Provide the full JSON, closing all curly braces:
    {{
        "relevance": float,
        "adherence": float,
        "answerability": float,
        "correctness": float,
        "feedback": str
    }}  
    """
    user_prompt = f"""Evaluate a top quality multiple-choice questions related to the topic :
    The data below contains the area, context, question, options, also bloom and difficulty level, in this order.
    {questions}
	Provide the full JSON, closing all curly braces. 
    """

    return system_context, user_prompt

def solve_builder_PT(questions):
    system_context = f"""
    Você é um estudante e seu professor pediu para você completar todas as questões de múltipla escolha (QCM - multiple-choice question (MCQ)). 
    Seu objetivo é responder a essas perguntas da melhor maneira possível, demonstrando seu entendimento sobre o contexto e pergunta, também forneça o passo a passo para chegar na solução, não adicione citação direta.

    Formato de Saída:
    Sua resposta deve seguir este exemplo modelo no formato JSON. Forneça o JSON completo, fechando todas as chaves:
	{{"answer_letter": str,  "answer_text": str, "steps_answer": str}}
    """
    user_prompt = f"""Responda a todas as questões de múltipla escolha, mantendo todas as informações fornecidas:
    Os dados abaixo contêm as questões de múltipla escolha. Por favor, responda a todas as seguintes perguntas. Suas respostas devem ser concisas e precisas. 
    {questions}
    Forneça o JSON completo, fechando todas as chaves.
    """

    return system_context, user_prompt

def solve_builder_EN(questions):
    system_context = f"""
    You are a student and your teacher has asked you to complete all MCQs test. 
    Your goal is to answer these questions to the best of your ability, demonstrating your understanding of the context and question, also provide a step-by-step guide to reach the solution, do not add direct quote.

    Output Format:
    Your response should follow this example template in json format. Provide the full JSON, closing all curly braces:
	{{"answer_letter": str,  "answer_text": str, "steps_answer": str}}
    """
    user_prompt = f"""Answer all multiple-choice questions, keeping all the information passed:
    The data below contains the multiple-choice questions. Please answer all the following questions. Your responses should be concise and accurate.
    {questions} 
	Provide the full JSON, closing all curly braces. 
    """

    return system_context, user_prompt

def translate_PT_EN(infos):
    system_context = f"""
    Você é um experiente tradutor formado em Letras - Lingua Inglesa. 

    Seu papel envolve a traduzir para ingles questões de múltipla escolha (QCM). Uma questão de múltipla escolha (QCM) é composta por duas partes: um enunciado que identifica a pergunta ou problema, e um conjunto de alternativas ou possíveis respostas que contêm uma chave que é a melhor resposta para a pergunta, e um número de distratores que são respostas plausíveis, mas incorretas para a pergunta.
    Além do contexto, área, bloom's taxonomy e difficult level.

    Formato de Saída:
    Sua resposta seguir esse template no formato JSON. Forneça o JSON completo, fechando todas as chaves. Também forneça apenas as tradução de cada item da lista e colocado com aspas duplas, seguindo a mesma ordem, não forneça nenhuma informação adicional:
    {{"1": str,  "2": str, "3": str, "...": str}}
	"""
    user_prompt = f"""Traduza as informações abaixo de português para inglês:
    Os dados abaixo contêm todas as informações em tipo lista, cada item da lista deve ser traduzido, e colocado com aspas duplas.
    {infos}
    Forneça o JSON completa com apenas as traduções, não forneça nenhuma informação adicional.
    """

    return system_context, user_prompt