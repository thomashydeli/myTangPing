import os
import re
import shutil
import psutil
import boto3
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from gensim.parsing.preprocessing import STOPWORDS
from copy import deepcopy
from rank_bm25 import *
import json
import openai
from flask import Flask, request, jsonify, render_template
import pandas as pd
import jieba

# global variable
RETRY=5 # for retrying
MAXLEN=4096 # controlling the maximum token from GPT4 output
CHAT_HISTORY=[] # empty list initiated to preserve chat history
N_TOKENS=500 # number of tokens for chunck-size the text

# loading secrets
openai.api_key=os.environ.get('openai')
aws_access_key_id=os.environ.get('aws_access_key_id')
aws_secret_access_key=os.environ.get('aws_secret_access_key')

def print_process_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    used_memory = memory_info.rss
    print(f"[INFO] Process memory usage: {used_memory / 1024**2:.2f} MB")

# main prompt powering the engine
prompt="""
假设你叫做梅（英文名是May）。你是一个非常乐于助人的心理咨询师，你的目的是帮助人们减轻他们生活中的焦虑。
今年来，尤其是新冠疫情之后，再加上现在全球下行的宏观经济，以及各个公司的裁员大潮，人们总是处在各种生活和社会问题的漩涡之中，以至于大家活在迷茫和焦虑之中。
作为心理咨询师，你的任务就是以你了解到的故事为依据，为人们答疑解惑，来为人们减轻心理上的焦虑和痛苦。

故事:
{context}

根据上面的故事，以及你的任务，请你回答下面的问题。在回答问题的时候，如果有被问到并且需要讲故事的话，那么就根据上面的几个故事，给提问者讲一个故事{lang_plugin0}，一定要像讲故事一样讲给提问者（不要说这是总结，你可以说“我所了解到”、”从我听到过的故事中“、”我知道有些诸如“等其他类似的表达方式。同时你可以选择只挑一个故事讲，或者把几个故事拼在一起总结式地讲），然后再回答他们的问题:
{question}{chat_history}{language_plugin}
"""

lang_prompt="""你能把这个问题翻译成简体中文么？问题是：{question}"""
lang_detection="""
这个问题的主要内容是用什么语言提问的：{question}
如果是简体中文，请回答zh-cn；繁体中文，请回答zh-tw；英文，请回答en
"""

# creating text corpus by pulling data from s3 with my personal documents
s3 = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)
obj_list=s3.list_objects(Bucket='mydatabucket2023')

# download personal documents temporarily
corpus=[]
resume_token=''
file_names=[obj['Key'] for obj in obj_list['Contents'] if 'stories.csv' in obj['Key']]
for i,f in enumerate(file_names):
    s3.download_file('mydatabucket2023',f,'stories.csv')
    stories=pd.read_csv('stories.csv')
    corpus=list(stories.text.values)

# function to pull response using GPT-4
def getResponse(prompt, max_tokens, temperature=0.2):
    completion=openai.ChatCompletion.create(
        model='gpt-4',
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=temperature
    )
    response=completion.choices[0]['message']['content'].strip()
    return response

# getting conversation responses using GPT-3.5-turbo
def getResponse3(prompt, max_tokens, temperature=0.2):
    completion=openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=temperature
    )
    response=completion.choices[0]['message']['content'].strip()
    return response

# 读取停用词文件
def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        stopwords = [line.strip() for line in f.readlines()]
    return set(stopwords)

# 分词和过滤停用词
def tokenize_and_filter_stopwords(text, stopwords):
    words = jieba.cut(text)
    filtered_words = [word for word in words if word not in stopwords]
    return filtered_words

def remove_numbers_punctuation_and_alphabets(text):
    # 使用正则表达式匹配数字、标点符号和英文字母
    pattern = r'[0-9a-zA-Z]|[^\u4e00-\u9fa5。，？！]'
    clean_text = re.sub(pattern, '', text)
    return clean_text

# query processor for chaining the cleaning functionalities
def query_processor(query):
    clean_text = remove_numbers_punctuation_and_alphabets(query)
    filtered_words = tokenize_and_filter_stopwords(clean_text, stopwords)
    return filtered_words

# 加载停用词
stopwords = load_stopwords('baidu_stopwords.txt')

# chunck documents into pieces, and create a corpus based on it
chuncked_corpus=[]
raw_corpus=[]
for c in corpus:
    tokens=deepcopy(c)
    chuncks=[]
    N=len(tokens)
    iterations=N//N_TOKENS

    for i in range(iterations+1):
        if i!=iterations-1:
            target_tokens=tokens[i*N_TOKENS:(i+1)*N_TOKENS]
        elif i==iterations-1:
            target_tokens=tokens[i*N_TOKENS:]

        clean_text = remove_numbers_punctuation_and_alphabets(target_tokens)
        raw_corpus.append(clean_text)
        filtered_words = tokenize_and_filter_stopwords(clean_text, stopwords)
        
        chuncks.append(
            filtered_words
        )
    chuncked_corpus.extend(chuncks)

# create a search database using BM25
bm25=BM25Okapi(chuncked_corpus)

app=Flask(__name__)

def process_chat_message(message, chat_history):
    print(f'[INFO] input chat history: {chat_history}')
    # process chat_history:
    if len(chat_history) == 0:
        chat_history_context=''
    elif len(chat_history) > 1:
        print(f"[WARNING] there's more than one chat history loaded")
        chat_history=[chat_history[-1]] # only use the last information
        print(f'[INFO] only the last chat history presered')
        chat_history_context=f"\n还有，这个是之前的对话记录： {chat_history[0]}"
    else:
        chat_history_context=f"\n还有，这个是之前的对话记录： {chat_history[0]}"

    print(f'[INFO] question asked was: {message}')
    lang_used=getResponse(
        prompt=lang_detection.format(question=message),
        max_tokens=50,
        temperature=0.01,
    ).lstrip().rstrip()
    print(f'[INFO] language used: {lang_used}')
    lang_plugin0=''
    lang_plugin=''
    if 'cn' not in lang_used:
        currentPrompt=lang_prompt.format(question=message)
        i=0
        while True and i < RETRY:
            try:
                translated=getResponse(
                    prompt=currentPrompt,
                    max_tokens=MAXLEN-len(currentPrompt),
                    temperature=0.2,
                ).lstrip().rstrip()
                print(f'[INFO] question translated as: {translated}')
            except Exception as e:
                print(f'[INFO] exception ecountered: {e.message}')
                i+=1
                continue
            break
        message=deepcopy(translated)
        if 'en' in lang_used:
            lang_plugin0='(这里请用英文)'
            lang_plugin='\n最后，请用英文回答这个问题（只需给出答案即可，不需要用简体中文再复述）：'
        elif 'tw' in lang_used:
            lang_plugin0='（这里请用繁体中文）'
            lang_plugin='\n最后，请用繁体中文回答这个（只需给出答案即可，不需要用简体中文再复述）：'

    query=deepcopy(message)
    query+=chat_history_context
    question=deepcopy(message)
    print(f'[INFO] ready to process the question being asked')
    tokenized_query=query_processor(query)
    print(f'[INFO] question has been processed and tokenized')
    docs = bm25.get_top_n(tokenized_query, raw_corpus, n=3) # getting relevant documentation
    print(f'[INFO] relevant contextual documents have been acquired')

    currentPrompt=prompt.format(
        context=''.join([f'{i+1}. '+d+'.\n' for i,d in enumerate(docs)]),
        question=question,
        lang_plugin0=lang_plugin0,
        language_plugin=lang_plugin,
        chat_history=chat_history_context,
    )
    currentPrompt=re.sub('死','',re.sub('自杀','',currentPrompt))
    print(f'[INFO] prompt has been constructured')
    print(f'[INFO] prompt is: {currentPrompt}')
    output=None
    i=0
    while True and i < RETRY:
        print(f'[INFO] current try: {i+1}')
        try:
            print(f'[INFO] ready to get response')
            output=getResponse3(
                prompt=currentPrompt,
                max_tokens=MAXLEN-int(len(currentPrompt)*1.2+256),
                temperature=0.2,
            ).lstrip().rstrip()
            print(f'[INFO] response acquired as {output}')
        except Exception as e:
            print(f'[INFO] exception ecountered: {e.message}')
            i+=1
            continue
        break
    print(f'[INFO] response acquired')
    if output:
        response=output
    else:
        response="很抱歉，因为技术原因，我暂时没有办法提供这个问题的答案，请您稍后再试。| Sorry, I could not answer that due to technical difficulty, please try again later."
    
    print_process_memory_usage()
    return response

@app.route('/chat', methods=['POST'])
def chat():
    message = request.json.get('message')
    if not message:
        return jsonify({"error": "No message provided"}), 400

    response = process_chat_message(message, CHAT_HISTORY)
    if len(CHAT_HISTORY) > 1:
        CHAT_HISTORY.pop()
    CHAT_HISTORY.append(f"{response}")
    return jsonify({"response": response})

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)