# %%
import os, sys
import inspect
sys.path.append("..")

src_file_path = inspect.getfile(lambda: None)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(src_file_path))))

import pandas as pd 
import numpy as np
import pickle
from transformers import BertTokenizerFast
import tables
import random
import numpy as np
import argparse
import json
import tables
import os
import re
from tqdm import tqdm
import pickle as pkl
from transformers import BertTokenizer
from sklearn.utils import shuffle
import sys
from data_loader import load_dict, save_vecs



alephbert_tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')
# with open('/home/izmaylov/Thesis/Preprocess/40K data/Non_labeled.pkl', 'rb') as f:
#     df = pickle.load(f)
with open('/home/izmaylov/Thesis/Preprocess/40K data/Non_labeled_with_id.pkl', 'rb') as f:
    df = pickle.load(f)

# %%
df.Transcript.iloc[0]

# %%
#read json file
with open('/home/izmaylov/Thesis/SR_BERT/data/Data_2/id_dict.json', 'rb') as f:
    id_dict = json.load(f)


# %%
exclude_id= id_dict["test"]+id_dict["valid"]

# %%
#exclude test and validation
df = df[~df['Engagement ID'].isin(exclude_id)]

# %%
left_colums=['דיכאון 21      ',
 'מחשבות אובדניות',
 'פגיעה עצמית',
 'ניסיון אובדני ק',
 'היעדר שייכות-מע',
 'בריונות/שיימינג',
 'היעדר תקווה- מע',]


lex_df=pd.read_csv("/home/izmaylov/Thesis/lexicon_sim/gsr_lex_4.5.csv")
lex_dict={}
i=0
for col in lex_df.columns:
    i+=1
    examples= lex_df[col].dropna()
    try:
        gender= examples[0]
    except:
        print(examples)
    examples=examples[1:]
    key= col[:15]
    if key[-2:]==".1":
        key=key[:-2]
    if key in lex_dict:
        lex_dict[key]+=[example.strip() for example in examples]
    else:
        lex_dict[key]=[example.strip() for example in examples]
for key in lex_dict.keys():
    lex_dict[key]=list(filter(None, lex_dict[key]))

#leave only the keys that are in left_colums
lex_dict={key:lex_dict[key] for key in lex_dict.keys() if key in left_colums} 

# %%
#print how many  cores
import multiprocessing
print(multiprocessing.cpu_count())

# %%
def strip_white(df):
    # print(df)
    df["text"]=df["text"].replace(r'\s+', ' ', regex=True)
    return df

df.Transcript=df.Transcript.apply(strip_white)

# %%
from multiprocesspandas import applyparallel
from tqdm import tqdm
tqdm.pandas()
from tqdm import tqdm

def count_contains(converasion, lex_dict):
    for k,v in lex_dict.items():
        to_return=[]
        for text in converasion.text:
            n=0
            for feature in v:
                if re.search(r"\b{}\b".format(feature), text):
                    n+=1
            to_return.append(n)
        converasion[k]=to_return
    return converasion
    
# ls=df.apply_parallel(lambda x: count_contains(x,lex_dict),num_processes=1)
# ls=df.Transcript.apply_parallel(lambda x: count_contains(x,lex_dict),num_processes=32)
# ls=ls.droplevel(1)
# ls=ls.reset_index()
# for i in tqdm(range(0,len(df))):
#     df.Transcript.iloc[i]= ls[ls["index"]==i]

# %%
# df.to_pickle("/home/izmaylov/Thesis/Preprocess/40K data/Non_labeled_with_lex.pkl")
df= pd.read_pickle("/home/izmaylov/Thesis/Preprocess/40K data/Non_labeled_with_lex.pkl")

# %%
from sklearn.model_selection import train_test_split

train_data, test_data= train_test_split(df,test_size=0.05,random_state=42)
test_data, valid_data= train_test_split(test_data,test_size=0.5,random_state=42)


# %%
#save the engagment id if the train test and validation data in json
id_dict={}
id_dict["train"]=train_data["Engagement ID"].tolist()
id_dict["test"]=test_data["Engagement ID"].tolist()
id_dict["valid"]=valid_data["Engagement ID"].tolist()
#save the dict to json 
with open('/home/izmaylov/Thesis/SR_BERT/data/Sahar_30K_4.5/id_dict.json', 'w') as fp:
    json.dump(id_dict, fp)

# %%
class Index(tables.IsDescription):
    pos_utt = tables.Int32Col() # start offset of an utterance
    res_len = tables.Int32Col() # number of tokens till the end of response
    ctx_len = tables.Int32Col() # number of tokens from the start of dialog 

class Labels(tables.IsDescription):
    GSR = tables.IntCol() # start offset of an utterance
    IMSR = tables.IntCol() # number of tokens till the end of response



def compact_dialog(conv):
    conv["texter"]=conv["texter"].replace({0:"A","False":"A","True":"B",1:"B",False:"A",True:"B"})
    # conv["features"]= [np.zeros((1, 1)) for i in range(len(conv.index))]
    conv["features"]=conv[left_colums].astype(str).apply(','.join, axis=1)
    lst=conv[["texter","text","features"]].to_records(index=False)
    dialog = {'knowledge': '', 'utts': lst}
    return dialog

def get_Sahar_data(df):
    dialogs=[] 
    for i in range(len(df.index)):
        dialog=df.iloc[i].Transcript
        if len(dialog.index)<20:
            continue
        res= compact_dialog(dialog)
        # res["IMSR"]=df.iloc[i].IMSR
        dialogs.append(res)
    return dialogs

train_data= get_Sahar_data( train_data)
test_data= get_Sahar_data(test_data)
valid_data= get_Sahar_data(valid_data)



# %%
len(train_data),len(valid_data),len(test_data)

# %%
           
def binarize(dialogs, tokenizer, output_path):
    """binarize data and save the processed data into a hdf5 file
       :param dialogs: an array of dialogs, 
        each element is a list of <caller, utt, feature> where caller is a string of "A" or "B",
        utt is a sentence, feature is an 2D numpy array 
    """
    with tables.open_file(output_path, 'w') as f:
        filters = tables.Filters(complib='blosc', complevel=5)
        arr_contexts = f.create_earray(f.root, 'sentences', tables.Int32Atom(),shape=(0,),filters=filters)
        indices = f.create_table("/", 'indices', Index, "a table of indices and lengths")
        pos_utt = 0
        for i, dialog in enumerate(tqdm(dialogs)):
            n_tokens=0
            ctx_len=0
            for k, (caller, utt, feature) in enumerate(dialog['utts']):
                floor = -1 if caller == 'A' else -2
                # print("caller: ",caller)
                # print("utt: ",utt)
                idx_utt = tokenizer.encode(utt[:tokenizer.max_len_single_sentence])
                # print(tokenizer.decode(idx_utt))
                if idx_utt[0]!=tokenizer.cls_token_id: idx_utt = [tokenizer.cls_token_id] + idx_utt
                # print(tokenizer.decode(idx_utt))
                arr_contexts.append([floor])
                # idx_utt.append([int(j) for j in feature.split(",")])
                # print(idx_utt)
                # idx_utt+= [-int(j) for j in feature.split(",")]
                # print(idx_utt)
                idx_utt=idx_utt[:-1]+ [-int(j) for j in feature.split(",")]+ [idx_utt[-1]]
                # arr_contexts.append([-int(j) for j in feature.split(",")])

                if idx_utt[-1]!=tokenizer.sep_token_id: idx_utt = idx_utt + [tokenizer.sep_token_id]
                arr_contexts.append(idx_utt)
                n_tokens+=len(idx_utt)+1
                if k>0: # ignore the first utterance which has no context
                    ind = indices.row
                    ind['pos_utt'] = pos_utt
                    ind['res_len'] = len(idx_utt)+1
                    ind['ctx_len'] = ctx_len   
                    ind.append()
                ctx_len+=len(idx_utt)+1
                pos_utt += len(idx_utt)+1
            ctx_len=0

data_dir="../data/Sahar_30K_4.5"
# os.path(data_dir).mkdir(parents=True, exist_ok=False)

train_out_path = os.path.join(data_dir, "train.h5")
train_data_binary=binarize(train_data, alephbert_tokenizer, train_out_path)

valid_out_path = os.path.join(data_dir, "valid.h5")
valid_data_binary=binarize(valid_data, alephbert_tokenizer, valid_out_path)

test_out_path = os.path.join(data_dir, "test.h5")
test_data_binary=binarize(test_data, alephbert_tokenizer, test_out_path)
# # from pathlib import Path

# data_dir="data/Sahar_balanced"
# Path(data_dir).mkdir(parents=True, exist_ok=True)

# train_out_path = os.path.join(data_dir, "train.pkl")
# train_data_binary=conv_saver(train_data, alephbert_tokenizer, train_out_path)

# valid_out_path = os.path.join(data_dir, "valid.pkl")
# valid_data_binary=conv_saver(valid_data, alephbert_tokenizer, valid_out_path)

# test_out_path = os.path.join(data_dir, "test.pkl")
# test_data_binary=conv_saver(test_data, alephbert_tokenizer, test_out_path)



# %%
# table = tables.open_file(train_out_path)
# # contexts = table.get_node('/sentences')[:].astype(np.long)
# # label = table.get_node('/label')[:].astype(np.long)
# # index = table.get_node('/indices')[:]
# table.close()

#  
with tables.open_file(train_out_path, 'r') as f:
    contexts_m= f.get_node('/sentences')[:].astype(np.long)
    # label_m = f.get_node('/labels')[:]
    index_m = f.get_node('/indices')[:]

# %%
index_m

# %%
len(contexts_m)

# %%
# pos_utt = tables.Int32Col() # start offset of an utterance
# res_len = tables.Int32Col() # number of tokens till the end of response
# ctx_len = tables.Int32Col() # number of tokens from the start of dialog

# %%
for offset in range(0,1):
    index = index_m[offset]
    label= label_m[offset]
    
    pos_utt, ctx_len, res_len,  = index['pos_utt'], index['ctx_len'], index['res_len']
    print(pos_utt, ctx_len, res_len)
    # pos_knowl, knowl_len = index['pos_knowl'], index['knowl_len']
    ctx_arr=contexts_m[pos_utt-ctx_len:pos_utt].tolist()
    res_arr=contexts_m[pos_utt:pos_utt+res_len].tolist()
    # knowl_arr =knowledge[pos_knowl:pos_knowl+knowl_len].tolist()

# %%
    context = []
    tmp_utt = []
    for i, tok in enumerate(ctx_arr):
        tmp_utt.append(ctx_arr[i])
        if tok == alephbert_tokenizer.sep_token_id:
            floor = tmp_utt[0]
            tmp_utt = tmp_utt[1:] 
            utt_len = len(tmp_utt) # floor is not counted in the utt length
            utt = tmp_utt[:utt_len]            
            context.append(utt)  # append utt to context          
            tmp_utt=[]  # reset tmp utt
    response = res_arr[1:] # ignore cls token at the begining            
    res_len = len(response)
    response = response[:res_len-1] + [alephbert_tokenizer.sep_token_id] 

# %%
list(map(alephbert_tokenizer.decode, context))

# %%
alephbert_tokenizer.decode(response)

# %%


# %%



