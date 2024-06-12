# %%
import os, sys
import inspect
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
sys.path.append("..")
from data_loader import load_dict, save_vecs
alephbert_tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')



# alephbert_tokenizer = BertTokenizerFast.f rom_pretrained('onlplab/alephbert-base')
# with open('/home/izmaylov/Thesis/Preprocess/data/September_Dataset/result_combine.pkl', 'rb') as f:
#     temp_df = pickle.load(f)

with open('/home/izmaylov/Thesis/Preprocess/data/Jan22/labeld_with_IMSR.pkl', 'rb') as f:
    df = pickle.load(f)

def Clear_until_first_texter(row):
     row["text"]=row[
         "text"][(row["text"].texter == 0).idxmax():]
     row["number of rows"]=len(row["text"].index)
     return row

# df=temp_df.apply(lambda row : Clear_until_first_texter(row), axis = 1)
# df=df[df["number of rows"]>10]
def Clear_until_first_texter2(row):
    row.reset_index(drop=True, inplace=True)
    # display(row)
    row.texter= row.texter.astype(int)
    row=row[(row.texter == 0).idxmax():]
    # if len(row.index)>10:
    row=CombineRows(row)
    return row
    # else:


def CombineRows(text_df):
  text_df["del"]=False
  texter_col=text_df.columns.get_loc('texter')
  text_col=text_df.columns.get_loc('text')
  del_col=text_df.columns.get_loc('del')
  for i in range(len(text_df)-2,0,-1):
    if text_df.iloc[i+1,texter_col]== text_df.iloc[i,texter_col] :
      text_df.iloc[i,text_col]=text_df.iloc[i,text_col]+" "+ text_df.iloc[i+1]["text"]
      text_df.iloc[i+1,del_col]=True
  text_df=text_df[text_df["del"]==False]
  text_df.drop(["del"], axis=1, inplace=True)
  return text_df

def Clear_until_first_texter(row):
    # display(row)
    row["text"]=Clear_until_first_texter2(row["Transcript"])

    row["number of rows"]=len(row["text"].index)
    return row

# df=temp_df.apply(lambda row : Clear_until_first_texter(row),axis = 1)
# Clear_until_first_texter2(temp_df.iloc[0].Transcript)
df["GSR"]=df["GSR"].astype(float).astype(int)
df.loc[(df.IMSR==1) & (df.GSR==0), 'GSR'] = 1

# %%
#read json file 
with open('/home/izmaylov/Thesis/SR_BERT/data/Data_2/id_dict.json', 'r') as fp:
    id_dict = json.load(fp)


# %%


# %%
left_colums=['IPT2 תפיסת העצמ', 'פגיעה עצמית', 'חוסר תקווה- מעל', 'ניסיון אובדני ק',
       'מחשבות אובדניות',"GSR"]
lex_df=pd.read_csv("/home/izmaylov/Thesis/lexicon_sim/older_ver/gsr_lex_4.csv")
lex_df
lex_dict={}
for col in lex_df.columns:
    if col[:15] in left_colums:
        examples= lex_df[col].dropna()
        lex_dict[col[:15]]=[set(example.split(" ")) for example in examples]

# %%
from multiprocesspandas import applyparallel
from tqdm import tqdm
tqdm.pandas()
from tqdm import tqdm
# for i in tqdm(range (0,len(df))):
#     df.iloc[i]["tokens"]=df.iloc[i].text.apply(lambda row : set(row.split(" ")))
    
def check_contains(converasion, lex_dict):
    for k,v in lex_dict.items():
        to_return=[]
        # print(converasion)
        for text in converasion.text:
            tokens= text.split(" ")
            flag=False
            for feature in v:
                if flag==False and feature.issubset(tokens):
                    flag=True
            to_return.append(int(flag))
        converasion[k]=to_return
    return converasion
    
def check_contains2(converasion, lex_dict):
    for k,v in lex_dict.items():
        to_return=[]
        # print(converasion)
        for text in converasion.text:
            flag=False
            to_return.append(int(flag))
        converasion[k]=to_return
    return converasion

df["Transcript"]=df.Transcript.progress_apply(lambda x: check_contains2(x,lex_dict)) 


# %%
left_colums

# %%
df["gender"]=df["gender"].apply(lambda x: 1 if x=="גבר" else 0)
# add the gender to each converasion
for i in tqdm(range (0,len(df))):
    df.iloc[i].conv

# %%

def compact_dialog(conv):
    conv["texter1"]=conv["texter"].map({0:"A","False":"A","True":"B",1:"B",False:"A",True:"B"})
    # print(conv["texter"]
    # conv["features"]= [np.zeros((1, 1)) for i in range(len(conv.index))]
    conv["features"]=conv[left_colums[:-1]].astype(str).apply(','.join, axis=1)
    # conv["features"]=conv["gender"].astype(str).apply(','.join, axis=1)
    lst=conv[["texter1","text","features"]].to_records(index=False)
    dialog = {'knowledge': '', 'utts': lst}
    return dialog

def get_Sahar_data_turns(df):
    dialogs=[] 
    for i in range(len(df.index)):
        dialog=df.iloc[i].Transcript
        if len(dialog.index)<20:
            continue

        res= compact_dialog(dialog)
        res["GSR"]=df.iloc[i].GSR
        res["IMSR"]=df.iloc[i].IMSR
        # res["VED"]=df.iloc[i]["Engagement ID"]
        # res["IMSR"]=df.iloc[i].IMSR
        dialogs.append(res)
    return dialogs


# ready_data= get_Sahar_data_turns(df)
# split the data to train, test and valid
train_data=df[df["Engagement ID"].isin(id_dict["train"])]
train_data=get_Sahar_data_turns(train_data)
test_data=df[df["Engagement ID"].isin(id_dict["test"])]
test_data=get_Sahar_data_turns(test_data)
valid_data=df[df["Engagement ID"].isin(id_dict["valid"])]
valid_data=get_Sahar_data_turns(valid_data)
print("train_data",len(train_data),"test_data",len(test_data),"valid_data",len(valid_data))


# %%

from pathlib import Path

def conv_saver(dialogs, tokenizer, output_path):
    """binarize data and save the processed data into a hdf5 file
       :param dialogs: an array of dialogs, 
        each element is a list of <caller, utt, feature> where caller is a string of "A" or "B",
        utt is a sentence, feature is an 2D numpy array 
    """

    preprosses_dialogs=[]
    labels= []
    for i, dialog in enumerate(tqdm(dialogs)):
        # print(dialog)
        labels.append((float(dialog["GSR"]),float(dialog["IMSR"])))
        # IMSR=dialog["IMSR"]
        dialog_message= []
        for k, (caller, utt, feature) in enumerate(dialog['utts']):
            floor = -1 if caller == 'A' else -2
            idx_utt = tokenizer.encode(utt[:tokenizer.max_len_single_sentence])
            if idx_utt[0]!=tokenizer.cls_token_id: idx_utt = [tokenizer.cls_token_id] + idx_utt
            idx_utt.insert(0,floor)
            idx_utt.append([int(j) for j in feature.split(",")])

            dialog_message.append(idx_utt)
        # print(dialog_message)
        preprosses_dialogs.append(dialog_message)
        
    with open(output_path, 'wb') as f:
        pickle.dump((labels,preprosses_dialogs), f)
    # print(labels)
    return (labels,preprosses_dialogs)


# data_dir="data/Sahar_labeld"

# train_out_path = os.path.join(data_dir, "train.h5")
# train_data_binary=binarize(train_data, alephbert_tokenizer, train_out_path)

# valid_out_path = os.path.join(data_dir, "valid.h5")
# valid_data_binary=binarize(valid_data, alephbert_tokenizer, valid_out_path)

# test_out_path = os.path.join(data_dir, "test.h5")
# test_data_binary=binarize(test_data, alephbert_tokenizer, test_out_path)
# from pathlib import Path

data_dir="../data/Sahar_2_with_imsr"
Path(data_dir).mkdir(parents=True, exist_ok=True)

train_out_path = os.path.join(data_dir, "train.pkl")
train_data_binary=conv_saver(train_data, alephbert_tokenizer, train_out_path)
# train_out_path = os.path.join(data_dir, "train_un.pkl")

# with open(train_out_path, 'wb') as f:
#         pickle.dump(train_data, f)


valid_out_path = os.path.join(data_dir, "valid.pkl")
valid_data_binary=conv_saver(valid_data, alephbert_tokenizer, valid_out_path)
# train_out_path = os.path.join(data_dir, "valid_un.pkl")

# with open(valid_out_path, 'wb') as f:
#         pickle.dump(valid_data, f)


test_out_path = os.path.join(data_dir, "test.pkl")
test_data_binary=conv_saver(test_data, alephbert_tokenizer, test_out_path)
# test_out_path = os.path.join(data_dir, "test_un.pkl")

# with open(test_out_path, 'wb') as f:
#         pickle.dump(test_data, f)
# conv_saver(test_data, alephbert_tokenizer, test_out_path)[1]

# %%
t=pickle.load(open("data/Sahar_balanced/train.pkl", "rb"))
labels=t[0]
dialogs=t[1]


# %%
gsr, ved = zip(*labels)

# %%
type(ved[3])

# %%
import math
for i in gsr:
    if  math.isnan(i):
        print(i)

# %%
# table = tables.open_file(train_out_path)
# # contexts = table.get_node('/sentences')[:].astype(np.long)
# # label = table.get_node('/label')[:].astype(np.long)
# # index = table.get_node('/indices')[:]
# table.close()

#  
with tables.open_file(train_out_path, 'r') as f:
    contexts_m= f.get_node('/sentences')[:].astype(np.long)
    label_m = f.get_node('/labels')[:]
    index_m = f.get_node('/indices')[:]

# %%
contexts_m

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



