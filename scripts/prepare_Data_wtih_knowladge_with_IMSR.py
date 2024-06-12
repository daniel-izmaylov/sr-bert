
import copy
import os, sys
import inspect
src_file_path = inspect.getfile(lambda: None)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(src_file_path))))

import pandas as pd 
import numpy as np
import pickle
from transformers import BertTokenizerFast
import random
import numpy as np
import json
import os
import re
from tqdm import tqdm
import pickle as pkl
import sys
sys.path.append("..")
from data_loader import load_dict, save_vecs



from sklearn.utils import resample
from multiprocesspandas import applyparallel
from tqdm import tqdm
tqdm.pandas()
from tqdm import tqdm

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
seed_everything(42)
    
def check_contains2(converasion, lex_dict):
    for k,v in lex_dict.items():
        to_return=[]
        # print(converasion)
        for text in converasion.text:
            flag=False
            to_return.append(int(flag))
        converasion[k]=to_return
    return converasion


# alephbert_tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')
# with open('/home/izmaylov/Thesis/Preprocess/data/September_Dataset/result_combine.pkl', 'rb') as f:
#     temp_df = pickle.load(f)


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
    
def make_imporant_dict(lex_df):
    def return_important(examples):
        lst=[]
        for example in examples:
            example=example.strip()
            if "*"  in example:
                example=example.replace("*","")
                lst+=[example]
        return lst
                

    lex_dict={}

    for col in lex_df.columns:
        examples= lex_df[col].dropna()
        try:
            gender= examples[0]
        except:
            print(examples)
            
        examples=examples[1:]
        key = col[:15].rstrip(".1")
        if key in lex_dict:
            lex_dict[key]+=return_important(examples)
        else:
            lex_dict[key]=return_important(examples)
        
    #remove empty string from the dict
    for key in lex_dict.keys():
        lex_dict[key]=list(filter(None, lex_dict[key]))
    return lex_dict

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


def make_data_set(df,from_gsr=True,oversample_factor=1, undersample_factor=1,over_sample_augment="IMSR"):
    def make_imporant_dict(lex_df,important=True):
        def return_important(examples):
            lst=[]
            for example in examples:
                example=example.strip()
                if "*"  in example:
                    example=example.replace("*","")
                    lst+=[example]
            return lst
                    

        lex_dict={}

        for col in lex_df.columns:
            examples= lex_df[col].dropna()
            try:
                gender= examples[0]
            except:
                print(examples)
                
            examples=examples[1:]
            key = col[:15].rstrip(".1")
            if key in lex_dict:
                if important:
                    lex_dict[key]+=return_important(examples)
                else:
                    lex_dict[key]+=list(examples)
            else:
                if important:
                    lex_dict[key]=return_important(examples)
                else:
                    lex_dict[key]=list(examples)
            
        #remove empty string from the dict
        for key in lex_dict.keys():
            lex_dict[key]=list(filter(None, lex_dict[key]))
        return lex_dict 
    
    def generate_dict(Type):
        if Type=="Both":
            imsr_dict=generate_dict("IMSR")
            gsr_dict=generate_dict("GSR")
            return {**imsr_dict, **gsr_dict}
        
        if Type=="IMSR":
            left_colums=["שיטה","הימלכדות","תכנית (שיטה+כוו","כאב נפשי 26","כוונה"]
            important=False
            lex_path="/home/izmaylov/Thesis/lexicon_sim/IMSR_4.6.csv"              
        else:
            left_colums= ['ניסיון אובדני ק', 'היסטוריה אובדני', 'פגיעה עצמית', 'בדידות- מעלה', 'בדידות- מוריד', 'היעדר תקווה- מע']
            important=True
            lex_path="/home/izmaylov/Thesis/lexicon_sim/gsr_lex_4.6.csv"

        lex_df=pd.read_csv(lex_path)
        lex_dict=make_imporant_dict(lex_df,important)
        lex_dict={k: lex_dict[k] for k in left_colums}

        return lex_dict
    
    
    lex_dict=generate_dict(over_sample_augment)

    #read json
    with open('/home/izmaylov/Thesis/SR_BERT/data/Data_2/id_dict.json', 'rb') as f:
        id_dict = json.load(f)
    id_dict.keys()


    #use the id_dict to split the data to train and test
    train_ids = id_dict['train']
    test_ids = id_dict['test']
    val_ids = id_dict['valid']
    train_df=df[df["Engagement ID"].isin(train_ids)]
    test_df=df[df["Engagement ID"].isin(test_ids)]
    valid_df=df[df["Engagement ID"].isin(val_ids)]


    print("train_data",len(train_df))
    print("Number of positive samples in train_data",len(train_df[train_df.IMSR==1]),(len(train_df[train_df.IMSR==1])/len(train_df)*100),"%")
    print("Number of positive samples in test_df",len(test_df[test_df.IMSR==1]),(len(test_df[test_df.IMSR==1])/len(test_df)*100),"%")
    print("Number of positive samples in valid_df",len(valid_df[valid_df.IMSR==1]),(len(valid_df[valid_df.IMSR==1])/len(valid_df)*100),"%")
    
    #remove empty keys from lex_dict
    lex_dict={k: v for k, v in lex_dict.items() if v}


    def augment_sentence(sentence, lex_dict):
        for key, features in lex_dict.items():
            for feature in features:
                found = re.search(r"\b{}\b".format(feature), sentence)
                if found:
                    # print("found",feature)
                    start, end = found.start(), found.end()
                    random_replacement = random.choice(lex_dict[key])
                    # print(sentence)
                    sentence = sentence[:start] + random_replacement + sentence[end:]
                    # print(sentence)
                    return sentence  # exit the loop after the first match is found
        return sentence

    def augment_Transcript(df,lex_dict):
        df1=df.copy(deep=True)
        df1["Transcript"]=df["Transcript"].copy(deep=True)
        df1["Transcript"]["text"]=df1["Transcript"].apply(lambda x: augment_sentence(x.text, lex_dict),axis=1)
        return df1

    if from_gsr:
        train_df=train_df[train_df.GSR==1]
        valid_df=valid_df[valid_df.GSR==1]
        test_data=test_df
        

    train_data=train_df.copy(deep=True)
    valid_data=valid_df.copy(deep=True)
    test_data=test_df.copy(deep=True)


    df_majority = train_df[train_df.IMSR==0]
    df_minority = train_df[train_df.IMSR==1]



    if oversample_factor>1:
        if over_sample_augment!=None:
            df_minority_upsampled=df_minority
            for i in range(oversample_factor-1):
                aug_data=df_minority.apply(lambda x: augment_Transcript(x,lex_dict),axis=1)
                # aug_data=df_minority.apply(lambda x: augment_Transcript(x),axis=1)
                df_minority_upsampled=pd.concat([df_minority_upsampled, aug_data])
        else:
            df_minority_upsampled = resample(df_minority,replace=True,n_samples=len(df_minority)*oversample_factor,random_state=42)
        df_minority = pd.concat([df_minority, df_minority_upsampled])
        
    if undersample_factor>1:
        df_majority = resample(df_majority,replace=False,n_samples=int(len(df_majority)/undersample_factor),random_state=42)
    train_data = pd.concat([df_majority, df_minority])



    #save to pickle
    # with open('../data/IMSR_newl', 'wb') as f:
    #     pickle.dump({"train_data":train_data,"valid_data":valid_data,"test_data":test_data}, f)


    print("train_data",len(train_data))
    print("Number of positive samples in train_data",len(train_data[train_data.IMSR==1]),(len(train_data[train_data.IMSR==1])/len(train_data)*100),"%")
    print("Number of negative samples in train_data",len(train_data[train_data.IMSR==0]),(len(train_data[train_data.IMSR==0])/len(train_data)*100),"%")

        
    def compact_dialog(conv):
        left_colums=['IPT2 תפיסת העצמ', 'פגיעה עצמית', 'חוסר תקווה- מעל', 'ניסיון אובדני ק',
       'מחשבות אובדניות']
        conv["texter1"]=conv["texter"].map({0:"A","False":"A","True":"B",1:"B",False:"A",True:"B"})
        # print(conv["texter"]
        # conv["features"]= [np.zeros((1, 1)) for i in range(len(conv.index))]
        # conv["features"]=conv[left_colums].astype(str).apply(','.join, axis=1)
        # conv["features"]=conv[left_colums].astype(str).apply(','.join, axis=1)
        conv["features"]=("0,0,0,0,0")
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
            res["IMSR"]=df.iloc[i]["IMSR"]
            res["GSR"]=df.iloc[i]["GSR"]
            # res["IMSR"]=df.iloc[i].IMSR
            dialogs.append(res)
        return dialogs


    train_data= get_Sahar_data_turns(train_data)
    test_data= get_Sahar_data_turns(test_data)
    valid_data= get_Sahar_data_turns(valid_data)



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
                idx_utt.append([ int(j) for j in feature.split(",")])

                dialog_message.append(idx_utt)
            # print(dialog_message)
            preprosses_dialogs.append(dialog_message)
            
        with open(output_path, 'wb') as f:
            pickle.dump((labels,preprosses_dialogs), f)
        # print(labels)
        return (labels,preprosses_dialogs)


    alephbert_tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')

    data_dir="/home/izmaylov/Thesis/SR_BERT/data/IMSR"

    if from_gsr:
        data_dir+="_from_gsr"

    if oversample_factor>1:
        data_dir+="_oversample_"+str(oversample_factor)
    if over_sample_augment:
        data_dir+="_over_sample_augment_"+str(over_sample_augment)
    if undersample_factor>1:
        data_dir+="_undersample_"+str(undersample_factor)


    Path(data_dir).mkdir(parents=True, exist_ok=True)
    print(data_dir + " created")

    #save index_dict to the folder 
    # with open(data_dir+"/index_dict.pkl", 'wb+') as f:
    #     pickle.dump(index_dict, f)
        

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
    
    
    

left_colums=['IPT2 תפיסת העצמ', 'פגיעה עצמית', 'חוסר תקווה- מעל', 'ניסיון אובדני ק',
       'מחשבות אובדניות',"GSR"]
lex_df=pd.read_csv("/home/izmaylov/Thesis/lexicon_sim/gsr_lex_4.6.csv")
lex_dict={}
for col in lex_df.columns:
    if col[:15] in left_colums:
        examples= lex_df[col].dropna()
        lex_dict[col[:15]]=[set(example.split(" ")) for example in examples]
        
with open('/home/izmaylov/Thesis/Preprocess/data/Jan22/labeld_with_IMSR.pkl', 'rb') as f:
    df = pickle.load(f)
df["GSR"]=df["GSR"].astype(float).astype(int)
df.loc[(df.IMSR==1) & (df.GSR==0), 'GSR'] = 1  
df["Transcript"]=df.Transcript.progress_apply(lambda x: check_contains2(x,lex_dict)) 

from_gsr=True
for over_sample_augment in ["Both"]:
# for over_sample_augment in ["Both"]:
    for oversample_factor in [5]:
        for undersample_factor in [1,2,5]:
            t=df.copy()
            print("oversample_factor:",oversample_factor,
                  "\nundersample_factor:",undersample_factor,
                  "\nover_sample_augment:",over_sample_augment)
            
            make_data_set(t,from_gsr=from_gsr,oversample_factor=oversample_factor,
                          over_sample_augment=over_sample_augment,undersample_factor=undersample_factor)


