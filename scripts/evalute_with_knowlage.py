# %%
import os, sys
import inspect
sys.path.append("..")
src_file_path = inspect.getfile(lambda: None)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(src_file_path))))
import pandas as pd

import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from inference import DialogBert_Infernce
IMSR=0

# %%

t=DialogBert_Infernce(lexicon=1, with_feature=1,IMSR=0)

model=t.return_model(model_version="SSK_4.5_gender",reload_from="BestModel--1600",model_size="base")
# model=t.return_model(model_version="SSK_4.5_longer",reload_from="BestModel--1200",model_size="base")

# t=DialogBert_Infernce(lexicon=1, with_feature=1,IMSR=1 ,number_labels=2)
# model=t.return_model(model_version="IMSR_only_gsr_oversample_1_undersample_1",reload_from="BestModel--600",model_size="base")


# model=t.return_model(model_version="no_ssk",reload_from="BestModel--1000",model_size="base")


# %%
from data_loader import DialogTransformerDataset
import pickle
class DialogTransformerDatasetClassification_parts_withKnowalge(DialogTransformerDataset):
    """
    A base class for Transformer dataset
    """
    def __init__(self, file_path, tokenizer, 
                 min_num_utts=1, max_num_utts=7,percentage_utts=1.0, max_utt_len=30, 
                 block_size=256, utt_masklm=False, utt_sop=False, 
                 context_shuf=False, context_masklm=False,turns=False):


        # 1. Initialize file path or list of file names.
        """read training sentences(list of int array) from a hdf5 file"""
        self.tokenizer = tokenizer
        self.min_num_utts = min_num_utts #if not context_shuf and not context_masklm else 3
        self.max_num_utts = max_num_utts
        self.max_utt_len =max_utt_len
        self.turns=turns
        self.block_size = block_size # segment size to train BERT. when set -1 by default, use indivicual sentences(responses) as BERT inputs.
                            # Otherwise, clip a block from the context.
        self.percentage_utts=percentage_utts
        self.utt_masklm = utt_masklm
        self.utt_sop =utt_sop
        self.context_shuf =context_shuf
        self.context_masklm =context_masklm
        self.cls_utt = [tokenizer.cls_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id]
        self.sep_utt = [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.sep_token_id]

        print("loading data...")
        #read pickle 
        temp=pickle.load(open(file_path, "rb"))
        self.labels=temp[0]
        self.contexts=temp[1]

        self.data_len = len(self.labels)
        print("{} entries".format(self.data_len))

    def __getitem__(self, offset):
            contexts = self.contexts[offset]
            label= self.labels[offset]
            GSR_label, VED_label = label
            ## split context array into utterances        
            context = []
            turns=[]
            features=[] 
            for i, uter in enumerate(contexts):
                
                floor= uter[0]
                feature= uter[-1]
                uter = uter[:-1] 
                features.append(np.array(feature))
                tmp_utt = uter[1:] 

                utt_len = min(len(tmp_utt), self.max_utt_len) # floor is not counted in the utt length
                utt = tmp_utt[:utt_len] 
                context.append(utt)
                turns.append(floor)
            features=np.vstack(features) 

            # context = context[-num_utts:]
            
            context_len=  min(int(self.percentage_utts*len(context)), self.max_num_utts-2)
            context = [self.cls_utt] + context[:context_len] + [self.sep_utt]
            context_len+=2
            context_attn_mask = [1]*context_len
            context_mlm_target = [[-100]*len(utt) for utt in context]
            context_position_perm_id = -100
            context_position_ids = list(range(context_len))   #               
                

            context_utts_attn_mask = [[1]*len(utt) for utt in context]
            
            context = self.list2array(context, self.max_utt_len, self.max_num_utts, pad_idx=self.tokenizer.pad_token_id) 
            context_utts_attn_mask = self.list2array(context_utts_attn_mask, self.max_utt_len, self.max_num_utts)
            context_attn_mask = self.list2array(context_attn_mask, self.max_num_utts)
            context_mlm_target = self.list2array(context_mlm_target, self.max_utt_len, self.max_num_utts, pad_idx=-100)
            context_position_ids = self.list2array(context_position_ids, self.max_num_utts)
            turns=turns[:context_len]        

            turns= np.array(turns+ [self.tokenizer.pad_token_id]* (self.max_num_utts-len(turns)))
            features=features[:context_len]        
            features= features.sum(axis=0)
            
            # response = self.list2array(response, self.max_utt_len, pad_idx=self.tokenizer.pad_token_id) # for decoder training
            return context, context_utts_attn_mask, context_attn_mask, \
                context_mlm_target, context_position_perm_id, context_position_ids, GSR_label, VED_label,turns,features  
        

# %%
from data_loader import DialogTransformerDataset, HBertMseEuopDataset, DialogTransformerDatasetClassification
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler



# %%
from collections import defaultdict
score= defaultdict(dict)

# %%
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import fbeta_score
#adddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd model evel
percentage_lst=[0.2,0.4,0.6,0.8,1]
# percentage_lst=[0.8,1]
# percentage_lst=[1]
model=model.eval()
for percentage in percentage_lst:
    print("percentage:", percentage)
    # test_set = DialogTransformerDatasetClassification_parts("data/Sahar_labeld/test.pkl", model.tokenizer, model.tokenizer ,max_num_utts=40,percentage_utts=percentage)
    # test_set = DialogTransformerDatasetClassification_parts_withKnowalge("../data/Amir/test.pkl", model.tokenizer, model.tokenizer ,max_num_utts=140,percentage_utts=percentage)
    # test_set = DialogTransformerDatasetClassification_parts_withKnowalge("../data/IMSR_from_gsr_oversample_1_undersample_1/test.pkl", model.tokenizer, model.tokenizer ,max_num_utts=70,percentage_utts=percentage)
    # test_set = DialogTransformerDatasetClassification_parts_withKnowalge("../data/Sahar_2/test.pkl", model.tokenizer, model.tokenizer ,max_num_utts=140,max_utt_len=50,percentage_utts=percentage)
    test_set = DialogTransformerDatasetClassification_parts_withKnowalge("../data/Sahar_2_with_gender/test.pkl", model.tokenizer, model.tokenizer ,max_num_utts=140,max_utt_len=50,percentage_utts=percentage)

    eval_batch_size = 32
    sampler = SequentialSampler(test_set)
    dataloader = DataLoader(test_set, sampler=sampler, batch_size=eval_batch_size)       

    device = next(model.parameters()).device
    tokenizer = model.tokenizer

    # accuracy,F1,precision = [], [], []
    valid_losses = []
    preds=[]
    labels=[]
    probs=[]
    dlg_id = 0
    text=[]
    for batch in tqdm(dataloader): 
        
        batch_gpu = [t.to(device) for t in batch]
        if IMSR:
            labels.append(batch_gpu[-3].cpu().tolist())
        else:
            labels.append(batch_gpu[-4].cpu().tolist())

        # print(batch_gpu[:-2])
        # batch_gpu=batch_gpu[:-2]
        with torch.no_grad():
            loss, pred, prob  = model.validate_classification(*batch_gpu)

            probs+=prob[:, 1].cpu().tolist()
            # pred= torch.argmax(pred, dim=1)
            text.append(batch_gpu[0])
            # print(pred)
            # pred=np.argmax(pred)
            # labels+=batch_gpu[-2].cpu().tolist() #GSR LAbels
            preds.append(pred.cpu().tolist())

        # break
        labels
    t=[]
    for lst in labels:
        for label in lst:
            t.append(int(label))
    labels=t
    t=[]
    for lst in preds:
        for pred in lst:
            t.append(int(pred))
    preds=t

    # t=[]
    # for lst in probs:
    #     for pred in lst:
    #         t.append(int(pred))
    roc_auc= roc_auc_score(labels, probs, average=None)
    # roc_auc= 0.4

    score[percentage]["roc_auc"]=roc_auc
    score[percentage]["F1"]=f1_score(labels, preds, average=None)
    score[percentage]["precision"]=precision_score(labels, preds, average=None)
    score[percentage]["recall"]=recall_score(labels, preds, average=None)
    score[percentage]["accuracy"]=accuracy_score(labels, preds)
    score[percentage]["F2"]=fbeta_score(labels, preds, average=None, beta=2)
    print(classification_report(labels, preds))



# %%

for key in score:
    score[key]["F1"]=score[key]["F1"][1]
    score[key]["F2"]=score[key]["F2"][1]
    score[key]["recall"]=score[key]["recall"][1]
    score[key]["precision"]=score[key]["precision"][1]
results= pd.DataFrame(score).transpose()
pd.set_option('precision', 4)

results

# %%
from transformers import BertTokenizerFast

alephbert_tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')
temp=[]
for convs in text:
    for conv in convs:
        temp.append(list(map(lambda p: alephbert_tokenizer.decode(p, skip_special_tokens=True), conv)))

convs=temp 
del temp
# list(map(lambda p: alephbert_tokenizer.decode(p, skip_special_tokens=True), convs[0]))
import pandas as pd
conv_df=pd.DataFrame()
conv_df["text"]=convs
conv_df["preds"]=preds
conv_df["labels"]=labels
conv_df

# %%
conv_df[]

# %%
conv_df["preds"].value_counts()

# %%
FN= conv_df[(conv_df["preds"]==0)& (conv_df["labels"]==1)]
len(FN)

# %%
# test= [0.4 for i in range(len(labels))]
# roc_auc_score(labels, test, average=None)


# %%
#calcalute and plot confusion matrix with labels and predictions
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

cf_matrix = confusion_matrix(conv_df["labels"], conv_df["preds"],normalize='true')

ax = sns.heatmap(cf_matrix, annot=True, 
            fmt='.2%', cmap='Blues')

ax.set_title('SRF-Lexicon\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

plt.show()
def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)


TP, FP, TN, FN = perf_measure(conv_df["labels"],conv_df["preds"])
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)

# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)

Omission_Rate = FN / (FN + TN)

print("FNR: ",FNR)
print("Omission_Rate: ",Omission_Rate)



# %%


# %%
# results.reset_index(inplace=True)

# %%


# %%
results= results[["roc_auc","F2"]]

# %%
#plot line using seaborn
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()
#plot sns line and name the x and y axis and legend
# sns.lineplot(x="percentage", y="roc_auc", data=results)
p=sns.lineplot(data=results,markers=True, dashes=False)
p.set(xlabel='percentage', ylabel='score')

# %%



