import os
from pyexpat import features
import random
from copy import deepcopy
import numpy as np
import tables
import json
import itertools
from tqdm import tqdm
import torch
import torch.utils.data as data
import logging
import pickle
logger = logging.getLogger(__name__)



class DialogTransformerDataset(data.Dataset):
    """
    A base class for Transformer dataset
    """
    def __init__(self, file_path, tokenizer, 
                 min_num_utts=1, max_num_utts=7, max_utt_len=30, 
                 block_size=256, utt_masklm=False, utt_sop=False, 
                 context_shuf=False, context_masklm=False,with_feature=7):
        # 1. Initialize file path or list of file names.
        """read training sentences(list of int array) from a hdf5 file"""
        self.tokenizer = tokenizer
        self.min_num_utts = min_num_utts #if not context_shuf and not context_masklm else 3
        self.max_num_utts = max_num_utts
        self.max_utt_len =max_utt_len
        self.block_size = block_size # segment size to train BERT. when set -1 by default, use indivicual sentences(responses) as BERT inputs.
                            # Otherwise, clip a block from the context.
        self.with_feature = with_feature>0
        self.feature_number= with_feature
        self.utt_masklm = utt_masklm
        self.utt_sop =utt_sop
        self.context_shuf =context_shuf
        self.context_masklm =context_masklm
        
        self.rand_utt = [tokenizer.mask_token_id]*(max_utt_len-1) + [tokenizer.sep_token_id] # update during loading
        
        # a cache to store context and response that are longer than min_num_utts
        self.cache = [[tokenizer.mask_token_id]*max_utt_len]*max_num_utts, [tokenizer.mask_token_id]*max_utt_len
        
        # self.sbaperm_list = [list(itertools.permutations(range(L))) for L in range(1, max_num_utts+1)]
        self.perm_list= None
        print("loading data...")
        table = tables.open_file(file_path)
        self.contexts = table.get_node('/sentences')[:].astype(np.long)
        #self.knowlege = table.get_node('/knowledge')[:].astype(np.long)
        # self.label = table.get_node('/label')[:].astype(np.long)
        self.index = table.get_node('/indices')[:]
        self.data_len = self.index.shape[0]
        print("{} entries".format(self.data_len))

    def __getitem__(self, offset):
        index = self.index[offset]
        # label= self.label[offset]
        
        pos_utt, ctx_len, res_len,  = index['pos_utt'], index['ctx_len'], index['res_len']
        # pos_knowl, knowl_len = index['pos_knowl'], index['knowl_len']
        
        ctx_len = min(ctx_len, self.block_size) if self.block_size>-1 else ctx_len# trunck too long context
        
        ctx_arr=self.contexts[pos_utt-ctx_len:pos_utt].tolist()
        res_arr=self.contexts[pos_utt:pos_utt+res_len].tolist()
        turns=[]
        # knowl_arr = self.knowledge[pos_knowl:pos_knowl+knowl_len].tolist()
        
        ## split context array into utterances        
        context = []
        tmp_utt = []
        features= []
                # features=[] 
        # for i, uter in enumerate(contexts):
            
        #     floor= uter[0]
        #     if self.with_feature:
        #         feature= uter[-1]
        #         uter = uter[:-1] 
        #         features.append(np.array(feature))
        #     tmp_utt = uter[1:] 

        #     utt_len = min(len(tmp_utt), self.max_utt_len) # floor is not counted in the utt length
        #     utt = tmp_utt[:utt_len] 
        #     context.append(utt)
        #     turns.append(floor)
     

        for i, tok in enumerate(ctx_arr):
            tmp_utt.append(ctx_arr[i])
            if tok == self.tokenizer.sep_token_id:
                floor = tmp_utt[0]
                if self.with_feature:
                    feature= tmp_utt[-self.feature_number-1:-1]
                    features.append(np.array(feature)*-1)
                # tmp_utt = tmp_utt[1:] 
                tmp_utt = tmp_utt[:-self.feature_number-1]+[tmp_utt[-1]]

                tmp_utt = tmp_utt[1:] 
                utt_len = min(len(tmp_utt), self.max_utt_len) # floor is not counted in the utt length
                utt = tmp_utt[:utt_len]            
                context.append(utt)  # append utt to context          
                tmp_utt=[]  # reset tmp utt
                turns.append(floor)
                  
        response = res_arr[1:] # ignore cls token at the begining 
        response= response[:-self.feature_number-1]+[response[-1]]
        response_turn= res_arr[0]
        res_len = min(len(response),self.max_utt_len)
        response = response[:res_len-1] + [self.tokenizer.sep_token_id] 
        
        # knowledge = knowl_arr[:]              
        # knowl_len = min(len(knowledge),self.max_utt_len)
        # knowledge = knowledge[:knowl_len-1] + [self.tokenizer.sep_token_id] 
        

        
        num_utts = min(len(context), self.max_num_utts)
        context = context[-num_utts:]
        turns=turns[-num_utts:]          
        features=features[-num_utts:] 
#turn this on 
        # if self.with_feature:
        #     features=np.vstack(features)
        #     features=features[-num_utts:]
        #     features= features.sum(axis=0)
        # else:
        #     features=0 

        return context, response, response_turn,turns,features  #, knowlege
    

        
    def list2array(self, L, d1_len, d2_len=0, d3_len=0, dtype=np.long, pad_idx=0):
        '''  convert a list to an array or matrix  '''            
        def list_dim(a):
            if type(a)!=list: return 0
            elif len(a)==0: return 1
            else: return list_dim(a[0])+1
        
        if type(L) is not list:
            print("requires a (nested) list as input")
            return None
        
        if list_dim(L)==0: return L
        elif list_dim(L) == 1:
            arr = np.zeros(d1_len, dtype=dtype)+pad_idx
            for i, v in enumerate(L): arr[i] = v
            return arr
        elif list_dim(L) == 2:
            arr = np.zeros((d2_len, d1_len), dtype=dtype)+pad_idx
            for i, row in enumerate(L):
                for j, v in enumerate(row):
                    arr[i][j] = v
            return arr
        elif list_dim(L) == 3:
            arr = np.zeros((d3_len, d2_len, d1_len), dtype=dtype)+pad_idx
            for k, group in enumerate(L):
                for i, row in enumerate(group):
                    for j, v in enumerate(row):
                        arr[k][i][j] = v
            return arr
        else:
            print('error: the list to be converted cannot have a dimenson exceeding 3')
    
    def mask_words(self, utt):
        output_label = []
        tokens = [tok for tok in utt]
        for i, token in enumerate(utt):
            prob = random.random()
            if prob < 0.15 and not token in [self.tokenizer.pad_token_id, self.tokenizer.sep_token_id, self.tokenizer.cls_token_id]:
                prob /= 0.15                
                if prob < 0.8: 
                    tokens[i] = self.tokenizer.mask_token_id   # 80% randomly change token to mask token                
                elif prob < 0.9: 
                    tokens[i] = random.randint(5, len(self.tokenizer)-5)# 10% randomly change token to random token            
                output_label.append(token)
            else:
                output_label.append(-100)
        return tokens, output_label             
    
    def swap_utt(self, utt):
        utt_sop_label = 0 if random.random()>0.6 or len(utt)<5 else 1
        tokens = [tok for tok in utt]
        utt_len = len(tokens)
        if utt_len == self.max_utt_len: # if utt has reached the maximum length, then remove the last token because we will add a new sep token
            tokens = tokens[:-2]+ [self.tokenizer.sep_token_id]
            utt_len-=1
        sep_pos = random.randrange(2, utt_len-1) # seperate position where tokens to the right are random or coherent contexts

        # new utt
        L_utt, R_utt = tokens[1:sep_pos]+[self.tokenizer.sep_token_id], tokens[sep_pos:]
        swaped_utt = L_utt + R_utt if utt_sop_label ==0 else R_utt + L_utt
        swaped_utt = [self.tokenizer.cls_token_id] + swaped_utt
        utt_attn_mask = [1]*len(swaped_utt)
        # segment_ids                                                 
        utt_segment_ids = [0]*(sep_pos+1)+[1]*(utt_len-sep_pos) if utt_sop_label == 0 else [0]*(utt_len-sep_pos+1)+[1]*(sep_pos)       
        
        return swaped_utt, utt_attn_mask, utt_segment_ids, utt_sop_label
                    
    def mask_context(self, context):


        def is_special_utt(utt):
            return len(utt)==3 and utt[1] in [self.tokenizer.mask_token_id, self.tokenizer.sep_token_id, self.tokenizer.cls_token_id]
        
        utts = [utt for utt in context]
        lm_label = [[-100]*len(utt) for utt in context] 
        context_len = len(context)
        assert context_len>1, 'a context to be masked should have at least 2 utterances'

        mlm_probs = [0.0, 0.1, 0.4, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0]
        if context_len>=6:
            mlm_prob=1.0
        else:
            mlm_prob = mlm_probs[context_len-1]
        
        prob = random.random()
        if prob < mlm_prob:
            i = random.randrange(context_len)
            while is_special_utt(utts[i]): 
                i = random.randrange(context_len)
            utt = utts[i]
            prob = prob/mlm_prob
            if prob < 0.8: # 80% randomly change utt to mask utt
                utts[i] = [self.tokenizer.cls_token_id, self.tokenizer.mask_token_id, self.tokenizer.sep_token_id] 
            elif prob < 0.9: # 10% randomly change utt to a random utt  
                utts[i] = deepcopy(self.rand_utt)
            lm_label[i]= deepcopy(utt)
            #assert len(utts[i]) == len(lm_label[i]), "the size of the lm label is different to that of the masked utterance"
            self.rand_utt = deepcopy(utt) # update random utt
        return utts, lm_label
    
    # def shuf_ctx2(self,context,turns,features):
    #     num_utts = len(context)
    #     if num_utts==1: 
    #         return context, 0, [0], turns,features
    #     x = list(enumerate(context))
    #     random.shuffle(x)
    #     indices, shuffled = zip(*x)
    #     shuf_turns = [turns[i] for i in indices]
    #     shuf_features = [features[i] for i in indices]

    #     return list(shuffled), 94, list(indices),shuf_turns,shuf_features

    def shuf_ctx2(self,context,turns):
        num_utts = len(context)
        if num_utts==1: 
            return context, 0, [0], turns
        x = list(enumerate(context))
        random.shuffle(x)
        indices, shuffled = zip(*x)
        shuf_turns = [turns[i] for i in indices]

        return list(shuffled), 94, list(indices),shuf_turns

    def shuf_ctx(self, context,turns):    
        if self.perm_list is None:
            self.perm_list= [list(itertools.permutations(range(L))) for L in range(1, self.max_num_utts+1)]
        perm_label = 0
        num_utts = len(context)
        if num_utts==1: 
            return context, perm_label, [0]
        for i in range(num_utts-1): perm_label += len(self.perm_list[i])
        perm_id = int(random.random()*len(self.perm_list[num_utts-1]))
        perm_label += perm_id
        ctx_position_ids = self.perm_list[num_utts-1][perm_id]
        # new context
        shuf_context = [context[i] for i in ctx_position_ids]
        shuf_turns = [turns[i] for i in ctx_position_ids]
        return shuf_context, perm_label, ctx_position_ids,shuf_turns

    def __len__(self):
        return self.data_len    
    

    

class HBertMseEuopDataset(DialogTransformerDataset):
    """
    A hierarchical Bert data loader where the context is masked with ground truth utterances and to be trained with MSE matching.
    The context is shuffled for a novel energy-based order prediction approach (EUOP)
    """
    def __init__(self, file_path, tokenizer,
                 min_num_utts=1, max_num_utts=9, max_utt_len=30, 
                 block_size=-1, utt_masklm=False, utt_sop=False, 
                 context_shuf=False, context_masklm=False,with_feature=False):
        self.max_num_utts= max_num_utts
        super(HBertMseEuopDataset, self).__init__(
            file_path, tokenizer, min_num_utts, max_num_utts, max_utt_len, block_size, utt_masklm, utt_sop, context_shuf, context_masklm,with_feature)
        
        self.cls_utt = [tokenizer.cls_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id]
        self.sep_utt = [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.sep_token_id]


    def __getitem__(self, offset):
        context, response,_, turns,features  = super().__getitem__(offset)
        context_len=  min(len(context), self.max_num_utts-2)
        context = [self.cls_utt] + context[-context_len:] + [self.sep_utt]
        context_len+=2
        context_attn_mask = [1]*context_len
        context_mlm_target = [[-100]*len(utt) for utt in context]
        context_position_perm_id = -100
        context_position_ids = list(range(context_len))   #               
        turn_position_ids= list(range(context_len))
        if self.context_shuf and random.random()<0.4 and len(context)>2:
            if self.max_num_utts!= 7:
                context_, context_position_perm_id, context_position_ids_,turns = self.shuf_ctx2(context[1:-1],turns)
            else: 
                context_, context_position_perm_id,context_position_ids_,turns = self.shuf_ctx(context[1:-1],turns)

            context = [self.cls_utt] + context_ + [self.sep_utt]
            context_position_ids = [0] + [p+1 for p in context_position_ids_] + [context_len-1]
            # turn_position_ids = [0] + [p+1 for p in turn_position_ids_] + [context_len-1]
            context_mlm_target = [[-100]*len(utt) for utt in context]
            
        if self.context_masklm and context_position_perm_id<2 and len(context)>4:
            context, context_mlm_target = self.mask_context(context)
        
        context_utts_attn_mask = [[1]*len(utt) for utt in context]
        
        context = self.list2array(context, self.max_utt_len, self.max_num_utts, pad_idx=self.tokenizer.pad_token_id) 
        turns= np.array(turns+ [self.tokenizer.pad_token_id]* (self.max_num_utts-len(turns)))

        # features= np.array(turns+ [self.tokenizer.pad_token_id]* (self.max_num_utts-len(turns)))
        context_utts_attn_mask = self.list2array(context_utts_attn_mask, self.max_utt_len, self.max_num_utts)
        context_attn_mask = self.list2array(context_attn_mask, self.max_num_utts)
        context_mlm_target = self.list2array(context_mlm_target, self.max_utt_len, self.max_num_utts, pad_idx=-100)
        context_position_ids = self.list2array(context_position_ids, self.max_num_utts)
        # turn_position_ids = self.list2array(turn_position_ids, self.max_num_utts)

        response = self.list2array(response, self.max_utt_len, pad_idx=self.tokenizer.pad_token_id) # for decoder training


        return context, context_utts_attn_mask, context_attn_mask, \
              context_mlm_target, context_position_perm_id, context_position_ids, response,turns,features  
    
class DialogTransformerDatasetClassification(DialogTransformerDataset):
    """
    A base class for Transformer dataset
    """
    def __init__(self, file_path, tokenizer, 
                 min_num_utts=1, max_num_utts=7, max_utt_len=30, 
                 block_size=256, utt_masklm=False, utt_sop=False, 
                 context_shuf=False, context_masklm=False,turns=False,with_feature=False,number_labels=2):


        # 1. Initialize file path or list of file names.
        """read training sentences(list of int array) from a hdf5 file"""
        self.tokenizer = tokenizer
        self.min_num_utts = min_num_utts #if not context_shuf and not context_masklm else 3
        self.max_num_utts = max_num_utts
        self.max_utt_len =max_utt_len
        self.with_turns=turns
        self.block_size = block_size # segment size to train BERT. when set -1 by default, use indivicual sentences(responses) as BERT inputs.
                            # Otherwise, clip a block from the context.
        self.number_labels=number_labels
        self.utt_masklm = utt_masklm
        self.utt_sop =utt_sop
        self.context_shuf =context_shuf
        self.context_masklmwith_feature=context_masklm
        self.with_feature=with_feature
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
        GSR_label, IMSR_label = label
        ## split context array into utterances        
        context = []
        turns=[]
        features=[] 
        for i, uter in enumerate(contexts):
            
            floor= uter[0]
            if self.with_feature:
                feature= uter[-1]
                uter = uter[:-1] 
                features.append(np.array(feature))
            else: 
                uter=uter[:-1]
            tmp_utt = uter[1:] 

            utt_len = min(len(tmp_utt), self.max_utt_len) # floor is not counted in the utt length
            utt = tmp_utt[:utt_len] 
            context.append(utt)
            turns.append(floor)
     
        num_utts = min(len(context), self.max_num_utts)
        # context = context[-num_utts:]
         
        context_len=  min(len(context), self.max_num_utts-2)
        # context = [self.cls_utt] + context[-context_len:] + [self.sep_utt]
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
        # response = self.list2array(response, self.max_utt_len, pad_idx=self.tokenizer.pad_token_id) # for decoder training
        turns=turns[-num_utts:]          
        turns= np.array(turns+ [self.tokenizer.pad_token_id]* (self.max_num_utts-len(turns)))
        # features=np.append(features, b, axis=0)
        if self.with_feature:
            features=np.vstack(features)
            features=features[-num_utts:]
            features= features.sum(axis=0)
        else:
            features=0    
        if self.number_labels==3: 
            #if GSR=0, IMSR=0, then label=0
            #if GSR=1, IMSR=0, then label=1
            #if GSR=1, IMSR=1, then label=2

            GSR_label = GSR_label+IMSR_label
            IMSR_label = GSR_label
        return context, context_utts_attn_mask, context_attn_mask, \
              context_mlm_target, context_position_perm_id, context_position_ids, GSR_label, IMSR_label,turns,features    
    

    def list2array(self, L, d1_len, d2_len=0, d3_len=0, dtype=np.long, pad_idx=0):
        '''  convert a list to an array or matrix  '''            
        def list_dim(a):
            if type(a)!=list: return 0
            elif len(a)==0: return 1
            else: return list_dim(a[0])+1
        
        if type(L) is not list:
            print("requires a (nested) list as input")
            return None
        
        if list_dim(L)==0: return L
        elif list_dim(L) == 1:
            arr = np.zeros(d1_len, dtype=dtype)+pad_idx
            for i, v in enumerate(L): arr[i] = v
            return arr
        elif list_dim(L) == 2:
            arr = np.zeros((d2_len, d1_len), dtype=dtype)+pad_idx
            for i, row in enumerate(L):
                for j, v in enumerate(row):
                    arr[i][j] = v
            return arr
        elif list_dim(L) == 3:
            arr = np.zeros((d3_len, d2_len, d1_len), dtype=dtype)+pad_idx
            for k, group in enumerate(L):
                for i, row in enumerate(group):
                    for j, v in enumerate(row):
                        arr[k][i][j] = v
            return arr
        else:
            print('error: the list to be converted cannot have a dimenson exceeding 3')
    

    def __len__(self):
        return self.data_len 



   
def load_dict(filename):
    return json.loads(open(filename, "r").readline())

def load_vecs(fin):         
    """read vectors (2D numpy array) from a hdf5 file"""
    h5f = tables.open_file(fin)
    h5vecs= h5f.root.vecs
    
    vecs=np.zeros(shape=h5vecs.shape,dtype=h5vecs.dtype)
    vecs[:]=h5vecs[:]
    h5f.close()
    return vecs

def save_vecs(vecs, fout):
    fvec = tables.open_file(fout, 'w')
    atom = tables.Atom.from_dtype(vecs.dtype)
    filters = tables.Filters(complib='blosc', complevel=5)
    ds = fvec.create_carray(fvec.root,'vecs', atom, vecs.shape,filters=filters)
    ds[:] = vecs
    print('done')
    fvec.close()
