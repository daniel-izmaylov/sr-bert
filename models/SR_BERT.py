
from pathlib import Path

from pyexpat import features
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss, NLLLoss, BCELoss
from transformers import WEIGHTS_NAME, BertForSequenceClassification, BertTokenizerFast
import json
import os
from os import path as path
from copy import deepcopy
import numpy as np
import random
import itertools
import sys
parentPath = os.path.abspath("..")
sys.path.insert(0, parentPath)# add parent folder to path so as to import common modules

from transformers import (top_k_top_p_filtering,BertConfig, BertForPreTraining, BertPreTrainedModel, BertTokenizer, BertModel, BertLMHeadModel)
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform
# from modules import Classification_head
from modules import BertForSequenceClassification_v2, BertForSequenceClassification_lex_features

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score

# from modules import MLP, MixtureDensityNetwork

class SSK_Task(nn.Module):
    """ A Self Sorting Network which evaluates the sorting energy of each element in a sequence, adopted from the self-attention.
    Args:
        dimensions (int): Dimensionality of the sequence
    Example:
         >>> sortnet = SelfSorting(256)
         >>> seq = torch.randn(6, 5, 256)
         >>> sorting_scores = sortnet(seq)
         >>> sorting_scores.size()
         torch.Size([6, 5])
    """

    def __init__(self, dimensions,n_features,deepSSK=False):
        super(SSK_Task, self).__init__()
        self.deepSSK=deepSSK
        if self.deepSSK:
            self.ff1 = nn.Linear(dimensions, 128 )
            self.relu1 = nn.ReLU()
            self.ff2 = nn.Linear( 128, n_features)
        else:
            self.ff1 = nn.Linear(dimensions, n_features)
    
    def forward(self, sequence,labels):
        if self.deepSSK:
            # batch_size, seq_len, dimensions = sequence.size()
            # sequence = sequence.reshape(batch_size * seq_len, dimensions)
            sequence = self.relu1(self.ff1(sequence))
            sequence = self.ff2(sequence)
        else:
            sequence = self.ff1(sequence)
        loss_ctx_mlm = MSELoss()(sequence, labels) 
        return loss_ctx_mlm

# class Feature_learning_danse(nn.Module):
#     """ A Self Sorting Network which evaluates the sorting energy of each element in a sequence, adopted from the self-attention.
#     Args:
#         dimensions (int): Dimensionality of the sequence
#     Example:
#          >>> sortnet = SelfSorting(256)
#          >>> seq = torch.randn(6, 5, 256)
#          >>> sorting_scores = sortnet(seq)
#          >>> sorting_scores.size()
#          torch.Size([6, 5])
#     """

#     def __init__(self, dimensions,n_features):
#         super(SSK_Task, self).__init__()
#         self.ff1 = nn.Linear(dimensions, 128 )
#         self.relu1 = nn.ReLU()
#         self.ff2 = nn.Linear( 128, n_features)

    
#     def forward(self, sequence,labels):
#         # TODO check what about normalizion 
#         """
#         Args:
#             sequence (:[batch size, seq length, dimensions]): input encodings of a sequence
#             attention_mask ([batch_size, seq _length]): attention mask of the input sequence. 1=attened 0=not attend
#         Returns:
#             sorting scores ([batch size, sequence length]): Tensor containing sorting scores.
#         """
#         # batch_size, seq_len, dimensions = sequence.size()

#         # sequence = sequence.reshape(batch_size * seq_len, dimensions)
#         sequence = self.ff1(sequence)
#         sequence = self.relu1(sequence)
#         sequence = self.ff2(sequence)
#         loss_ctx_mlm = MSELoss()(sequence, labels) 
#         return loss_ctx_mlm

class SelfSorting(nn.Module):
    """ A Self Sorting Network which evaluates the sorting energy of each element in a sequence, adopted from the self-attention.
    Args:
        dimensions (int): Dimensionality of the sequence
    Example:
         >>> sortnet = SelfSorting(256)
         >>> seq = torch.randn(6, 5, 256)
         >>> sorting_scores = sortnet(seq)
         >>> sorting_scores.size()
         torch.Size([6, 5])
    """

    def __init__(self, dimensions):
        super(SelfSorting, self).__init__()
        self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

    def forward(self, sequence, attention_mask):
        """
        Args:
            sequence (:[batch size, seq length, dimensions]): input encodings of a sequence
            attention_mask ([batch_size, seq _length]): attention mask of the input sequence. 1=attened 0=not attend
        Returns:
            sorting scores ([batch size, sequence length]): Tensor containing sorting scores.
        """
        batch_size, seq_len, dimensions = sequence.size()

        sequence = sequence.reshape(batch_size * seq_len, dimensions)
        sequence = self.linear_in(sequence)
        sequence = sequence.reshape(batch_size, seq_len, dimensions)

        # (batch_size, seq_len, dimensions) * (batch_size, seq_len, dimensions) -> (batch_size, seq_len, seq_len)
        sorting_scores = torch.bmm(sequence, sequence.transpose(1, 2).contiguous())
        if attention_mask is not None:
            score_mask = attention_mask[:,None,:].expand(-1,seq_len,-1)# [batch_size x seq_len x seq_len]
            #score_mask = torch.matmul(attention_mask[:,:,None].float(), attention_mask[:,None,:].float())
            sorting_scores = (sorting_scores*score_mask).sum(2)/score_mask.sum(2)
        else:
            sorting_scores = sorting_scores.mean(2) # use the averate score for each item
        sorting_scores = sorting_scores*attention_mask
        return sorting_scores # [batch_size x seq_len]
    
def listNet(scores_pred, perm_true, pad_mask, eps=1e-3):
    """
    ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
    Adopted from https://github.com/allegro/allRank/blob/master/allrank/models/losses/listNet.py
    :param scores_pred: predicted sorting scores from the model, shape [batch_size, seq_length]
    :param perm_true: ground truth order, shape [batch_size, seq_length]
    :param eps: epsilon value, used for numerical stability
    :param pad_mask: an indicator of the perm_true containing a padded item
    :return: loss value, a torch.Tensor
    """
    max_pos = perm_true.max(dim=1,keepdim=True)[0]
    scores_true = (max_pos-perm_true).float()/max_pos
    
    scores_pred[pad_mask] = float('-inf')
    scores_true[pad_mask] = float('-inf')
    
    top1prob_pred = torch.softmax(scores_pred, 1) + eps
    top1prob_true = torch.softmax(scores_true, 1)
    loss_type = 'KL'
    if loss_type == 'CE':
        loss = - top1prob_true * torch.log(top1prob_pred)
    elif loss_type == 'KL':
        loss = nn.KLDivLoss(reduction='none')(torch.log(top1prob_pred), top1prob_true)
    elif loss_type == 'JS': # JS Divergence between two probabilities
        top1prob_mean = 0.5*(top1prob_pred+top1prob_true)
        kld = nn.KLDivLoss(reduction='none')
        loss = 0.5*kld(torch.log(top1prob_pred),top1prob_mean)+0.5*kld(torch.log(top1prob_true+eps), top1prob_mean)  
    loss[pad_mask] = 0.0   #[batch_size x seq_len]                      
    loss = loss.sum(1).mean()
    return loss 

def listMLE(scores_pred, perm_true, pad_mask, eps=1e-10):
    """
    ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
    Adopted from https://github.com/allegro/allRank/blob/master/allrank/models/losses/listMLE.py
    :param scores_pred: predicted sorting scores from the model, shape [batch_size, seq_length]
    :param perm_true: ground truth order, shape [batch_size, seq_length]
    :param eps: epsilon value, used for numerical stability
    :param pad_mask: an indicator of the perm_true containing a padded item
    :return: loss value, a torch.Tensor
    """
    preds_by_true = torch.gather(scores_pred, dim=1, index=perm_true)
    preds_by_true[pad_mask] = float("-inf")
    preds_by_true = preds_by_true - preds_by_true.max(dim=1, keepdim=True)[0] # substruct the maximum value
    cumsums = torch.cumsum(preds_by_true.exp().flip(dims=[1]), dim=1).flip(dims=[1])

    loss = torch.log(cumsums + eps) - preds_by_true
    loss[pad_mask] = 0.0
    loss = loss.sum(1).mean()
    
    return loss 

    
class SR_BERT(nn.Module):
    '''Hierarchical BERT for dialog v5 with two features:
    - Masked context utterances prediction with direct MSE matching of their vectors
    - Energy-based Utterance order prediction: A novel approach to shuffle the context and predict the original order with distributed order prediction'''
    

    
    def __init__(self):
        super(SR_BERT, self).__init__()  

    def __init__(self, args, base_model_name='bert-base-uncased'):
        super(SR_BERT, self).__init__()  

        if args.language == 'chinese': base_model_name = 'bert-base-chinese'
        elif args.language == 'hebrew': base_model_name = 'onlplab/alephbert-base'

        self.tokenizer = BertTokenizerFast.from_pretrained(base_model_name, cache_dir='./cache/')
        if args.model_size == 'tiny':
            self.encoder_config = BertConfig(vocab_size=30522, hidden_size=256, num_hidden_layers=6,
                num_attention_heads=2, intermediate_size=1024)
            self.utt_encoder = BertForPreTraining(self.encoder_config)
        elif args.model_size == 'small':
            self.encoder_config = BertConfig(vocab_size=30522, hidden_size=512, num_hidden_layers=8,
                num_attention_heads=4, intermediate_size=2048)
            self.utt_encoder = BertForPreTraining(self.encoder_config)


        else:
            self.encoder_config = BertConfig.from_pretrained(base_model_name, cache_dir='./cache/')
            self.utt_encoder = BertForPreTraining.from_pretrained(base_model_name, config=self.encoder_config, cache_dir='./cache/')


        if  args.model_size == 'tiny_heb':
            self.context_config = BertConfig(vocab_size=52000, hidden_size=768, num_hidden_layers=6,
                num_attention_heads=8, intermediate_size=1024)
        else:
            self.context_config= self.encoder_config
        if args.with_turns:
            self.context_config.hidden_size+=12 #for the turn indicator  
        # self.context_config.output_hidden_states=768 #for the turn indicator  
        self.context_encoder = BertModel(self.context_config) # context encoder: encode context to vector
        
        self.mlm_mode = 'mse' # 'mdn', 'mse'
        if self.mlm_mode == 'mdn':
            self.context_mlm_trans = MixtureDensityNetwork(self.encoder_config.hidden_size, self.encoder_config.hidden_size, 3)
        else:
            self.context_mlm_trans = BertPredictionHeadTransform(self.encoder_config) # transform context hidden states back to utterance encodings
        
        self.train_base=False
        if args.train_base==1:
            self.train_base=True
        if args.lexicon==1:     
            self.lexicon=True
        else:
            self.lexicon=False
        self.number_labels= args.number_labels
        self.with_feature= args.with_feature==1
        self.IMSR= args.IMSR==1
        self.dropout = nn.Dropout(self.encoder_config.hidden_dropout_prob)
        self.context_order_trans = SelfSorting(self.encoder_config.hidden_size)
        # for param in self.utt_encoder.parameters():
        #     param.requires_grad = False
        self.decoder_config = deepcopy(self.encoder_config)
        self.decoder_config.is_decoder=True
        self.decoder_config.add_cross_attention=True
        self.decoder = BertLMHeadModel(self.decoder_config)
        self.with_turns= args.with_turns
        if args.fine_tune==0:
            self.SSK_task= SSK_Task(self.encoder_config.hidden_size,args.with_feature,args.deepSSK)
            self.feature_loss_alpha= args.feature_loss_alpha
   
        
    def init_weights(self, m):# Initialize Linear Weight for GAN
        if isinstance(m, nn.Linear):        
            m.weight.data.uniform_(-0.08, 0.08)#nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0.)  
            
    #@classmethod       
    def from_pretrained(self, model_dir):
        self.encoder_config = BertConfig.from_pretrained(model_dir)
        self.tokenizer = BertTokenizer.from_pretrained(path.join(model_dir, 'tokenizer'))
        self.utt_encoder = BertForPreTraining.from_pretrained(path.join(model_dir, 'utt_encoder'))
        self.context_encoder = BertForSequenceClassification.from_pretrained(path.join(model_dir, 'context_encoder'))
        # self.context_encoder = BertForPreTraining.from_pretrained(path.join(model_dir, 'context_encoder'))
        self.context_mlm_trans = BertPredictionHeadTransform(self.encoder_config)
        self.context_mlm_trans.load_state_dict(torch.load(path.join(model_dir, 'context_mlm_trans.pkl')))
        self.context_order_trans = SelfSorting(self.encoder_config.hidden_size)
        self.context_order_trans.load_state_dict(torch.load(path.join(model_dir, 'context_order_trans.pkl')))
        self.decoder_config = BertConfig.from_pretrained(model_dir)
        self.decoder = BertLMHeadModel.from_pretrained(path.join(model_dir, 'decoder'))

        #@classmethod       
    def init_finetune(self, model_dir):
        # self.encoder_config = BertConfig.from_pretrained(model_dir)
        # self.encoder_config = BertModel.from_pretrained('onlplab/alephbert-base')
        # for param in self.encoder_config.parameters():
        #     param.requires_grad = False
        del self.decoder
        del self.context_mlm_trans
        del self.context_order_trans
        del self.utt_encoder

        self.tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base', cache_dir='./cache/')
        utt_path= path.join(model_dir, 'utt_encoder')
        #read json file
        with open(utt_path+'/config.json') as json_file:
            data = json.load(json_file)
            data["hidden_size"]=768
        #save to json file
        with open(utt_path+'/config.json', 'w') as f:
            json.dump(data, f)
        
        self.utt_encoder = BertForPreTraining.from_pretrained(path.join(model_dir, 'utt_encoder'))
        for param in self.utt_encoder.parameters():
            param.requires_grad = False
        if self.lexicon:
            self.context_encoder = BertForSequenceClassification_lex_features.from_pretrained(path.join(model_dir, 'context_encoder'),
                                                                num_labels = self.number_labels,   
                                                                output_attentions = False, # Whether the model returns attentions weights.
                                                                output_hidden_states = False)

        else:
            self.context_encoder = BertForSequenceClassification_v2.from_pretrained(path.join(model_dir, 'context_encoder'),
                                                                                num_labels = 2,   
                                                                                output_attentions = False, # Whether the model returns attentions weights.
                                                                                output_hidden_states = False) 

        # config= BertConfig.from_pretrained(path.join(model_dir, 'context_encoder'),
        #                                                                     num_labels = 2,   
        #                                                                     output_attentions = False, # Whether the model returns attentions weights.
        #                                                                     output_hidden_states = False)
        # # config.lex_embed_path=path.join(model_dir, 'data/data.json')                                                            
        # # config.lex_embed_dim=28
        # self.context_encoder = BertForSequenceClassification_lex_features.from_pretrained(config)     

        # self.context_encoder = BertModel.from_pretrained(path.join(model_dir, 'context_encoder'))
        self.context_encoder.train()


            
        # self.classifier= Classification_head(num_labels=2,dim=768)
        # self.classifier.train()
        # # print("df")


    def save_finetuning(self, output_dir):   
        def save_module(model, save_path):       
            torch.save(model_to_save.state_dict(), save_path)
        def make_list_dirs(dir_list):
            for dir_ in dir_list: os.makedirs(dir_, exist_ok=True)
        make_list_dirs([path.join(output_dir, name) for name in ['tokenizer', 'utt_encoder', 'context_encoder', 'decoder']])        
        model_to_save = self.module if hasattr(self, 'module') else self
        model_to_save.encoder_config.save_pretrained(output_dir) # Save configuration file
        # model_to_save.tokenizer.save_pretrained(path.join(output_dir,'tokenizer'))
        # model_to_save.utt_encoder.save_pretrained(path.join(output_dir, 'utt_encoder'))
        model_to_save.context_encoder.save_pretrained(path.join(output_dir, 'context_encoder'))
        # torch.save(self.state_dict(), path.join(output_dir, 'all'))
        # save_module(model_to_save.classifier, path.join(output_dir, 'classifier.pkl')) 


    def load_finetuning(self, model_dir):       
        self.encoder_config = BertConfig.from_pretrained(model_dir,local_files_only=True)
        # self.tokenizer = BertTokenizer.from_pretrained(path.join(model_dir, 'tokenizer'))
        config=torch.load(model_dir+"/training_args.bin")

        output_dir_base = os.path.join(f"{Path(model_dir).parents[5]}",f"{config.model}/{config.model_size}/{config.version_load}/models/", f'checkpoint-{config.reload_from}') 
        self.tokenizer = BertTokenizer.from_pretrained(path.join(output_dir_base, 'tokenizer'))
        self.utt_encoder = BertForPreTraining.from_pretrained(path.join(output_dir_base, 'utt_encoder'))


        if self.lexicon:
            self.context_encoder = BertForSequenceClassification_lex_features.from_pretrained(path.join(model_dir, 'context_encoder'))
        else:
            self.context_encoder = BertForSequenceClassification_v2.from_pretrained(path.join(model_dir, 'context_encoder'))


    def save_pretrained(self, output_dir):   
        def save_module(model, save_path):       
            torch.save(model_to_save.state_dict(), save_path)
        def make_list_dirs(dir_list):
            for dir_ in dir_list: os.makedirs(dir_, exist_ok=True)
        make_list_dirs([path.join(output_dir, name) for name in ['tokenizer', 'utt_encoder', 'context_encoder', 'decoder']])        
        model_to_save = self.module if hasattr(self, 'module') else self
        model_to_save.encoder_config.save_pretrained(output_dir) # Save configuration file
        model_to_save.tokenizer.save_pretrained(path.join(output_dir,'tokenizer'))
        model_to_save.utt_encoder.save_pretrained(path.join(output_dir, 'utt_encoder'))
        model_to_save.context_encoder.save_pretrained(path.join(output_dir, 'context_encoder'))
        save_module(model_to_save.context_encoder, path.join(output_dir, 'context_encoder.pkl')) 
        save_module(model_to_save.context_mlm_trans, path.join(output_dir, 'context_mlm_trans.pkl')) 
        save_module(model_to_save.context_order_trans, path.join(output_dir, 'context_order_trans.pkl')) 
        model_to_save.decoder_config.save_pretrained(output_dir) # Save configuration file
        model_to_save.decoder.save_pretrained(path.join(output_dir, 'decoder'))
            
    def utt_encoding(self, context, utts_attn_mask):
        batch_size, max_ctx_len, max_utt_len = context.size() #context: [batch_size x diag_len x max_utt_len]
        
        utts = context.view(-1, max_utt_len) # [(batch_size*diag_len) x max_utt_len]
        utts_attn_mask = utts_attn_mask.view(-1, max_utt_len)
        _, utts_encodings, *_ = self.utt_encoder.bert(utts, utts_attn_mask,return_dict=False )     
        utts_encodings = utts_encodings.view(batch_size, max_ctx_len, -1)            
        return utts_encodings
    
    def add_features(self,vector,knowledge_vectors):
        knowledge_vectors=knowledge_vectors.unsqueeze(-1)

        if len(vector.shape)==3:
            filler=torch.zeros(knowledge_vectors.shape[0], knowledge_vectors.shape[1],11,device=vector.device)
            vector= torch.cat((vector,knowledge_vectors), dim=2)
            vector= torch.cat((vector,filler), dim=2)
        elif len(vector.shape)==2:

            filler=torch.zeros(knowledge_vectors.shape[0],11,device=vector.device)
            vector= torch.cat((vector,knowledge_vectors), dim=1)
            vector= torch.cat((vector,filler), dim=1)
        return vector

    def context_encoding(self, context, utts_attn_mask, ctx_attn_mask,turns):
        if self.train_base:
            utt_encodings = self.utt_encoding(context, utts_attn_mask)
        else:
            with torch.no_grad():
                
                utt_encodings = self.utt_encoding(context, utts_attn_mask)
        if self.with_turns:
            utt_encodings=self.add_features(utt_encodings,turns)
        # utt_encodings=self.add_features(utt_encodings,turns)
 
        context_hiddens, pooled_output, *_ = self.context_encoder(None, ctx_attn_mask, None, None, None, utt_encodings,  return_dict=False)
        # context_hiddens:[batch_size x ctx_len x dim]; pooled_output=[batch_size x dim]``
        
        return context_hiddens, pooled_output
    
    def forward(self, context, context_utts_attn_mask, context_attn_mask, context_mlm_targets, context_position_perm_id, context_position_ids, response,turns,features):
        self.train()
        batch_size, max_ctx_len, max_utt_len = context.size() #context: [batch_size x diag_len x max_utt_len]
        context_hiddens, context_encoding = self.context_encoding(context, context_utts_attn_mask, context_attn_mask,turns)
        
        
        ## train dialog flow modeling
        context_mlm_targets[context_mlm_targets==-100] = 0
        ctx_mlm_mask = context_mlm_targets.sum(2) #[batch_size x num_utts]
        if (ctx_mlm_mask>0).sum()==0: ctx_mlm_mask[0, 0]=1
        ctx_mlm_mask = ctx_mlm_mask>0
        
        with torch.no_grad():            
            _, mlm_tgt_encodings, *_ = self.utt_encoder.bert(context_mlm_targets[ctx_mlm_mask], context_utts_attn_mask[ctx_mlm_mask],return_dict=False) 
        if self.with_turns:
            mlm_tgt_encodings= self.add_features(mlm_tgt_encodings, turns[ctx_mlm_mask]) 
        if self.mlm_mode == 'mdn': # mixture density network
            mlm_pred_pi, mlm_pred_normal = self.context_mlm_trans(self.dropout(context_hiddens[ctx_mlm_mask])) 
            loss_ctx_mlm = self.context_mlm_trans.loss(mlm_pred_pi, mlm_pred_normal, mlm_tgt_encodings)
        else: # simply mean square loss
            mlm_pred_encodings = self.context_mlm_trans(self.dropout(context_hiddens[ctx_mlm_mask]))
            loss_ctx_mlm = MSELoss()(mlm_pred_encodings, mlm_tgt_encodings) # [num_selected_utts x dim]
        features_loss = self.SSK_task(self.dropout(context_encoding), features.to(torch.float32)) if self.lexicon else 0

                              
                                
        # context order prediction
        if isinstance(self.context_order_trans, SelfSorting):
            sorting_scores = self.context_order_trans(context_hiddens, context_attn_mask)
        else:
            sorting_scores = self.context_order_trans(context_hiddens)
        sorting_pad_mask = context_attn_mask==0
        sorting_pad_mask[context_position_perm_id<1]=True # exclude single-turn and unshuffled dialogs
        loss_ctx_uop = listNet(sorting_scores, context_position_ids, sorting_pad_mask) 
        #loss_ctx_uop = listMLE(sorting_scores, context_position_ids, sorting_pad_mask)
        
                  
        
                                 
        ## train decoder  
        dec_input, dec_target = response[:,:-1].contiguous(), response[:,1:].clone()

        dec_output,*_ = self.decoder(
            dec_input, dec_input.ne(self.tokenizer.pad_token_id).long(), None, None, None, None,
            encoder_hidden_states=context_hiddens, encoder_attention_mask=context_attn_mask, return_dict=False    
        )
            
        batch_size, seq_len, vocab_size = dec_output.size()
        dec_target[response[:, 1:] == self.tokenizer.pad_token_id] = -100
        dec_target[context_position_perm_id>1] = -100 # ignore responses whose context was shuffled
        loss_decoder = CrossEntropyLoss()(dec_output.view(-1, vocab_size), dec_target.view(-1)) 
        
        loss = loss_ctx_mlm + loss_ctx_uop + loss_decoder+ features_loss
        
        results = {'loss': loss, 
                   'loss_ctx_mlm':loss_ctx_mlm, 
                   'loss_ctx_uop':loss_ctx_uop, 
                   'loss_decoder': loss_decoder,
                   "features_loss": features_loss,
                  }
        
        return results
    
        
    def context_encoding2(self, context, utts_attn_mask, ctx_attn_mask,label,turns,features=None): #with prediction head
        with torch.no_grad():
            utt_encodings = self.utt_encoding(context, utts_attn_mask)
        if self.with_turns:
            utt_encodings=self.add_features(utt_encodings,turns)
        if features!=None:
            loss, logits  = self.context_encoder(None, ctx_attn_mask, None, None, None, utt_encodings,  return_dict=False,labels=label,additional_features=features)
        else: 
            loss, logits  = self.context_encoder(None, ctx_attn_mask, None, None, None, utt_encodings,  return_dict=False,labels=label)

        # out = self.context_encoder(None, ctx_attn_mask, None, None, None, utt_encodings,  return_dict=False)
        return loss, logits

    def context_encoding3(self, context, utts_attn_mask, ctx_attn_mask,label,turns,features=None):
        with torch.no_grad():
            utt_encodings = self.utt_encoding(context, utts_attn_mask)
        if self.with_turns:
            utt_encodings=self.add_features(utt_encodings,turns)
        if features!=None:
            loss, logits  = self.context_encoder(None, ctx_attn_mask, None, None, None, utt_encodings,  return_dict=False,labels=label,additional_features=features)
        return self.context_encoder.bert(None, ctx_attn_mask, None, None, None, utt_encodings,  return_dict=False)

    
    def context_encoding_with_lex(self, context, utts_attn_mask, ctx_attn_mask):
        if self.train_base:
            utt_encodings = self.utt_encoding(context, utts_attn_mask)
        else:
            with torch.no_grad():
                utt_encodings = self.utt_encoding(context, utts_attn_mask)

        context_hiddens, pooled_output, *_ = self.context_encoder(None, ctx_attn_mask, None, None, None, utt_encodings,  return_dict=False)
        # context_hiddens:[batch_size x ctx_len x dim]; pooled_output=[batch_size x dim]``
        
        return context_hiddens, pooled_output
    
    def train_classifier(self, context, context_utts_attn_mask, context_attn_mask, context_lm_targets, context_position_perm_id, context_position_ids, GSR_label=0,IMSR_label=0,turns=None,features=None):
            def flat_accuracy(preds, labels):
                pred_flat = np.argmax(preds, axis=1).flatten()
                labels_flat = labels.flatten()
                return np.sum(pred_flat == labels_flat) / len(labels_flat)

            label= GSR_label.type(torch.LongTensor).to( next(self.parameters()).device)
            if self.IMSR:
                label=IMSR_label.type(torch.LongTensor).to( next(self.parameters()).device)
            # with torch.no_grad():
            # context_hiddens, context_encoding = self.context_encoding(context, context_utts_attn_mask, context_attn_mask)
            self.train()
            # self.classifier.train()

            if not self.with_feature:
                features=None
            loss, logits = self.context_encoding2(context, context_utts_attn_mask, context_attn_mask, label=label,turns=turns,features=features)
                

            pred_res= torch.argmax(logits, dim=1)

            logits = logits.detach().cpu().numpy()
            label_ids = label.to('cpu').numpy()
            acu= flat_accuracy(logits, label_ids)
            f1_score_b = f1_score(label.cpu().data, pred_res.cpu().data, average='weighted',zero_division=0)
            precision_score_b= precision_score(label.cpu().data, pred_res.cpu().data, average='weighted',zero_division=0)
            results = {'loss': loss, 'loss_decoder': loss, "Accuracy": acu, "F1": f1_score_b, "precision_score_b":precision_score_b}
            # results = {'loss': loss, "Accuracy": acu}
            return results


    def train_dialog_flow(self, context, context_utts_attn_mask, context_attn_mask, context_lm_targets, context_position_perm_id, context_position_ids, response): 
            """
            only train the dialog flow model
            """
            self.context_encoder.train()# set the module in training mode.  
            self.context_mlm_trans.train()
            
            context_hiddens, context_encoding = self.context_encoding(context, context_utts_attn_mask, context_attn_mask)
            lm_pred_encodings = self.context_mlm_trans(self.dropout(context_hiddens))
            
            context_lm_targets[context_lm_targets==-100] = 0
            ctx_lm_mask = context_lm_targets.sum(2)
            if (ctx_lm_mask>0).sum()==0: ctx_lm_mask[0, 0]=1
            lm_pred_encodings = lm_pred_encodings[ctx_lm_mask>0]
            context_lm_targets = context_lm_targets[ctx_lm_mask>0]
            context_lm_targets_attn_mask = context_utts_attn_mask[ctx_lm_mask>0]
            
            with torch.no_grad():            
                _, lm_tgt_encodings, *_ = self.utt_encoder.bert(context_lm_targets, context_lm_targets_attn_mask, return_dict=False) 
        
            loss_ctx_mlm = MSELoss()(lm_pred_encodings, lm_tgt_encodings) # [num_selected_utts x dim]
            
            
            # context order prediction
            if isinstance(self.context_order_trans, SelfSorting):
                sorting_scores = self.context_order_trans(context_hiddens, context_attn_mask)
            else:
                sorting_scores = self.context_order_trans(context_hiddens)
            sorting_pad_mask = context_attn_mask==0
            sorting_pad_mask[context_position_perm_id<1]=True # exclude single-turn and unshuffled dialogs
            loss_ctx_uop = listNet(sorting_scores, context_position_ids, sorting_pad_mask) 
            loss = loss_ctx_mlm + loss_ctx_uop      
            return {'loss': loss, 'loss_ctx_mlm': loss_ctx_mlm, 'loss_ctx_uop':loss_ctx_uop}
        
    def train_decoder(self, context, context_utts_attn_mask, context_attn_mask, context_lm_targets, context_position_perm_id, context_position_ids, response):
            """
            only train the decoder
            """
            self.decoder.train()
            
            with torch.no_grad():
                context_hiddens, context_encoding = self.context_encoding(context, context_utts_attn_mask, context_attn_mask)
                
            ## train decoder  
            dec_input, dec_target = response[:,:-1].contiguous(), response[:,1:].clone()

            dec_output,*_ = self.decoder(dec_input, dec_input.ne(self.tokenizer.pad_token_id).long(), None, None, None, None,
                encoder_hidden_states=context_hiddens, encoder_attention_mask=context_attn_mask, return_dict=False)
                
            batch_size, seq_len, vocab_size = dec_output.size()
            dec_target[response[:, 1:] == self.tokenizer.pad_token_id] = -100
            dec_target[context_position_perm_id>1] == -100 # ignore responses whose contexts are shuffled
            loss_decoder = CrossEntropyLoss()(dec_output.view(-1, vocab_size), dec_target.view(-1)) 

            results = {'loss': loss_decoder, 'loss_decoder': loss_decoder}
            
            return results

    def  validate_classification(self, context, context_utts_attn_mask, context_attn_mask, context_lm_targets, context_position_perm_id, context_position_ids, GSR_label=0,IMSR_label=0,turns=None,features=None):
        
        self.eval()
        label= GSR_label.type(torch.LongTensor).to( next(self.parameters()).device)
        if self.IMSR:
                label=IMSR_label.type(torch.LongTensor).to( next(self.parameters()).device)        
        if not self.with_feature:
            features=None
        loss, logits = self.context_encoding2(context, context_utts_attn_mask, context_attn_mask, label=label,turns=turns,features=features)
        # logits= 
        self.train()
        return loss.item(), logits.argmax(1), logits

    def get_vector(self, context, context_utts_attn_mask, context_attn_mask, context_lm_targets, context_position_perm_id, context_position_ids, GSR_label=0,VED_label=0,turns=None,features=None):
        self.eval()
        label= GSR_label.type(torch.LongTensor).to( next(self.parameters()).device)
        context_hiddens, context_encoding = self.context_encoding3(context, context_utts_attn_mask, context_attn_mask, label=label,turns=turns,features=features)
        return context_encoding




    # def infrance_classifier(self, context, context_utts_attn_mask, context_attn_mask, context_lm_targets, context_position_perm_id, context_position_ids ,turns=None):
    #     self.eval()
    #     with torch.no_grad():
    #         utt_encodings = self.utt_encoding(context, context_utts_attn_mask)
    #     if turns is not None:
    #         utt_encodings=self.add_features(utt_encodings,turns)

    #     logits  = self.context_encoder(None, context_attn_mask, None, None, None, utt_encodings,  return_dict=False)
    #     self.train()

    #     return logits


    def predict(self, context, context_utts_attn_mask, context_attn_mask, context_lm_targets, context_position_perm_id, context_position_ids):
            self.eval()

            label= label.type(torch.LongTensor).to( next(self.parameters()).device)
            loss, logits = self.context_encoding2(context, context_utts_attn_mask, context_attn_mask)
            self.train()

            return logits



    def validate(self, context, context_utts_attn_mask, context_attn_mask, context_lm_targets, context_position_perm_id, context_position_ids, response,turns,features):
        with torch.no_grad():            

            results = self.forward(
                context, context_utts_attn_mask, context_attn_mask, context_lm_targets, context_position_perm_id, context_position_ids, response,turns,features)
        return results
    
    def generate(self, input_batch, max_len=30, num_samples=1, mode='sample'):    
        self.eval()
        device = next(self.parameters()).device
        context, context_utts_attn_mask, context_attn_mask = [t.to(device) for t in input_batch[:3]]    
        ground_truth = input_batch[6].numpy()
            
        context_hiddens, context_encoding = self.context_encoding(
            context, context_utts_attn_mask, context_attn_mask)

        generated = torch.zeros((num_samples,1), dtype=torch.long, device=device).fill_(self.tokenizer.cls_token_id)
                                                 # [batch_sz x 1] (1=seq_len)
        
        sample_lens= torch.ones((num_samples,1), dtype=torch.long, device=device) 
        len_inc = torch.ones((num_samples,1), dtype=torch.long, device=device) 
        for _ in range(max_len):
            outputs,*_ = self.decoder(
                generated, generated.ne(self.tokenizer.pad_token_id).long(), None, None, None, None,
                encoder_hidden_states=context_hiddens, encoder_attention_mask=context_attn_mask,    
             return_dict=False) # [batch_size x seq_len x vocab_size]
            next_token_logits = outputs[:, -1, :]/ self.decoder_config.temperature

            # repetition penalty from CTRL (https://arxiv.org/abs/1909.05858)
            for i in range(num_samples):
                for _ in set(generated[i].tolist()):
                    next_token_logits[i, _] /= self.decoder_config.repetition_penalty

            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=self.decoder_config.top_k, top_p=self.decoder_config.top_p)
            if mode == 'greedy': # greedy sampling:
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(torch.softmax(filtered_logits, dim=-1), num_samples=num_samples)
            next_token[len_inc==0]=self.tokenizer.pad_token_id
            generated = torch.cat((generated, next_token), dim=1)
            len_inc=len_inc*(next_token!=self.tokenizer.sep_token_id).long() # stop incresing length (set 0 bit) when EOS is encountered
            if len_inc.sum()<1: break
            sample_lens=sample_lens+len_inc   
                
        # to numpy
        sample_words = generated.data.cpu().numpy()
        sample_lens = sample_lens.data.cpu().numpy() 
        
        context = context.data.cpu().numpy()
        return sample_words, sample_lens, context, ground_truth # nparray: [repeat x seq_len]