from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.init as weight_init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import CrossEntropyLoss
import os
import numpy as np
import math
import random
from queue import PriorityQueue
import operator
import sys
from transformers import BertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
import json
import pickle
from sklearn.metrics.pairwise import cosine_similarity

parentPath = os.path.abspath("..")
sys.path.insert(0, parentPath)# add parent folder to path so as to import common modules

class MLP(nn.Module):
    def __init__(self, input_size, arch, output_size, activation=nn.ReLU(), batch_norm=True, init_w=0.02, discriminator=False):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.init_w= init_w
        
        if type(arch) == int: arch= str(arch) # simple integer as hidden size
        layer_sizes = [input_size] + [int(x) for x in arch.split('-')]
        self.layers = []

        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
            self.add_module("layer"+str(i+1), layer)            
            if batch_norm and not(discriminator and i==0):# if used as discriminator, then there is no batch norm in the first layer
                bn = nn.BatchNorm1d(layer_sizes[i+1], eps=1e-05, momentum=0.1)
                self.layers.append(bn)
                self.add_module("bn"+str(i+1), bn)
            self.layers.append(activation)
            self.add_module("activation"+str(i+1), activation)

        layer = nn.Linear(layer_sizes[-1], output_size)
        self.layers.append(layer)
        self.add_module("layer"+str(len(self.layers)), layer)

        self.init_weights()

    def forward(self, x):
        if x.dim()==3:
            sz1, sz2, sz3 = x.size()
            x = x.view(sz1*sz2, sz3)
        for i, layer in enumerate(self.layers):
            x = layer(x)
        if x.dim()==3:
            x = x.view(sz1, sz2, -1)
        return x

    def init_weights(self):
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, self.init_w)
                layer.bias.data.fill_(0)
            except:
                pass
  

class BertForSequenceClassification_v2(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()
        self.post_init()
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            # loss_fct = CrossEntropyLoss(weight=[1.000,4.00])
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertForSequenceClassification_lex_features(BertForSequenceClassification):
    def __init__(self,config):
        super().__init__(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size +5, config.num_labels)
        self.init_weights()
        self.post_init()



    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        additional_features=None, 
        return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output= torch.cat((pooled_output,additional_features),axis=1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # loss_fct = CrossEntropyLoss(weight=[1.000,4.00])
            # loss_fct = CrossEntropyLoss(weight=torch.FloatTensor([1.000,4.00]).cuda())
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class MixtureDensityNetwork(nn.Module):
    """
    Mixture density network. [Bishop, 1994]. Adopted from https://github.com/tonyduan/mdn
    References: 
        http://cbonnett.github.io/MDN.html
        https://towardsdatascience.com/a-hitchhikers-guide-to-mixture-density-networks-76b435826cca
    ----------
    dim_in: int; dimensionality of the covariates
    dim_out: int; dimensionality of the response variable
    n_components: int; number of components in the mixture model
    """
    def __init__(self, dim_in, dim_out, n_components):
        super().__init__()
        self.n_components = n_components
        
        self.pi_network = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ELU(),
            nn.Linear(dim_in, n_components)
        )
        self.normal_network = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ELU(),
            nn.Linear(dim_in, 2 * dim_out * n_components),
        )

    def forward(self, x):
        pi_logits = self.pi_network(x)
        pi = torch.distributions.OneHotCategorical(logits=pi_logits)
        
        normal_params = self.normal_network(x)
        mean, sd = torch.split(normal_params, normal_params.shape[1] // 2, dim=1)
        mean = torch.stack(mean.split(mean.shape[1] // self.n_components, 1))
        sd = torch.stack(sd.split(sd.shape[1] // self.n_components, 1))
        normal = torch.distributions.Normal(mean.transpose(0, 1), torch.exp(sd).transpose(0, 1))
        
        return pi, normal

    def loss(self, pi, normal, y):
        loglik = normal.log_prob(y.unsqueeze(1).expand_as(normal.loc))
        loglik = torch.sum(loglik, dim=2)
        loss = -torch.logsumexp(torch.log(pi.probs) + loglik, dim=1)
        return loss.mean()

    def sample(self, x):
        pi, normal = self.forward(x)
        samples = torch.sum(pi.sample().unsqueeze(2) * normal.sample(), dim=1)
        return samples

