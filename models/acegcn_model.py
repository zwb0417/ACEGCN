import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):

        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):

        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ACEGCNClassifier(nn.Module):
    def __init__(self, bert, opt):

        super().__init__()
        self.opt = opt
        self.gcn_model = GCNAbsaModel(bert, opt=opt)
        self.classifier = nn.Linear(100, opt.polarities_dim)

    def forward(self, input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch, src_mask_batch, syntax_matrix_batch):
        outputs1 = self.gcn_model(input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch, src_mask_batch, syntax_matrix_batch)
        logits = self.classifier(outputs1)

        return logits, None


class GCNAbsaModel(nn.Module):
    def __init__(self, bert, opt):

        super().__init__()
        self.opt = opt
        self.gcn = GCNBert(bert, opt, opt.num_layers)

    def forward(self, input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch, src_mask_batch, syntax_matrix_batch):
        h = self.gcn(input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch, src_mask_batch, syntax_matrix_batch)
        valid_length = mask_ids_batch.sum(dim=1).unsqueeze(-1)
        valid_length = torch.clamp(valid_length, min=1)

        global_sum = h.sum(dim=1)
        outputs1 = global_sum / valid_length
        return outputs1


class GCNBert(nn.Module):
    def __init__(self, bert, opt, num_layers):

        super(GCNBert, self).__init__()
        self.bert = bert
        self.embedding = bert.embeddings
        self.encoder = bert.encoder

        self.opt = opt
        self.layers = num_layers
        self.mem_dim = opt.bert_dim // 2
        self.attention_heads = opt.attention_heads
        self.bert_dim = opt.bert_dim
        self.bert_drop = nn.Dropout(opt.bert_dropout)
        self.pooled_drop = nn.Dropout(opt.bert_dropout)
        self.gcn_drop = nn.Dropout(opt.gcn_dropout)
        self.layernorm = LayerNorm(opt.bert_dim)

        self.attdim = 100
        self.W = nn.Linear(self.attdim,self.attdim)
        self.Wx= nn.Linear(self.attention_heads+self.attdim*2, self.attention_heads)
        self.Wxx = nn.Linear(self.bert_dim, self.attdim)
        self.Wi = nn.Linear(self.attdim,50)
        self.aggregate_W = nn.Linear(self.attdim*(self.layers+1), self.attdim)


        self.attn = MultiHeadAttention(opt.attention_heads, self.attdim)
        self.weight_list = nn.ModuleList()
        for j in range(self.layers):
            input_dim = self.bert_dim if j == 0 else self.mem_dim
            self.weight_list.append(nn.Linear(input_dim, self.mem_dim))

        self.affine1 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))
        self.affine2 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))

    def forward(self, input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch, src_mask_batch, syntax_matrix_batch):

        text_bert_indices, label_ids, attention_mask, pos_ids, vms, src_mask, syntax_matrix= input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch, src_mask_batch, syntax_matrix_batch
        src_mask = src_mask.unsqueeze(-2)
        batch = src_mask.size(0)
        len = src_mask.size()[2]

        emb = self.embedding(input_ids=text_bert_indices, position_ids=pos_ids)
        seq_length = emb.size(1)

        if vms is None:
            mask = (attention_mask > 0). \
                    unsqueeze(1). \
                    repeat(1, seq_length, 1). \
                    unsqueeze(1)
            mask = mask.float()
            mask = (1.0 - mask) * -10000.0
        else:
            mask = vms.unsqueeze(1)
            mask = mask.float()
            mask = (1.0 - mask) * -10000.0

        sequence_output = self.encoder(hidden_states=emb, attention_mask=mask, return_dict=True)
        sequence_output = sequence_output.last_hidden_state

        sequence_output = self.layernorm(sequence_output)
        gcn_inputs = self.bert_drop(sequence_output)

        gcn_inputs = self.Wxx(gcn_inputs)

        attn_tensor = self.attn(gcn_inputs, gcn_inputs, vms, syntax_matrix, src_mask)


        weight_adj=attn_tensor
        gcn_outputs=gcn_inputs
        layer_list = [gcn_inputs]

        for i in range(self.layers):
            gcn_outputs = gcn_outputs.unsqueeze(1).expand(batch, self.attention_heads, len, self.attdim)
            Ax = torch.matmul(weight_adj, gcn_outputs)
            Ax = Ax.mean(dim=1)

            Ax = self.W(Ax)
            weights_gcn_outputs = F.relu(Ax)

            gcn_outputs = weights_gcn_outputs
            layer_list.append(gcn_outputs)
            gcn_outputs = self.gcn_drop(gcn_outputs) if i < self.layers - 1 else gcn_outputs

            weight_adj=weight_adj.permute(0, 2, 3, 1).contiguous()
            node_outputs1 = gcn_outputs.unsqueeze(1).expand(batch, len, len, self.attdim)
            node_outputs2 = node_outputs1.permute(0, 2, 1, 3).contiguous()
            node = torch.cat([node_outputs1, node_outputs2], dim=-1)
            edge_n=torch.cat([weight_adj, node], dim=-1)
            edge = self.Wx(edge_n)
            edge = self.gcn_drop(edge) if i < self.layers - 1 else edge
            weight_adj=edge.permute(0,3,1,2).contiguous()


        outputs = torch.cat(layer_list, dim=-1)
        node_outputs=self.aggregate_W(outputs)
        node_outputs = F.relu(node_outputs)

        return node_outputs


def attention(query, key, vm_matrix, syntax_matrix, num_heads, mask=None, dropout=None,):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:

        scores = scores.masked_fill(mask == 0, -1e9)

    scores = torch.add(scores, syntax_matrix)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)



    def forward(self, query, key, vm, syntax_matrix, mask=None):
        mask = mask[:, :, :query.size(1)]
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key))
        ]
        nums_head=self.h
        attn = attention(query, key, vm, syntax_matrix, nums_head, mask=mask, dropout=self.dropout)
        return attn
