import torch
import torch.nn as nn 
import torch.nn.functional as F 
from transformers import BertModel, BertConfig
from layer import H_TransformerEncoder
from torch.nn.parameter import Parameter
import numpy as np
from torch.linalg import det
from datetime import datetime


class BiSyn_GAT_plus(nn.Module):
    def __init__(self, args):
        super().__init__()
        in_dim = args.bert_hidden_dim + args.hidden_dim # 拼接之后的向量长度
        self.args = args 

        self.intra_context_module = Intra_context(args)

        # inter_context_module
        if args.plus_AA:
            self.inter_context_module = Inter_context(args)

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(in_dim, args.num_class)
    
    def forward(self, inputs):
        length, bert_length, word_mapback, \
        adj, aa_graph, aa_graph_length, \
        map_AS, map_AS_idx, \
        bert_sequence, bert_segments_ids, \
        aspect_indi, \
        con_spans, \
        map_AA, map_AA_idx,\
        aa_choice_inner_bert, aa_choice_inner_bert_length,\
        con_node_leaf_token_map, con_node_adj, D = inputs

        # aspect-level
        Intra_context_input = (length[map_AS], bert_length[map_AS], word_mapback[map_AS], aspect_indi, bert_sequence, bert_segments_ids, adj[map_AS], con_spans, con_node_leaf_token_map, con_node_adj, D)

        Intra_context_output = self.intra_context_module(Intra_context_input)

        # sentence-level
        if self.args.plus_AA and map_AA.numel(): # BiSyn-GAT+
            Inter_context_input = (aa_choice_inner_bert, aa_choice_inner_bert_length, 
                                    map_AA, map_AA_idx, map_AS, map_AS_idx, 
                                        aa_graph_length, aa_graph)
            # sentence-level to aspect-level
            hiddens = self.inter_context_module(as_features = Intra_context_output, 
                                                inputs = Inter_context_input, \
                                                context_encoder = self.intra_context_module.context_encoder if self.args.borrow_encoder else None)
            
        else: # BiSyn-GAT
            hiddens = Intra_context_output

        # aspect-level
        logits = self.classifier(self.dropout(hiddens))
        return logits
class GCN(nn.Module):
     def __init__(self, input_dim, output_dim):
        super(GCN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        print('input dim:', input_dim)
        print('output dim:', output_dim)
        self.layers = GraphConvLayer(self.input_dim, self.output_dim)

     def forward(self, x, adj, D):
        h, _ = self.layers(x, adj, D)
        return h
# *************新增代码结束**************
class GraphConvLayer(nn.Module):
    """ A GCN module operated on adj graphs. """

    def __init__(self, input_dim, output_dim):
        super(GraphConvLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = nn.Dropout(0.1)
        self.W = nn.Linear(self.input_dim, self.output_dim )
        self.sigmoid = nn.Sigmoid()
        self.layernorm = nn.LayerNorm(self.output_dim)
        self.non_linearity = nn.ReLU(inplace=True) 
        # self.b_gate = Parameter()
    def forward(self, bert_con_node_tokens, con_node_adj, con_node_adj_degree_matrix):

        weights = torch.bmm(con_node_adj_degree_matrix, con_node_adj)
        weights = torch.bmm(weights, con_node_adj_degree_matrix) # 8 x 12 x 12

        H = self.W(torch.bmm(weights, bert_con_node_tokens))
        return H, con_node_adj
        '''
        #con_node_tokens = self.W(bert_con_node_tokens) # 8 x 12 x 200
        con_node_tokens = bert_con_node_tokens
        con_node_tokens = con_node_tokens * bert_con_node_tokens_mask.unsqueeze(dim = -1)
        # print(con_node_tokens[3][8:12])
        # print(con_node_tokens[4][8:12])
        V_gate = Parameter(torch.Tensor(len(con_node_lens), self.gcn_dim, max(con_node_lens)))
        nn.init.xavier_normal_(V_gate) # 8 x 200 x 12
        V_gate = V_gate.to(args.device)
        # con_node_tokens = con_node_tokens.view(len(con_node_lens) * max(con_node_lens), self.gcn_dim) # 
        c_c_gate = torch.bmm(con_node_tokens, V_gate)  # 8 x 12 x 200 x 8 x 200 x 12 -> 8 x 12 x 12
        #c_c_gate = c_c_gate.view(len(con_node_lens), max(con_node_lens), max(con_node_lens)) # 8 x 12 x 12
        # con_node_tokens = con_node_tokens.view(len(con_node_lens), max(con_node_lens), self.gcn_dim) # 8 x 12 x 200
        c_c_gate = self.sigmoid(c_c_gate) * con_node_adj # 8 x 12 x 12  * 8 x 12 x 12 -> 8 x 12 x 12
        #print(con_node_adj_mask[3])
        #print(con_node_adj_mask[4])
        c_c_gate = c_c_gate * con_node_adj_mask # 8 x 12 x 12 
        b = len(con_node_lens) # 句子
        c = max(con_node_lens) # 最大成分数量
        gcn_con_node_tokens = con_node_tokens - con_node_tokens
        # print(sum(gcn_con_node_tokens))
        for idx_b in range(b):
            for idx_c_i in range(c):
                count = 0
                for idx_c_j in range(c):
                    if c_c_gate[idx_b][idx_c_i][idx_c_j] != 0. :
                        gcn_con_node_tokens[idx_b][idx_c_i] += con_node_tokens[idx_b][idx_c_j] * c_c_gate[idx_b][idx_c_i][idx_c_j]
                        count += 1
                # 这个地方需不需要加一个平均值呢，就是有几条边就除以几
                if count != 0:
                    gcn_con_node_tokens[idx_b][idx_c_i] /= count
        
        gcn_con_node_tokens = self.dropout(gcn_con_node_tokens)
        gcn_con_node_tokens += con_node_tokens
        gcn_con_node_tokens = self.layernorm(gcn_con_node_tokens)
        #gcn_con_node_tokens = self.W(gcn_con_node_tokens)
        # gcn_con_node_tokens = self.non_linearity(gcn_con_node_tokens)
        return gcn_con_node_tokens
    '''
# *************新增代码结束**************

# Intra-context module
class Intra_context(nn.Module):
    def __init__(self, args):
        super().__init__()
         
        self.args = args # 

        bert_config = BertConfig.from_pretrained('bert-base-uncased')
        bert_config.output_hidden_states = True
        bert_config.num_labels = 3

        self.layer_drop = nn.Dropout(args.layer_dropout)
        self.context_encoder = BertModel.from_pretrained('bert-base-uncased',config=bert_config)
        self.dense = nn.Linear(args.bert_hidden_dim, args.hidden_dim) # 768 -> 200
        self.graph_encoder = H_TransformerEncoder(
            d_model = args.hidden_dim,
            nhead = args.attn_head,
            num_encoder_layers = args.num_encoder_layer, # 3
            inner_encoder_layers = args.max_num_spans, # 3
            dropout = args.layer_dropout,
            dim_feedforward = args.bert_hidden_dim, # 
            activation = 'relu',
            layer_norm_eps = 1e-5
        )
        self.c_c_gcn = GCN(args.gcn_dim, args.gcn_dim) # 初始化
        self.dropout = nn.Dropout(args.con_attn_dropout)
        self.layernorm = nn.LayerNorm(args.gcn_dim)
       
    def forward(self, inputs):

        length, bert_lengths,  word_mapback, mask, bert_sequence, bert_segments_ids, adj, con_spans, con_node_leaf_token_map, con_node_adj, D = inputs

        ###############################################################
        # 1. contextual encoder
        bert_outputs = self.context_encoder(bert_sequence, token_type_ids=bert_segments_ids)

        bert_out, bert_pooler_out = bert_outputs.last_hidden_state, bert_outputs.pooler_output # 8 x 22 x 768; 8 x 768 pooler_out是cls再经过线性变换得到的，可理解为表示句子特征

        bert_out = self.layer_drop(bert_out)

        # rm [CLS] 
        bert_seq_indi = ~sequence_mask(bert_lengths).unsqueeze(dim=-1)
        bert_out = bert_out[:, 1:max(bert_lengths) + 1, :] * bert_seq_indi.float() # 8 x 22 x 768 -> 截取句子部分 -> 8 x 16 x 768
        word_mapback_one_hot = (F.one_hot(word_mapback).float() * bert_seq_indi.float()).transpose(1, 2) # 8 x 12 x 16 X 8 x 16 x 1

        # *************新增代码开始**************
        bert_out_con = torch.bmm(word_mapback_one_hot.float(), bert_out) # 8 x 12 x 768, b x token_len x token_dim用于构建成分树结点的bert向量
        # *************新增代码结束**************

        bert_out = torch.bmm(word_mapback_one_hot.float(), self.dense(bert_out)) # 8 x 16 x 768 -> 8 x 12 x 200
        

        # average 把累加的向量进行了平均，这个方法可以应用到后面
        wnt = word_mapback_one_hot.sum(dim=-1)
        wnt.masked_fill_(wnt == 0, 1)
        bert_out = bert_out / wnt.unsqueeze(dim=-1)  # 8 x 12 x 200

        # *************新增代码开始**************
        bert_out_con = bert_out_con / wnt.unsqueeze(dim=-1) # 用多个bert_id表示的单词，求和取平均
        # *************新增代码结束**************
        # *************新增代码开始**************
        
        #now_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        #print("构建开始时间：{}".format(now_time))
        
        # 1.构建成分结点
        bert_con_node_tokens = torch.bmm(con_node_leaf_token_map.float(), bert_out_con) # 通过叶子结点向量相加的形式构建成分结点
        # average
        con_wnt = con_node_leaf_token_map.sum(dim=-1) # 8 x 12
        con_wnt.masked_fill_(con_wnt == 0, 1)
        bert_con_node_tokens = bert_con_node_tokens / con_wnt.unsqueeze(dim=-1)
        #********成分结点**********
        bert_con_node_tokens = self.dense(bert_con_node_tokens)  # 8 x 12最大成分个数 x 768 -> 8 x 12最大成分个数 x 200
        # 2. gcn 成分结点之间的信息融合  
        '''   
        #生成成分结点mask矩阵
        con_node_lens_len = len(con_node_lens)
        bert_con_node_tokens_maxlen = max(con_node_lens)
        bert_con_node_tokens_mask = np.zeros((con_node_lens_len, bert_con_node_tokens_maxlen), dtype=np.float32)
        for idx in range(con_node_lens_len):
            con_len = con_node_lens[idx]
            bert_con_node_tokens_mask[idx, : con_len]=1
        bert_con_node_tokens_mask = torch.FloatTensor(bert_con_node_tokens_mask)
        bert_con_node_tokens_mask = bert_con_node_tokens_mask.to(self.args.device)  # 8 x 12成分
        #生成成分结点之间邻居矩阵mask矩阵
        con_node_adj_mask = np.zeros((con_node_lens_len, bert_con_node_tokens_maxlen, bert_con_node_tokens_maxlen), dtype=np.float32)
        for idx in range(con_node_lens_len):
            con_len = con_node_lens[idx]
            con_node_adj_mask[idx, :con_len, :con_len] = 1
        con_node_adj_mask = torch.FloatTensor(con_node_adj_mask)
        con_node_adj_mask = con_node_adj_mask.to(self.args.device)
        '''
        #now_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        #print("构建结束时间：{}".format(now_time))
        
        
                
        #now_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        #print("进入gcn时间：{}".format(now_time))
        
        con_node = self.c_c_gcn(bert_con_node_tokens, con_node_adj, D)  # b x con_len x dim
        
        #now_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        #print("出 gcn时间：{}".format(now_time))
        # 3. 将成分结点的信息返回给构成此成分结点的单词
        '''
        con_node(8 x max_con_len x gcn_dim), 8 x 12成分 x 200
        bert_out_con(8 x max_tokens_len x token_dim), 8 x 12单词 x 768维度
        con_node_leaf_token_map(8 x 12 x 12), 8 x 12个成分 x 12个单词
        利用con_node_leaf_token_map将con_node中的信息返回给bert_out_con
        '''
        #now_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        #print("解构开始时间：{}".format(now_time))
        
        # query
        bert_out_con_query = bert_out_con 
        bert_out_con_query = self.dense(bert_out_con_query)  # 768 -> 200
        # key
        con_node_key = con_node
        con_node_key = con_node_key.transpose(1,2)  # b x con_len x con_dim -> b x con_dim x con_len
        # query * key = score
        # query: b x tokens_len x token_dim768 X key : b x con_dim x con_len200 -> b x tokens_len x con_len 
        score = torch.bmm(bert_out_con_query, con_node_key) # b x tokens_len x con_len

        # mask
        con_node_leaf_token_map = con_node_leaf_token_map.transpose(1,2)
        score_masked = con_node_leaf_token_map * score
        weights = F.softmax(score_masked, dim =1) 

        '''
        bert_out_tokens_con = bert_out_con
        
        con_node_token_map = con_node_leaf_token_map.transpose(0,1) # 8 x 12 x 12 -> 12 x 8 x 12 sentences x con_len x token_len -> con_len x sentences x token_len
        con_len, sentences, token_len = con_node_token_map.shape  # 
        #print(con_node_token_map.shape)
        con_node_token_map = con_node_token_map.contiguous().view(con_len, -1)  # 合并后两维的维度 con_len x (sentences x token_len) 12 x 96 (0~1)
        _, sen_mutl_token = con_node_token_map.shape  # 96
        token_dim = bert_out_tokens_con.shape[2]  # 768
        con_node_token_map = con_node_token_map.unsqueeze(2).expand(con_len, sen_mutl_token, token_dim)  # 12 x 96 x 768 (0~1)

        bert_out_tokens_con = bert_out_tokens_con.view(-1, token_dim)  # 合并前两维的维度 8 x 12 x 768 -> 96 x 768
        b, d = bert_out_tokens_con.shape
        bert_out_tokens_con = bert_out_tokens_con.unsqueeze(0).expand(con_len, b, d) # 12 x 96 x 768

        con_token_query = bert_out_tokens_con * con_node_token_map  # 12 x 96 x 768(成分 x (句子数量 x 单词数量) x 单词维度)

        con_token_query = con_token_query.view(con_len, sentences, token_len, token_dim)  # 12 x 96 x 768 -> 12(成分) x 8（句子） x 12（单词） x 768

        con_token_query = con_token_query.transpose(0, 1)  # 12 x 8 x 12 x 768 -> 8 x 12(成分) x 12（单词） x 768
        
        # con_token_query: 8句子 x 12成分 x 12单词 x 768单词维度
        # con_node: 8句子 x 12成分 x 200维度
    
        con_node_key = con_node.contiguous()  # key 
        dim = con_node_key.shape[2]

        con_token_query = self.dense(con_token_query) # 8 x 12 x 12 x 768 -> 8 x 12 x 12 x 200
        con_token_query = con_token_query.view(sentences * con_len, token_len, dim)  # 96 x 12 x 200

        con_node_key = con_node_key.view(sentences * con_len, dim)  # 96 x 200
        con_node_key = con_node_key.unsqueeze(1)  # 96 x 200 -> 96 x 1 x 200

        con_token_query = con_token_query.transpose(1,2)  # 96 x 200 x 12

        source  = torch.bmm(con_node_key, con_token_query)  # 96 x 1 x 200 X 96 x 200 x 12 -> 96 x 1成分 x 12单词
        source = source.squeeze(1)  # 96 x 12
        source = source.view(sentences, con_len, -1) # 8 x 12成分 x 12单词

        weights = F.softmax(source, dim=-1)
        weights = self.dropout(weights)  # 8 x 12成分 x 12单词 8个句子每个句子最多12个成分，每个成分跟每个单词的权重
        '''


        bert_out_con = self.dense(bert_out_con) # 8 x 12 x 200
        out = bert_out_con - bert_out_con

        out = torch.bmm(weights, con_node)
        
        out = self.layernorm(out)
        
        out = out + bert_out_con
        out = self.layernorm(out)
        
        #now_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        #print("解构结束时间：{}".format(now_time))
        # *************新增代码结束**************

        ###############################################################
        # 2. graph encoder
        key_padding_mask = sequence_mask(length)  # [B, seq_len]

        # from phrase(span) to form mask
        B, N, L = con_spans.shape # 8, 3, 12 
        span_matrix = get_span_matrix_4D(con_spans.transpose(0, 1)) # B输入,N层数,L长度 -> N,B,L -> N,B,L,L 3x8x12x12 # transpose的目的应该是处理相同的层的跨度，增加一维的目的是确定两两结点之间有没有跨度的关系

        if self.args.con_dep_version == 'con_add_dep':
            # adj + span
            adj_matrix = adj.unsqueeze(dim=0).repeat(N, 1, 1, 1)
            assert ((adj_matrix[0] != adj_matrix[1]).sum() == 0)
            span_matrix = (span_matrix + adj_matrix).bool()

        elif self.args.con_dep_version == 'wo_dep':
            # only span
            pass 
    
        elif self.args.con_dep_version == 'wo_con':
            # only adj
            adj_matrix = adj.unsqueeze(dim=0)
            span_matrix = adj_matrix.repeat(N, 1, 1, 1)

        elif self.args.con_dep_version == 'con_dot_dep':
            # adj * span
            adj_matrix = adj.unsqueeze(dim=0).repeat(N, 1, 1, 1) # 8 x 12 x 12 -> 1 x 8 x 12 x 12 -> repeat3次 x 8 x 12 x 12 
            assert ((adj_matrix[0] != adj_matrix[1]).sum() == 0) # 判断repeat之后的n个相同的8 x 12 x 12 矩阵是不是都一样的
            span_matrix = (span_matrix * adj_matrix).bool() # 3 x 8 x 12 x 12 boolean类型, 只要不是0就返回true
        
        graph_out = self.graph_encoder(bert_out,
                                       mask=span_matrix, src_key_padding_mask=key_padding_mask)
        ###############################################################
        # 3. fusion
        asp_wn = mask.sum(dim=1).unsqueeze(-1)  # aspect words num 8 x 1
        mask = mask.unsqueeze(-1).repeat(1, 1, self.args.hidden_dim)  # mask for h 8 x 12 -> 8 x 12 x 200
        # h_t
        bert_enc_outputs = (bert_out * mask).sum(dim=1) / asp_wn # 

        # g_t
        graph_enc_outputs = (graph_out * mask).sum(dim=1) / asp_wn  # mask h

        # out
        #*********新增代码开始**********
        con_con_out = (out * mask).sum(dim=1) / asp_wn
        #*********新增代码结束**********

        as_features = torch.cat([graph_enc_outputs + con_con_out + bert_enc_outputs, bert_pooler_out],-1)
        return as_features

    def cof1(self, M,index):
        zs = M[:index[0]-1,:index[1]-1]
        ys = M[:index[0]-1,index[1]:]
        zx = M[index[0]:,:index[1]-1]
        yx = M[index[0]:,index[1]:]
        s = torch.cat((zs,ys),axis=1)
        x = torch.cat((zx,yx),axis=1)
        return torch.det(torch.cat((s,x),axis=0))
    
    def alcof(self, M,index):
        return pow(-1,index[0]+index[1])*self.cof1(M,index)
    
    def adj(self, M):
        result = torch.zeros((M.shape[0],M.shape[1]))
        result = result.to(self.args.device)
        for i in range(1,M.shape[0]+1):
            for j in range(1,M.shape[1]+1):
                result[j-1][i-1] = self.alcof(M,[i,j])
        return result
    
    def invmat(self, M):
        return 1.0/torch.det(M)*self.adj(M)
        

# Inter-context module
class Inter_context(nn.Module):
    def __init__(self, args, sent_encoder=None):
        super().__init__()
        self.args = args 
        in_dim = args.bert_hidden_dim + args.hidden_dim 
        self.layer_drop = nn.Dropout(args.layer_dropout)
        if not args.borrow_encoder:
            self.sent_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.dense = nn.Linear(args.bert_hidden_dim, args.hidden_dim)

        self.con_aspect_graph_encoder = H_TransformerEncoder(
                        d_model = in_dim,
                        nhead = args.attn_head,
                        num_encoder_layers = args.aspect_graph_num_layer,
                        inner_encoder_layers = args.aspect_graph_encoder_version,
                        dropout = args.layer_dropout,
                        dim_feedforward=args.hidden_dim,
                        activation = 'relu',
                        layer_norm_eps=1e-5
                    )

    def forward(self, as_features, inputs, context_encoder=None):
        aa_choice_inner_bert, aa_choice_inner_bert_length, \
            map_AA, map_AA_idx, map_AS, map_AS_idx, \
                aa_graph_length, aa_graph = inputs

        need_change = (aa_graph_length[map_AS] > 1)

        inner_v_node, inner_v = self.forward_bert_inner( aa_choice_inner_bert,
                                                        aa_choice_inner_bert_length,
                                                        context_encoder) 

        rela_v_inner = torch.cat((inner_v_node.sum(dim=1), inner_v), dim=-1)
        AA_features = self.con_aspect_graph(rela_v_inner, 
                                            as_features, 
                                            map_AA, map_AA_idx, 
                                            map_AS, map_AS_idx,
                                            aa_graph_length, aa_graph)



        AA_features = AA_features * need_change.unsqueeze(dim=-1) + as_features * ~(need_change).unsqueeze(dim=-1)

        fusion_features = AA_features + as_features
        
        return fusion_features



    def forward_bert_inner(self, aa_choice_inner_bert, aa_choice_inner_bert_length, context_encoder = None):
                
        bert_outputs = self.sent_encoder(aa_choice_inner_bert) if context_encoder is None else context_encoder(aa_choice_inner_bert)
        bert_out, bert_pool_output = bert_outputs.last_hidden_state, bert_outputs.pooler_output
        bert_out = self.layer_drop(bert_out)
        # rm [CLS] representation
        bert_seq_indi = ~sequence_mask(aa_choice_inner_bert_length).unsqueeze(dim=-1)
        bert_out = bert_out[:, 1:max(aa_choice_inner_bert_length) + 1, :] * bert_seq_indi.float()

        inner_v_node = self.dense(bert_out)
        inner_v = bert_pool_output
        return inner_v_node, inner_v


    def con_aspect_graph(self, 
                        rela_v, 
                        as_features, 
                        map_AA, map_AA_idx, map_AS, map_AS_idx, 
                        aa_graph_length, aa_graph):
        B = map_AS.max() + 1
        L = map_AA_idx.max() + 1
        graph_input_features = torch.zeros((B, L, as_features.shape[-1]), device=as_features.device)

        graph_input_features[map_AA, map_AA_idx] = rela_v
        graph_input_features[map_AS, map_AS_idx] = as_features


        aa_graph_key_padding_mask = sequence_mask(aa_graph_length)
        
        if self.args.aspect_graph_encoder_version == 1:
            # split and share parameters
            forward_ = self.con_aspect_graph_encoder(graph_input_features,
                                                    mask=aa_graph.unsqueeze(0),
                                                    src_key_padding_mask=aa_graph_key_padding_mask)
        
            backward_ = self.con_aspect_graph_encoder(graph_input_features,
                                                    mask=aa_graph.transpose(1, 2).unsqueeze(0),
                                                    src_key_padding_mask=aa_graph_key_padding_mask)
            mutual_influence = forward_ + backward_
        
        elif self.args.aspect_graph_encoder_version == 2:
            # not split    
            mutual_influence = self.con_aspect_graph_encoder(
                graph_input_features,
                mask = torch.cat((aa_graph.unsqueeze(dim=0), aa_graph.transpose(1,2).unsqueeze(dim=0)),dim=0),
                src_key_padding_mask = aa_graph_key_padding_mask
            )
        return mutual_influence[map_AS, map_AS_idx]


def sequence_mask(lengths, max_len=None):
    """
    create a boolean mask from sequence length `[batch_size,  seq_len]`
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return torch.arange(0, max_len, device=lengths.device).type_as(lengths).unsqueeze(0).expand(
        batch_size, max_len
    ) >= (lengths.unsqueeze(1))


def get_span_matrix_4D(span_list, rm_loop=False, max_len=None):
    '''
    span_list: [N,B,L]
    return span:[N,B,L,L]
    '''
    # [N,B,L]
    N, B, L = span_list.shape
    span = get_span_matrix_3D(span_list.contiguous().view(-1, L), rm_loop, max_len).contiguous().view(N, B, L, L)
    return span


def get_span_matrix_3D(span_list, rm_loop=False, max_len=None):
    # [N,L]
    origin_dim = len(span_list.shape)
    if origin_dim == 1:  # [L]
        span_list = span_list.unsqueeze(dim=0)
    N, L = span_list.shape
    if max_len is not None:
        L = min(L, max_len)
        span_list = span_list[:, :L]
    span = span_list.unsqueeze(dim=-1).repeat(1, 1, L)
    span = span * (span.transpose(-1, -2) == span)
    if rm_loop:
        span = span * (~torch.eye(L).bool()).unsqueeze(dim=0).repeat(N, 1, 1)
        span = span.squeeze(dim=0) if origin_dim == 1 else span  # [N,L,L]
    return span