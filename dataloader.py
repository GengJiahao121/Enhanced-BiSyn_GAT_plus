import os 
import json
import torch 
import numpy as np 
from transformers import BertTokenizer

import copy 
import random 
import itertools 
from itertools import chain

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter, _MultiProcessingDataLoaderIter
from spans import *

class ABSA_Dataset(Dataset):
    def __init__(self, args, file_name, vocab, tokenizer):
        super().__init__()

        # load raw data
        # 读取.json文件中的列表数据
        with open(file_name,'r',encoding='utf-8') as f:
            raw_data = json.load(f) # 列表，元素为单个句子构成各种类别的集合

            if args.need_preprocess:
                raw_data = self.process_raw(raw_data)
                new_file_name = file_name.replace('.json','_con.json')
                with open(new_file_name, 'w', encoding='utf-8') as f:
                    json.dump(raw_data,f)
                print('Saving to:', new_file_name)

        self.data = self.process(raw_data, vocab, args, tokenizer)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

    
    def process_raw(self, data):
        # get parserd data
        # we already provide here
        pass
    # **************改进部分开始***********
    def get_con_node_leafs(self, args, con_node_index, con_mapnode, con_children):
        leafs = []
        def _get_con_node_leafs(con_node_index, con_mapnode, con_children):
            if con_mapnode[con_node_index][-3: ] != args.special_token:
                leafs.append(con_node_index)
            else:
                for child in con_children[con_node_index]:
                    _get_con_node_leafs(child, con_mapnode, con_children)
        _get_con_node_leafs(con_node_index, con_mapnode, con_children)
        return leafs

    def get_con_node_leaf_dict(self, args, con_children, con_mapnode, mapback_con):
        con_leaf = {}
        for con_node_index in mapback_con: # 成分列表中的每一个结点
            # 去con_children中寻找其所有孩子
            # 孩子列表中的每个孩子
            for child_index in con_children[con_node_index]:
                # 去con_mapnode中寻找这个child_index对应的元素是单词还是成分结点
                if con_mapnode[child_index][-3: ] != args.special_token: # 是单词，就添加到con_leaf词典中
                    # 是单词，就添加到con_leaf词典中
                    # 如果词典中不存在该key值，就加入进去
                    if con_node_index not in con_leaf.keys():
                        con_leaf[con_node_index] = [con_mapnode[child_index]]
                    else:
                        con_leaf[con_node_index].append(con_mapnode[child_index])
                else:
                    # 不是单词，就是成分结点，那就递归下去
                    leafs = self.get_con_node_leafs(args, child_index, con_mapnode, con_children)
                    if con_node_index not in con_leaf.keys():
                        con_leaf[con_node_index] = [con_mapnode[leaf] for leaf in leafs]
                    else:
                        for leaf in leafs:
                            con_leaf[con_node_index].append(con_mapnode[leaf])
        return con_leaf


    def get_con_node_leafs_token_map(self, con_leaf, tokens):
        con_leaf_token_map = {}
        for con_node_index in con_leaf.keys():
            con_leaf_token_map[con_node_index] = [0] * len(tokens)
        for con_node_index, leaf_tokens in con_leaf.items():
            for idx,token in enumerate(tokens):
                if token in leaf_tokens:
                    con_leaf_token_map[con_node_index][idx] = 1
                else:
                    pass
        return con_leaf_token_map


    def get_con_node_adj(self, con_head, mapback_con):
        con_node_adj = np.zeros((len(mapback_con), len(mapback_con)))

        for idx_i, i in enumerate(mapback_con):
            for idx_j, j in enumerate(mapback_con):
                if con_head[i] == j or con_head[j] == i:
                    con_node_adj[idx_i][idx_j] = 1
                else:
                    pass
        return con_node_adj
    # **************改进部分结束***********

    def process(self, data, vocab, args, tokenizer):
        # vocab中的token和polarity分开
        token_vocab = vocab['token'] # 单词词典
        pol_vocab = vocab['polarity'] # 情感极性词典

        processed = [] # 初始化一个列表，用来存储处理之后的数据
        max_len = args.max_len # 
        CLS_id = tokenizer.convert_tokens_to_ids(["[CLS]"]) # [CLS]在bert-cased下的token2id为101 [SEP]为102
        SEP_id = tokenizer.convert_tokens_to_ids(["[SEP]"])
        sub_len = len(args.special_token) # '[N]'


        for d in data: # 处理单个数据
            tok = list(d['token'])
            if args.lower: 
                tok = [t.lower() for t in tok]
            
            text_raw_bert_indices, word_mapback, _ = text2bert_id(tok, tokenizer) # text_raw_bert_indices: 单词在tokenizer中对用的id， word_mapback: index, _ : 单词长度

            text_raw_bert_indices = text_raw_bert_indices[:max_len] # 句子最大长度100，直接截取
            word_mapback = word_mapback[:max_len]

            length = word_mapback[-1] + 1

            # tok = tok[:length]
            bert_length = len(word_mapback)

            dep_head = list(d['dep_head'])[:length] 

            # map2id 
            # tok = [token_vocab.stoi.get(t, token_vocab.unk_index) for t in tok]
            
            # con
            con_head = d['con_head']
            con_mapnode = d['con_mapnode']
            con_path_dict, con_children = get_path_and_children_dict(con_head)  # con_path_dict : 成分树所有结点到跟结点的路径， con_children: 所有成分结点的孩子就结点
            mapback = [ idx for idx ,word in enumerate(con_mapnode) if word[-sub_len: ]!= args.special_token] # 返回单词对应的id，不返回成分结点对应的id
            mapback_con = [idx for idx, word in enumerate(con_mapnode) if word[-sub_len: ] == args.special_token] 
            
            # # **************改进部分开始***********
            con_leaf = self.get_con_node_leaf_dict(args, con_children, con_mapnode, mapback_con)
             # con_leaf_map = [con_leaf[key] for key in con_leaf.keys()]
            con_leaf_token_map = self.get_con_node_leafs_token_map(con_leaf, d['token'])
            con_node_leaf_token_map = [con_leaf_token_map[key] for key in con_leaf_token_map.keys()]

            con_node_adj = self.get_con_node_adj(con_head, mapback_con)

            con_node_len = con_node_adj.shape[0]

            # **************改进部分结束***********

            #layers: 每层包含的结点, influende_range: 每个结点的影响范围, node2layerid: 结点属于成分树中的哪一层
            layers, influence_range, node2layerid = form_layers_and_influence_range(con_path_dict, mapback) # 

            spans = form_spans(layers, influence_range, length, con_mapnode) # 区分哪几个词在每一层是属于同一个成分的

            adj_i_oneshot = head_to_adj_oneshot(dep_head, length, d['aspects']) # 根据依赖关系生成双向的邻接矩阵

            cd_adj = np.ones((length,length))
            if args.con_dep_conditional:
                father = 1
                # 如果根结点的子结点含有两个S(N)，也就是句子中含有多个子句
                if father in con_children and [con_mapnode[node] for node in con_children[father]].count('S[N]') > 1 and con_mapnode[father] == 'S[N]':
                    cd_span = spans[node2layerid[father]+1] # 根结点所在层的下一层，的spans
                    cd_adj = get_conditional_adj(father, length, cd_span, con_children, con_mapnode) # 得到spans对应的成分邻接矩阵

            # ************************************************
            adj_i_oneshot = adj_i_oneshot * cd_adj # 依赖双向邻接矩阵 * 当前spans对应的成分邻接矩阵 = 将依赖树中的子句与子句之间的联系删除之后所得矩阵
            
            # aspect-specific
            bert_sequence_list = []
            bert_segments_ids_list = []
            label_list = []
            aspect_indi_list = []

            select_spans_list = []

            for aspect in d['aspects']: # 方面词列表中的每一个方面词
                asp = list(aspect['term']) # 每个方面词中的term属性对应的value
                asp_bert_ids, _, _ = text2bert_id(asp, tokenizer) # 将方面词转换为bert中的id
                bert_sequence = CLS_id  + text_raw_bert_indices +  SEP_id + asp_bert_ids + SEP_id # 构建bert的输入，每次输入都是一个方面词+句子
                bert_segments_ids = [0] * (bert_length + 2) + [1] * (len(asp_bert_ids ) +1) # （cls + bert句子 + sep）+ （asp + sep）# 用来区分句子输入和方面词输入

                bert_sequence = bert_sequence[:max_len+3] # +3是?
                bert_segments_ids = bert_segments_ids[:max_len+3]

                label = aspect['polarity']

                aspect_indi = [0] * length # 用来定位方面词

                for pidx in range(aspect['from'], aspect['to']):
                    aspect_indi[pidx] = 1
                
                label = pol_vocab.stoi.get(label) # 通过查找极性词典返回label对应的id

                aspect_range = list(range(mapback[aspect['from']], mapback[aspect['to']-1] + 1)) # 方面词在句子中所处的index

                con_lca = find_inner_LCA(con_path_dict, aspect_range) # 找到方面词在成分树形成的列表中所处的位置对应的index

                # ***他是根据方面词在成分树中的祖辈结点所在的层数来选择spans,然后进行抽取
                #select_spans: 成分树中方面词的祖辈结点所在层，的成分跨度列表 span_indications: 祖辈结点对应的结点名称
                select_spans, span_indications = form_aspect_related_spans(con_lca, spans, con_mapnode, node2layerid, con_path_dict) # 

                select_spans = select_func(select_spans, args.max_num_spans, length) # 返回规定的args.max_num_spans个数的spans

                select_spans = [[ x+ 1 for x in span] for span in select_spans] # spans中的每一个元素都加一


                label_list.append(label)
                aspect_indi_list.append(aspect_indi)
                bert_sequence_list.append(bert_sequence)
                bert_segments_ids_list.append(bert_segments_ids)

                select_spans_list.append(select_spans)
            
            # aspect-aspect
            choice_list = [(idx, idx + 1) for idx in range(len(d['aspects']) - 1)] if args.is_filtered else list(
                itertools.combinations(list(range(len(d['aspects']))), 2))


            aa_choice_inner_bert_id_list = []
            
            
            num_aspects = len(d['aspects'])

            cnum = num_aspects  + len(choice_list)
            aa_graph = np.zeros((cnum, cnum))
            cnt = 0

            for aa_info in d['aa_choice']:
                select_ = (aa_info['select_idx'][0], aa_info['select_idx'][1])

                if select_ in choice_list: # choicen
                    first = aa_info['select_idx'][0]
                    second = aa_info['select_idx'][1]

                    word_range = aa_info['word_range']
                    select_words = d['token'][word_range[0]:word_range[-1] + 1] if (word_range[0] <= word_range[-1]) else ['and'] # 

                    aa_raw_bert_ids, _, _ = text2bert_id(select_words, tokenizer)
                    aa_raw_bert_ids = aa_raw_bert_ids[:max_len]

                    if args.aa_graph_version == 1: #directional # 双向的， 构建矩阵
                        if first % 2 == 0:
                            aa_graph[cnt + num_aspects][first] = 1 
                            aa_graph[second][cnt + num_aspects] = 1
                        else:
                            aa_graph[cnt + num_aspects][second] = 1
                            aa_graph[first][cnt + num_aspects] = 1
                    else: # undirectional
                        aa_graph[cnt + num_aspects][first] = 1
                        aa_graph[second][cnt + num_aspects] = 1
                    

                    if args.aa_graph_self: # 是否带自身
                        aa_graph[first][first] = 1 
                        aa_graph[second][second] = 1
                        aa_graph[cnt + num_aspects][cnt + num_aspects] = 1 
                    

                    aa_choice_inner_bert_id_list.append(CLS_id + aa_raw_bert_ids + SEP_id)
                    
                    cnt += 1

            processed += [
                (
                    length, bert_length, word_mapback,
                    adj_i_oneshot, aa_graph,
                    # aspect-specific
                    bert_sequence_list, bert_segments_ids_list, aspect_indi_list, select_spans_list,
                    # con-con 改进部分
                    con_node_leaf_token_map, con_node_adj, con_node_len,
                    # aspect-aspect
                    aa_choice_inner_bert_id_list, 
                    # label
                    label_list
                )
            ]
        
        return processed 
                    

def ABSA_collate_fn(batch):
    batch_size = len(batch) # 3
    batch = list(zip(*batch)) # batch中单个数据中的属性放到一起，这样做的目的就是一次让计算机一次处理多条数据

    lens = batch[0] # 每条数据的长度

    (length_, bert_length_, word_mapback_,
    adj_i_oneshot_, aa_graph_,
    bert_sequence_list_, bert_segments_ids_list_, 
    aspect_indi_list_, select_spans_list_,
    con_node_leaf_token_map_, con_node_adj_, con_node_len_,# 新增两个变量
    aa_choice_inner_bert_id_list_, 
    label_list_) = batch # 一样属性的数据放到了一起

    max_lens = max(lens) # 同一batch中数据长度最长的

    
    length = torch.LongTensor(length_) # 转换为tensor向量 1
    bert_length = torch.LongTensor(bert_length_) # 同上 2 
    word_mapback = get_long_tensor(word_mapback_, batch_size) # 3 x 16 同上， 但是是矩阵了，由于句子的长度不一，所以需padding 3 

    adj_oneshot = np.zeros((batch_size, max_lens, max_lens), dtype=np.float32) # 4 

    for idx in range(batch_size):
        mlen = adj_i_oneshot_[idx].shape[0]
        adj_oneshot[idx,:mlen,:mlen] = adj_i_oneshot_[idx]
    
    adj_oneshot = torch.FloatTensor(adj_oneshot) # 3 x 12 x 12

    # Intra-context 
    map_AS = [[idx] * len(a_i) for idx, a_i in enumerate(bert_sequence_list_)] # 7 用来映射条数据数据中有几组输入，[[0,0,0], [1,1], [2,2,2]]就是第一条有3组输入，即方面词有三个所以有三组输入，同样第二条数据2组...
    map_AS_idx = [range(len(a_i)) for a_i in bert_sequence_list_] #  8 map_AS的index映射

    # *************新增代码开始************
    # 将同一batch中的，con_node_adj成分结点之间0-1邻接矩阵转换为相同大小，并将其转换为tensor形式
    con_node_lens = []
    for l in map_AS:
        for idx in l:
            con_node_lens.append(con_node_len_[idx])
    con_node_lens = torch.LongTensor(con_node_lens)

    max_con_node_lens = max(con_node_len_)
    con_node_adj_batch = np.zeros((batch_size, max_con_node_lens, max_con_node_lens), dtype=np.float32) 

    for idx in range(batch_size):
        mlen = con_node_adj_[idx].shape[0]
        con_node_adj_batch[idx,:mlen,:mlen] = con_node_adj_[idx]
    # 将batch -> 输入形式，每个方面词一个输入序列
    con_node_adj = []
    for l in map_AS:
        for idx in l:
            con_node_adj.append(con_node_adj_batch[idx])
            
    # 求邻接矩阵的度D矩阵
    con_node_adj = np.array(con_node_adj)
    b,c,c = con_node_adj.shape
    d = np.sum(con_node_adj, 2) # 8 x 12 
    d = np.where(d!=0, 1/d, d)
    #d = 1 / d
    #sprt_D = np.sqrt(D)
    for i, l in enumerate(d):
        if i == 0:
            D = np.sqrt(np.diag(l))
        else:
            D = np.append(D, np.sqrt(np.diag(l)))
    D= np.reshape(D, (b, c, c))
    
    D = torch.FloatTensor(D)
   
    con_node_adj = torch.FloatTensor(con_node_adj)
    # *************新增代码结束************

    # *************新增代码开始************
    #将con_node_leaf_token_map : batchsize x max_con_node_len x len(tokens)列表 -> tensor类型，形状不变
    con_node_leaf_token_map_ = list(con_node_leaf_token_map_)
    max_con_node_len = max([len(p) for p in con_node_leaf_token_map_])
    con_node_leaf_token_map_batch = np.zeros((batch_size, max_con_node_len, max_lens), dtype=np.int64)
    for idx in range(batch_size):
        mlen = len(con_node_leaf_token_map_[idx])
        nlen = len(con_node_leaf_token_map_[idx][0])
        con_node_leaf_token_map_batch[idx, :mlen, :nlen] = con_node_leaf_token_map_[idx] 

    con_node_leaf_token_map = []
    for l in map_AS:
        for idx in l:
            con_node_leaf_token_map.append(con_node_leaf_token_map_batch[idx])
            
    con_node_leaf_token_map = torch.LongTensor(con_node_leaf_token_map) 
    # *************新增代码结束************
    # *************新增代码开始************
    # *************新增代码结束************

    # add_pre = np.array([0] + [len(m) for m in map_AS[:-1]]).cumsum()
    
    map_AS = torch.LongTensor([m for m_list in map_AS for m in m_list]) # 合并一个列表中
    map_AS_idx = torch.LongTensor([m for m_list in map_AS_idx for m in m_list]) # 合并一个列表中

    as_batch_size = len(map_AS) # 

    bert_sequence = [p for p_list in bert_sequence_list_ for p in p_list] # 8 x 22 9 
    bert_sequence = get_long_tensor(bert_sequence, as_batch_size)

    bert_segments_ids = [p for p_list in bert_segments_ids_list_ for p in p_list] # 10 
    bert_segments_ids = get_long_tensor(bert_segments_ids, as_batch_size)

    aspect_indi = [p for p_list in aspect_indi_list_ for p in p_list] # 8 x 12 11 
    aspect_indi = get_long_tensor(aspect_indi, as_batch_size)
   
    con_spans_list = [p for p_list in select_spans_list_ for p in p_list] # 这个地方和依赖树矩阵处理的方法差不多
    max_num_spans = max([len(p) for p in con_spans_list])
    con_spans = np.zeros((as_batch_size, max_num_spans, max_lens), dtype=np.int64)  # 12 
    for idx in range(as_batch_size):
        mlen = len(con_spans_list[idx][0])
        con_spans[idx,:,:mlen] = con_spans_list[idx]
    
    con_spans = torch.LongTensor(con_spans)

    # label
    label = torch.LongTensor([sl for sl_list in label_list_ for sl in sl_list if isinstance(sl, int)]) # 17

    # aa_graph 5 
    aspect_num = [len(a_i) for a_i in bert_sequence_list_]
    max_aspect_num = max(aspect_num)

    if (max_aspect_num > 1):
        aa_graph_length = torch.LongTensor([2 * num - 1 for num in aspect_num])  # 每个样例的a-a关系的长度 5, 3, 5    6
        aa_graph = np.zeros((batch_size, 2 * max_aspect_num - 1, 2 * max_aspect_num - 1)) # a-a矩阵(padding) batchsize x max_aa_graph_length x max_aa_graph_length
        
        for idx in range(batch_size):
            cnum = aa_graph_length[idx]
            aa_graph[idx, :cnum, :cnum] = aa_graph_[idx]
        aa_graph = torch.LongTensor(aa_graph)
    else:
        aa_graph_length = torch.LongTensor([])
        aa_graph = torch.LongTensor([])

    aa_choice = [m for m_list in aa_choice_inner_bert_id_list_ for m in m_list] # 所有样本的中间词列表
    aa_batch_size = len(aa_choice) # 中间词总数量

    if aa_batch_size > 0:
        map_AA = [[idx] * len(a_i) for idx, a_i in enumerate(aa_choice_inner_bert_id_list_)] # 13 
        map_AA = torch.LongTensor([m for m_list in map_AA for m in m_list])

        map_AA_idx = torch.LongTensor([m + len(a_i) + 1 for a_i in aa_choice_inner_bert_id_list_ for m in range(len(a_i))]) # 14 


        aa_choice_inner_bert_id = [m for m_list in aa_choice_inner_bert_id_list_ for m in m_list if len(m) > 0] # 15 
        aa_choice_inner_bert_length = torch.LongTensor([len(m) - 2 for m in aa_choice_inner_bert_id]) # 16
        aa_choice_inner_bert_id = get_long_tensor(aa_choice_inner_bert_id, aa_batch_size)


    else:
        map_AA = torch.LongTensor([])
        map_AA_idx = torch.LongTensor([])
       
        aa_choice_inner_bert_id = torch.LongTensor([])
        aa_choice_inner_bert_length = torch.LongTensor([])

    
    return (
        length, bert_length, word_mapback, adj_oneshot,
        aa_graph, aa_graph_length,
        map_AS, map_AS_idx,
        bert_sequence, bert_segments_ids,
        aspect_indi, con_spans,
        map_AA, map_AA_idx,
        aa_choice_inner_bert_id, aa_choice_inner_bert_length,
        con_node_leaf_token_map, con_node_adj, D,
        label
    )

def text2bert_id(token, tokenizer):
    re_token = []
    word_mapback = []
    word_split_len = []
    for idx, word in enumerate(token):
        temp = tokenizer.tokenize(word)
        re_token.extend(temp)
        word_mapback.extend([idx] * len(temp))
        word_split_len.append(len(temp))
    re_id = tokenizer.convert_tokens_to_ids(re_token) # 词在tokenizer中对应的id
    return re_id ,word_mapback, word_split_len

class ABSA_DataLoader(DataLoader):
    def __init__(self, dataset, sort_key, sort_bs_num=None, is_shuffle=True, **kwargs):
        '''
        :param dataset: Dataset object 
        :param sort_idx: sort_function 
        :param sort_bs_num: sort range; default is None(sort for all sequence)
        :param is_shuffle: shuffle chunk , default if True
        :return:
        '''
        assert isinstance(dataset.data, list)
        super().__init__(dataset,**kwargs)
        self.sort_key = sort_key # ？？
        self.sort_bs_num = sort_bs_num
        self.is_shuffle = is_shuffle

    def __iter__(self):
        if self.is_shuffle:
            self.dataset.data = self.block_shuffle(self.dataset.data, self.batch_size, self.sort_bs_num, self.sort_key, self.is_shuffle)

        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            return _MultiProcessingDataLoaderIter(self)

    @staticmethod
    def block_shuffle(data, batch_size, sort_bs_num, sort_key, is_shuffle):
        # sort
        random.shuffle(data)
        data = sorted(data, key = sort_key) # 先按照长度排序
        batch_data = [data[i : i + batch_size] for i in range(0,len(data),batch_size)]
        batch_data = [sorted(batch, key = sort_key) for batch in batch_data]
        if is_shuffle:
            random.shuffle(batch_data)
        batch_data = list(chain(*batch_data))
        return batch_data

def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(0)
    for i, s in enumerate(tokens_list):
        tokens[i, : len(s)] = torch.LongTensor(s)
    return tokens


def get_float_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded FloatTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.FloatTensor(batch_size, token_len).fill_(0)
    for i, s in enumerate(tokens_list):
        tokens[i, : len(s)] = torch.FloatTensor(s)
    return tokens


def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]
