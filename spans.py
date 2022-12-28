import numpy as np

def find_inner_LCA(path_dict,aspect_range):
    path_range = [ [x] + path_dict[x] for x in aspect_range]
    path_range.sort(key=lambda l:len(l))
 
    for idx in range(len(path_range[0])):
        flag = True
        for pid in range(1,len(path_range)):
            if path_range[0][idx]  not in path_range[pid]:
                flag = False #其中一个不在
                break
            
        if flag: #都在
            LCA_node = path_range[0][idx]
            break #already find
    return LCA_node

def get_path_and_children_dict(heads):
    # path_dict词典中：key对应heads中对应元素的index, key对应的value列表中存放的是从该元素所在位置出发，到成分树的根结点所经过的元素（结点）对应的index
    # 所有结点到根结点的路径
    path_dict = {} # 路径词典
    remain_nodes = list(range(len(heads))) # 保留结点index
    delete_nodes = [] # 删除结点index
    
    while len(remain_nodes) > 0: # 直到没有结点
        for idx in remain_nodes: # 保留结点中的每个结点对应的index
            #初始状态
            if idx not in path_dict: # index没在路径词典中，但是head中是有的，也就是说需要将该结点index存储到路径词典中
                path_dict[idx] = [heads[idx]]  # no self # 将index对应的head结点中的index存到路径词典中
                if heads[idx] == -1:
                    delete_nodes.append(idx) #need delete root ，为什么要删除根结点呢？
            else:
                last_node = path_dict[idx][-1] # 为什么要最后一个结点？
                if last_node not in remain_nodes:
                    path_dict[idx].extend(path_dict[last_node])
                    delete_nodes.append(idx)
                else:
                    path_dict[idx].append(heads[last_node])
        #remove nodes
        for del_node in delete_nodes:
            remain_nodes.remove(del_node)
        delete_nodes = []

    #children_dict
    # 所有结点的孩子结点
    children_dict = {}
    for x,l in path_dict.items():
        if l[0] == -1:
            continue
        if l[0] not in children_dict:
            children_dict[l[0]] = [x]
        else:
            children_dict[l[0]].append(x)

    return path_dict, children_dict

def form_layers_and_influence_range(path_dict,mapback): # path_dict : 每个结点到根结点所经过的结点的路径， mapback: 成分树列表中单词对应的index列表
    sorted_path_dict = sorted(path_dict.items(),key=lambda x: len(x[1])) # sorted（）默认降序， key为根据什么来排序
    influence_range = { cid:[idx,idx+1] for idx,cid in enumerate(mapback) } # ？左闭右开，影响范围：成分结点包含的单词序列,[begin_index, end_index)
    layers = {} # 每层的结点
    node2layerid = {} # 结点所在层数，TOP(N)为第一层
    for cid,path_dict in sorted_path_dict[::-1]: 
    
        length = len(path_dict)-1 # 去掉root层
        if length not in layers:
            layers[length] = [cid]
            node2layerid[cid] = length
        else:
            layers[length].append(cid)
            node2layerid[cid] = length
        father_idx = path_dict[0] # cid的父亲结点
        
        
        assert(father_idx not in mapback) # 父亲结点不会是单词结点，只会是成分结点
        if father_idx not in influence_range: # 
            influence_range[father_idx] = influence_range[cid][:] #deep copy
        else:
            influence_range[father_idx][0] = min(influence_range[father_idx][0], influence_range[cid][0])
            influence_range[father_idx][1] = max(influence_range[father_idx][1], influence_range[cid][1])  
    
    layers = sorted(layers.items(),key=lambda x:x[0])
    layers = [(cid,sorted(l)) for cid,l in layers]  # or [(cid,l.sort()) for cid,l in layers]

    return layers, influence_range,node2layerid

# 区分哪几个词在某一层是同一成分的
def form_spans(layers, influence_range, token_len, con_mapnode, special_token = '[N]'):
    spans = []
    sub_len = len(special_token)
    
    for _, nodes in layers: # 每层的结点

        pointer = 0
        add_pre = 0
        temp = [0] * token_len
        temp_indi = ['-'] * token_len
        
        for node_idx in nodes:
            begin,end = influence_range[node_idx] 
            
            if con_mapnode[node_idx][-sub_len:] == special_token: # 如果结点属于成分结点
                temp_indi[begin:end] = [con_mapnode[node_idx][:-sub_len]] * (end-begin) # 将成分结点的影响范围内的单词标记为该成分标签
            
            if(begin != pointer): # 该结点的影响范围开始index不等于pointer
                sub_pre = spans[-1][pointer] 
                temp[pointer:begin] = [x + add_pre-sub_pre for x in spans[-1][pointer:begin]] #
                add_pre = temp[begin-1] + 1
            temp[begin:end] = [add_pre] * (end-begin)  

            add_pre += 1
            pointer = end
        if pointer != token_len: 
            sub_pre = spans[-1][pointer]
            temp[pointer:token_len] = [x + add_pre-sub_pre for x in spans[-1][pointer:token_len]]
            add_pre = temp[begin-1] + 1
        spans.append(temp)

    return spans

def head_to_adj_oneshot(heads, sent_len, aspect_dict, 
                        leaf2root=True, root2leaf=True, self_loop=True):
    """
    Convert a sequence of head indexes into a 0/1 matirx.
    """
    adj_matrix = np.zeros((sent_len, sent_len), dtype=np.float32)

    heads = heads[:sent_len]

    # aspect <self-loop>
    for asp in aspect_dict:
        from_ = asp['from']
        to_ = asp['to']
        for i_idx in range(from_, to_):
            for j_idx in range(from_, to_):
                adj_matrix[i_idx][j_idx] = 1



    for idx, head in enumerate(heads):
        if head != -1:
            if leaf2root:
                adj_matrix[head, idx] = 1
            if root2leaf:
                adj_matrix[idx, head] = 1

        if self_loop:
            adj_matrix[idx, idx] = 1

    return adj_matrix # 根据依赖关系生成双向的邻接矩阵

def get_conditional_adj(father, length, cd_span, 
                        con_children, con_mapnode):
    s_slist = [idx for idx, node in enumerate(con_children[father]) if con_mapnode[node] == 'S[N]' ] # 是S[N]的成分结点对应的index
    st_adj = np.ones((length,length)) # 
    for i in range(len(s_slist)-1):
        idx = s_slist[i]
        begin_idx = cd_span.index(idx)
        end_idx = len(cd_span) - cd_span[::-1].index(idx) # 

        for j in range(idx + 1, len(s_slist)):
            jdx = s_slist[j]
            begin_jdx = cd_span.index(jdx)
            end_jdx = len(cd_span) - cd_span[::-1].index(jdx)
            for w_i in range(begin_idx,end_idx):
                for w_j in range(begin_jdx,end_jdx):
                    st_adj[w_i][w_j] = 0
                    st_adj[w_j][w_i] = 0
    return st_adj


def form_aspect_related_spans(aspect_node_idx, spans, mapnode, node2layerid, path_dict,select_N = ['ROOT','TOP','S','NP','VP'], special_token = '[N]'):
    aspect2root_path = path_dict[aspect_node_idx]
    span_indications = []
    spans_range = []
    
    for idx,f in enumerate(aspect2root_path[:-1]):
        if mapnode[f][:-len(special_token)] in select_N: # 在成分树中，如果方面词的祖辈是select_N中某一个
            span_idx = node2layerid[f] # 祖辈所在成分树中的层数
            span_temp = spans[span_idx] # 该层中的成分跨度标记序列

            if len(spans_range) == 0 or span_temp != spans_range[-1]: # 最后一个0层不计入其中
                spans_range.append(span_temp) # 将祖辈所在层对应的跨度序列存放到spans_range列表中，直到根结点，每个父辈都是这样的操作
                span_indications.append(mapnode[f][:-len(special_token)]) # 将祖辈对应的成分名称存放到span_indications列表中， 直到根结点，每个父辈都是这样的操作
        
    return spans_range, span_indications





def select_func(spans, max_num_spans, length):
    if len(spans) <= max_num_spans: # 如果len(spans) 小于最大，那么补足到最大
        lacd_span = spans[-1] if len(spans) > 0 else [0] * length
        select_spans = spans + [lacd_span] * (max_num_spans - len(spans))

    else:
        if max_num_spans == 1: # 如果为1，那就最近的那个
            select_spans = spans[0] if len(spans) > 0 else [0] * length
        else: # 如果大于最大，有选择的选择成分跨度
            gap = len(spans)  // (max_num_spans-1)
            select_spans = [ spans[gap * i] for i in range(max_num_spans-1)] + [spans[-1]]

    return select_spans