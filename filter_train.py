import networkx

import networkx as nx
from tqdm import tqdm
import collections
import sys
import numpy as np
import gzip
import multiprocessing
import random

def all_simple_paths(G,source,target,cutoff):
    if source not in G:
        raise nx.NodeNotFound('source node %s not in graph' % source)
    if target in G:
        targets = {target}
    else:
        try:
            targets = set(target)
        except TypeError:
            raise nx.NodeNotFound('target node %s not in graph' % target)
    if source in targets:
        return []
    if cutoff is None:
        cutoff = len(G) - 1
    if cutoff < 1:
        return []
    if G.is_multigraph():
        return _all_simple_paths_multigraph(G, source, targets, cutoff)
    else:
        return None

def _all_simple_paths_multigraph(G, source, targets, cutoff):
    visited = collections.OrderedDict.fromkeys([(source,-1)])
    stack = [((v,c) for u, v, c in G.edges(source,keys = True))]

    while stack:
        children = stack[-1]
        child = next(children, None)

        if child is None:
            stack.pop()
            visited.popitem()
        elif len(visited) < cutoff:
            if child in visited:
                continue
            if child[0] in targets:
                yield list(visited) + [child]
            visited[child] = None
            if targets - set(visited.keys()):
                stack.append((v,c) for u, v, c in G.edges(child[0],keys = True))
            else:
                visited.popitem()
        else:  # len(visited) == cutoff:
            for target in targets - set(visited.keys()):
                count = ([child] + list(children)).count(target)
                for i in range(count):
                    yield list(visited) + [target]
            stack.pop()
            visited.popitem()


def construct_original_graph(graph_path):
    head_nodes = list()
    tail_nodes = list()
    edges = list()
    rel_dict = dict()
    # read train data
    with open (graph_path) as fin:
        for line in fin:
            line_list = line.strip('\n').split('\t')
            head_nodes.append(line_list[0])
            tail_nodes.append(line_list[2])
            edges.append((line_list[0],line_list[2],line_list[1]))

    all_nodes = set(head_nodes).union(set(tail_nodes))

    # networkx construct graph
    G = nx.MultiDiGraph()
    # G = nx.DiGraph()
    G.add_nodes_from(all_nodes)
    G.add_edges_from(edges)

    return G

def positive_pairs(relation_path):
    pairs = set()
    with open(relation_path) as fin:
        for line in fin:
            h,t,_ = line.strip("\n").split("\t")
            pairs.add((h,t))
    return pairs

def rule_find(graph_path,relation_path,rules_path):
    G = construct_original_graph(graph_path)
    print(G.size())
    f = open(rules_path,"w")
    pairs = positive_pairs(relation_path)
    rules = set()
    for pair in tqdm(pairs):
        if(G.has_node(pair[0]) and G.has_node(pair[1])):
            paths = list(all_simple_paths(G, pair[0], pair[1], cutoff=4))
            for pp in tqdm(paths):
                rl = []
                for ii in pp[1:]:
                    rl.append(ii[1])
                if(tuple(rl) not in rules):
                    print(rl)
                    rules.add(tuple(rl))
                    f.write("\t".join(rl))
                    f.write("\n")
                    f.flush()
    f.close()


def rules_read(rules_path):
    rules = set()
    with open(rules_path,"r") as f:
        fin = f.readlines()
        for line in fin:
            rules.add(tuple(line.strip("\n").split("\t")))

    return rules

def sorted_train(relationPath):
    train_pairs = set()
    with open(relationPath,"r") as f:
        fin = f.readlines()
        for line in fin:
            h,t = line.strip("\n").split(":")[0].strip().split(",")
            h = h.split("$")[1]
            t = t.split("$")[1]
            lbl = line.strip("\n").split(":")[1].strip()
            train_pairs.add((h,t,lbl))
    train_sorted = sorted(train_pairs,key = lambda t :(t[0],t[1]))
    with open(relationPath+"_sorted","w") as f:
        for pair in train_sorted:
            f.write("\t".join(list(pair)))
            f.write("\n")
    return train_sorted

def read_pairs(relationPath):
    train_pairs = list()
    with open(relationPath,"r") as f:
        fin = f.readlines()
        for line in fin:
            h,t = line.strip("\n").split(":")[0].strip().split(",")
            h = h.split("$")[1]
            t = t.split("$")[1]
            h = "/m/"+  h[2:]
            t = "/m/" + t[2:]
            lbl = line.strip("\n").split(":")[1].strip()
            train_pairs.append((h,t,lbl))
    return train_pairs

def sort_train_thing(relationPath):
    train_pairs = set()
    with open(relationPath,"r") as f:
        fin = f.readlines()
        for line in fin:
            h,t = line.strip("\n").split(":")[0].strip().split(",")
            # h = h.split("$")[1]
            # t = t.split("$")[1]
            lbl = line.strip("\n").split(":")[1].strip()
            kk = h.replace("_","~")+"~"+t.replace("_","~")
            train_pairs.add((kk,h,t,lbl))
    train_sorted = sorted(train_pairs,key = lambda t :t[0])
    with open(relationPath+"_sorted_thing","w") as f:
        for pair in train_sorted:
            f.write(pair[1]+","+pair[2]+": "+pair[3])
            f.write("\n")
    return train_sorted

def extract_pairs_thing(relationPath):
    train_pairs = set()
    with open(relationPath,"r") as f:
        fin = f.readlines()
        for line in fin:
            h,t = line.strip("\n").split(":")[0].strip().split(",")
            # h = h.split("$")[1]
            # t = t.split("$")[1]
            lbl = line.strip("\n").split(":")[1].strip()
            train_pairs.add((h,t,lbl))
    train_sorted = sorted(train_pairs,key = lambda t :(t[0],t[1]))
    with open(relationPath+"_sorted_thin","w") as f:
        for pair in train_sorted:
            f.write(pair[0]+","+pair[1]+": "+pair[2])
            f.write("\n")
    return train_sorted

def path_train_data(relationPath,graphpath,rulesPath):
    rules = rules_read(rulesPath)
    train_pairs = read_pairs(relationPath)
    sort_train = sorted(train_pairs,key =lambda t:(t[0],t[2]),reverse=False)
    out_train = []
    ii =0
    while ii < len(sort_train):
        if sort_train[ii][2] =="+":
            for idx in range(4):
                if(ii+idx<len(sort_train)):
                    out_train.append(sort_train[ii+idx])
            ii += 4
        ii+=1

    G = construct_original_graph(graphpath)
    fout = open(relationPath+"_path_filtered_train_noinv","w")
    for pair in tqdm(out_train):
        h,t,lbl = pair
        if(G.has_node(h) and G.has_node(t)):
            paths = list(all_simple_paths(G, pair[0], pair[1], cutoff=4))
            paths_ = list()
            if(lbl =="+"):
                for pp in tqdm(paths):
                    rl = []
                    for ii in pp[1:]:
                        print(pp)
                        rl.append(ii[1])
                    paths_.append("@".join(rl))
                fout.write((str(1) +"&"+" ".join(paths_)+"&"+h+"&"+t+"\n"))
                fout.flush()
            else:
                for pp in tqdm(paths):
                    rl = []
                    for ii in pp[1:]:
                        rl.append(ii[1])
                    if(tuple(rl) in rules):
                        paths_.append("@".join(rl))
                fout.write((str(0) +"&"+" ".join(paths_)+"&"+h+"&"+t+"\n"))
                fout.flush()
    fout.close()


def path_test_data(relationPath,graphpath,rulesPath):
    rules = rules_read(rulesPath)
    train_pairs = read_pairs(relationPath)

    G = construct_original_graph(graphpath)
    fout = open(relationPath+"_path_filtered_noinv","w")
    for pair in tqdm(train_pairs):
        h,t,lbl = pair
        if(G.has_node(h) and G.has_node(t)):
            paths = list(all_simple_paths(G, pair[0], pair[1], cutoff=4))
            paths_ = list()
            if(lbl =="+"):
                for pp in tqdm(paths):
                    rl = []
                    for ii in pp[1:]:
                        rl.append(ii[1])
                    paths_.append("@".join(rl))
                fout.write((str(1) +"&"+" ".join(paths_)+"&"+h+"&"+t+"\n"))
                fout.flush()
            else:
                for pp in tqdm(paths):
                    rl = []
                    for ii in pp[1:]:
                        rl.append(ii[1])
                    if(tuple(rl) in rules):
                        paths_.append("@".join(rl))
                fout.write((str(0) +"&"+" ".join(paths_)+"&"+h+"&"+t+"\n"))
                fout.flush()
        else:
            if(lbl == "+"):
                fout.write((str(1) + "&" + " ".join([]) + "&" + h + "&" + t + "\n"))
                fout.flush()
            else:
                fout.write((str(0) + "&" + " ".join([]) + "&" + h + "&" + t + "\n"))
                fout.flush()

    fout.close()


def sort_deep(relationPath):
    train_pairs = list()
    with open(relationPath,"r") as fdeep:
        fin = fdeep.readlines()
        for line in fin:
            h,t = line.strip("\n").split(":")[0].strip().split(",")
            h = h.split("$")[1]
            t = t.split("$")[1]
            lbl = line.strip("\n").split(":")[1].strip()
            train_pairs.append((h,t,lbl))
    train_read = dict()
    with  open(relationPath + "_path_filtered", "r") as ffilter:
        fin = ffilter.readlines()
        for line in fin:
            h = line.strip("\n").split("\t")[2]
            t = line.strip("\n").split("\t")[3]
            print(h,t)
            train_read[(h,t)] = line

    with open(relationPath + "_path_filtered_test_deep", "w") as fout:
        for key in train_pairs:
            fout.write(str("&".join(train_read[(key[0],key[1])]).strip("\n").split("\t")))


def rewriteFile(relationPath):
    fout = open(relationPath + "_path_filtered_train_deep", "w")
    with open(relationPath+"_path_filtered_train","r") as f:
        ff = f.readlines()
        for line in ff:
            print(line.split("\t"))
            fout.write("&".join(line.split("\t")))
    fout.close()

def rulesEmbedding(dataPath,relation):
    embedding_rules = []
    headpath2id = open(dataPath+"headpath2id_nohead@"+relation+".txt","w")
    headpath2vec = open(dataPath+"embedding_paths_new_train_nohead@"+relation+".npy", "wb")
    with open(dataPath+"rules.txt", "r")as f:
        lines = f.readlines()
        for line in lines:
            embedding_rules.append(line.strip().split("\t"))

    relation_id_dict = {}
    with open(dataPath +'relation2id.txt', "r") as f1:
        for line in f1.readlines():
            rel, idx = line.split("\t")
            relation_id_dict[rel] = idx

    rel_vec = np.loadtxt(dataPath + 'relation2vec.unif')

    count = 0
    headpath2id.write("@".join("") + "&" + str(count)+"\n")
    count = 1
    embedding_res = []
    x_0 = rel_vec[0]
    for ii in range(len(x_0)):
        x_0[ii]=0
    embedding_res.append(x_0)
    for head_rule in tqdm(embedding_rules):
        headpath2id.write("@".join(head_rule) + "&" + str(count)+"\n")
        count += 1
        x_0 = rel_vec[0]
        for ii in range(len(x_0)):
            x_0[ii] = 0

        for jj in range(0, len(head_rule)):
            #             x = y[relation_id_dict[jj]]
            idx = relation_id_dict[head_rule[jj]]
            x_0 = np.add(rel_vec[int(idx)], x_0)
        embedding_res.append(x_0)

    embedding_matrix = np.array(embedding_res).astype(np.float32)

    print(embedding_matrix.shape)
    np.save(headpath2vec, embedding_matrix, allow_pickle=True)


def oneEmbedding(dataPath,relation):
    embedding_rules = []
    headpath2vec = open("onehot.npy", "wb")
    with open(dataPath+"rules.txt", "r")as f:
        lines = f.readlines()
        for line in lines:
            embedding_rules.append(line.strip().split("\t"))

    relation_id_dict = {}
    with open(dataPath +'relation2id.txt', "r") as f1:
        for line in f1.readlines():
            rel, idx = line.split("\t")
            relation_id_dict[rel] = idx

    rel_vec = np.loadtxt(dataPath + 'relation2vec.unif')

    count = 0

    count = 1
    embedding_res = []
    x_0 = rel_vec[0]
    for ii in range(len(x_0)):
        x_0[ii]=0
    embedding_res.append(x_0)

    x_1 = rel_vec[1]
    for ii in range(len(x_1)):
        x_1[ii]=1
    embedding_res.append(x_1)

    embedding_matrix = np.array(embedding_res).astype(np.float32)

    print(type(embedding_matrix))
    np.save(headpath2vec, embedding_matrix, allow_pickle=True)



def deeppath_data(relationPath,graphpath,relation):
    train_pairs = list()
    with open(relationPath+"sort_test.pairs", "r") as fdeep:
        fin = fdeep.readlines()
        for line in fin:
            lbl = line.strip("\n").split(":")[-1].strip()
            h, t = line.strip("\n").split(":")[0].strip().split(",")
            h = h.split("$")[1]
            t = t.split("$")[1]
            train_pairs.append((h, t,lbl))

    G = construct_original_graph(graphpath)
    count = 0
    count_ = 1
    count__ = 0
    for pair in tqdm(train_pairs):
        h,t,lbl = pair
        if(not G.has_node(h) and not G.has_node(t)):
            count__+=1
        elif G.has_node(h) and not G.has_node(t):
            count += 1
        elif not G.has_node(h)  and not G.has_node(t):
            count_+=1
    print(relation,  len(train_pairs),count__)

    print ("h in Graph, t not in Graph {} \nh not in Graph, t not in Graph {} ".format(count/len(train_pairs),count_/len(train_pairs)))
    print("h in Graph, t not in Graph {} \nh not in Graph, t not in Graph {} ".format(count,
                                                                                      count_) )


def length(relationPath,relation):
    leng =0
    count = 0
    with gzip.open(relationPath,"r") as f:
        for idx, line in enumerate(f):
            lbl,txt,head,tail = line.decode('utf-8').strip("\n").split("&")
            txt = set(txt.split(" "))
            if(len(txt) ==1):
                if(list(txt)[0] ==""):
                    count+=1
            leng +=len(txt)
        print(relation,count/(idx+1),leng/(idx+1))

def multiple_filter(G,rules,pair,outpath):
    fout = open(outpath, "a+")
    h, t, lbl = pair
    if (G.has_node(h) and G.has_node(t)):
        paths = list(all_simple_paths(G, pair[0], pair[1], cutoff=4))
        paths_ = list()
        if (lbl == "+"):
            for pp in tqdm(paths):
                rl = []
                for ii in pp[1:]:
                    rl.append(ii[1])
                if (tuple(rl) in rules):
                    paths_.append("@".join(rl))
                else:
                    print("error!! positive sample's path not include ruleset")
            fout.write((str(1) + "&" + " ".join(paths_) + "&" + h + "&" + t + "\n"))
            fout.flush()
        else:
            for pp in tqdm(paths):
                rl = []
                for ii in pp[1:]:
                    rl.append(ii[1])
                if (tuple(rl) in rules):
                    paths_.append("@".join(rl))
            fout.write((str(0) + "&" + " ".join(paths_) + "&" + h + "&" + t + "\n"))
            fout.flush()
    else:
        if (lbl == "+"):
            fout.write((str(1) + "&" + " ".join([]) + "&" + h + "&" + t + "\n"))
            fout.flush()
        else:
            fout.write((str(0) + "&" + " ".join([]) + "&" + h + "&" + t + "\n"))
            fout.flush()

def reorderTest(relationPath,relation):
    train_pairs = list()
    with open(relationPath+"sort_test.pairs", "r") as fdeep:
        fin = fdeep.readlines()
        for line in fin:
            lbl = line.strip("\n").split(":")[-1].strip()
            h, t = line.strip("\n").split(":")[0].strip().split(",")
            h = "/m/"+h.split("$")[1][2:]
            t = "/m/"+t.split("$")[1][2:]
            train_pairs.append((h, t,lbl))
    train_read = dict()
    with  open(relationPath + "sort_test.pairs_filtered", "r") as ffilter:
        fin = ffilter.readlines()
        for line in fin:
            lbl,txt,h,t = line.strip("\n").split("&")
            h ="/m/"+ h.replace("/","_")[3:]
            t ="/m/"+ t.replace("/","_")[3:]
            train_read[(h, t)] = [lbl,txt,h,t]

    with open(relationPath+"testing_data_path_train_nohead@"+relation+".txt","w") as fout:
        for key in train_pairs:
            if((key[0], key[1]) in train_read):
                if key[2] =="+":
                    fout.write((str(1) + "&" + train_read[(key[0], key[1])][1] + "&" + key[0] + "&" + key[1] + "\n"))
                    fout.flush()
                else:
                    fout.write((str(0) + "&" + train_read[(key[0], key[1])][1]+ "&" + key[0] + "&" + key[1] + "\n"))
                    fout.flush()
            else:
                if key[2] =="+":
                    fout.write((str(1) + "&" + " ".join([]) + "&" + key[0] + "&" + key[1] + "\n"))
                    fout.flush()
                else:
                    fout.write((str(0) + "&" + " ".join([]) + "&" + key[0] + "&" + key[1] + "\n"))
                    fout.flush()


def reorderTrain(relationPath,relation,rulePath):
    train_pairs = list()
    with open(relationPath+"train.pairs", "r") as fdeep:
        fin = fdeep.readlines()
        for line in fin:
            lbl = line.strip("\n").split(":")[-1].strip()
            h, t = line.strip("\n").split(":")[0].strip().split(",")
            h = "/m/"+h.split("$")[1][2:]
            t = "/m/"+t.split("$")[1][2:]
            train_pairs.append((h, t,lbl))
    train_read = dict()
    with  open(relationPath + "train.pairs_filtered", "r") as ffilter:
        fin = ffilter.readlines()
        for line in fin:
            lbl,txt,h,t = line.strip("\n").split("&")
            h ="/m/"+ h.replace("/","_")[3:]
            t = "/m/"+ t.replace("/","_")[3:]
            train_read[(h, t)] = [lbl,txt,h,t]

    with open(relationPath+"training_data_path_train_nohead@"+relation+".txt","w") as fout:
        for key in train_read:
            if((key[0], key[1],"+") in train_pairs or (key[0], key[1],"-") in train_pairs):
                fout.write("&".join(train_read[key])+"\n")
                fout.flush()






if __name__ =="__main__":
    #dataPath = "./NELL-995/"
    relation = sys.argv[1]
    #graphpath = dataPath + 'tasks/' + relation + '/' + 'graph.txt'
    #relationPath = dataPath + 'tasks/' + relation + '/' + 'sort_test.pairs'
    #relationPathTest = dataPath + 'tasks/' + relation + '/' + 'test.pairs'
    #rulesPath = dataPath + 'tasks/' + relation + '/' + 'rules.txt'
    #sort_train_thing(relationPath)
    #path_train_data(relationPath,graphpath,rulesPath)

    dataPath = "./FB15k-237/"
    # relations = [
    #    # "sports@sports_team@sport",
    #   #  "people@person@place_of_birth",
    #   #  "people@person@nationality"
    #    "film@film@language"
    #   #  "film@director@film",
    #   #  "film@director@film",
    #   #  "film@film@written_by",
    #   #  "tv@tv_program@languages",
    #   #  "location@capital_of_administrative_division@capital_of.@location@administrative_division_capital_relationship@administrative_division" \
    #   #  "organization@organization_founder@organizations_founded",
    #    # "music@artist@origin"
    # ]

    # for relation in relations:
    # relation  ="tv@tv_program@languages"


    graphpath = dataPath + 'tasks/' + relation + '/' + 'graph.txt'
    relationPath = dataPath + 'tasks/' + relation+"/train.pairs"
    rulesPath = dataPath + 'tasks/' + relation + '/' + 'rules_inv.txt'
   # path_train_data(relationPath, graphpath, rulesPath)
    outpath  = relationPath+"__filtered_with_inv"
    #oneEmbedding(dataPath+ 'tasks/' + relation +'/',relation)
    G = construct_original_graph(graphpath)
    rules = rules_read(rulesPath)
    train_pairs = read_pairs(relationPath)
    print(len(train_pairs))
    sort_train = sorted(train_pairs,key =lambda t:(t[0],t[2]),reverse=False)
    out_train = []
    ii =0
    while ii < len(sort_train):
        if sort_train[ii][2] =="+":
            for idx in range(4):
                if(ii+idx<len(sort_train)):
                    out_train.append(sort_train[ii+idx])
            ii += 4
        ii+=1


    tasks = []
    for item in tqdm(out_train):
        tasks.append((G, rules, tuple(item), outpath))

    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_cores - 1)

    inputs = tqdm(tasks)

    processed_list = pool.starmap(multiple_filter, inputs)

        # deeppath_data(relationPath,graphpath,relation)
        # relationPath = "nohead_nell/testing_data_path_train_nohead"+relation+".txt.gz"
        # length(relationPath,relation)

    #path_train_data(relationPath,graphpath,rulesPath)

    #rewriteFile(relationPath)
   # rulesEmbedding(dataPath+ 'tasks/' + relation +'/',relation)
    #reorderTrain(relationPath,relation)
    #reorderTest(relationPath,relation)


    #sort_deep(relationPath)













