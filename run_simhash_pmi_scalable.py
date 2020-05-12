import networkx as nx
from simhash_pmi_scalable import *
#from preprocessing2 import *
from preprocessing import *
import time
import csv
import codecs
import os


def save_emb(N, dim, temp_emb_file, outputname):
    print("Emb 1")
    emb = np.zeros(shape=(N, dim), dtype=np.int )
    with open(temp_emb_file, 'r') as fin:
        csvreader = csv.reader(fin, delimiter=',')
        #csvreader = csv.reader(codecs.iterdecode(csvfile, 'utf-8'), delimiter=' ')
        #emb = np.loadtxt(fin, delimiter=' ').T
        #print(emb)
        d = 0
        for row in csvreader:
            #print(row)
            emb[:, d] = [1 if r == '1' else 0  for r in row]
            d += 1
    print("Emb2")

    with open(outputname, 'w') as f:
        f.write("{} {}\n".format(N, dim))
        for node in range(N):
            line = "{} {}\n".format(str(node), " ".join(str(v) for v in emb[node, :]))
            f.write(line)


def get_random_walks(g, walk_len=5, cont_prob=0.98, scale=True):
    A = load_network(g)
    A = RWR(A, walk_len, cont_prob, scale)

    return A


def get_pmi_matrix(g, walk_len=5, cont_prob=0.98, scale=True):
    A = load_network(g)
    A = RWR(A, walk_len, cont_prob, scale)
    A = PPMI_matrix(A)

    return A

dim = 8192
chunk_size=dim #8
walk_len = 5
cont_prob =0.98
scale=True
filename = "Homo_sapiens_renaissance" #"youtube_renaissance" #"wiki_new" #"Homo_sapiens_new"
matrix_path="./matrix_{}_walklen={}_prob={}_scale2={}".format(filename, str(walk_len), str(cont_prob), str(scale))
#graph_path="../NodeSketch/graphs/{}.gml".format(filename)
graph_path="../datasets/{}.gml".format(filename)
#output_path="./embeddings/renaissance_pmi_{}_walklen={}_prob={}_scale={}.embedding".format(filename, str(walk_len), str(cont_prob), str(scale))
output_path="./testuff_renaissance_rw_{}_walklen={}_prob={}_scale2={}.embedding".format(filename, str(walk_len), str(cont_prob), str(scale))
g = nx.read_gml(graph_path)
number_of_nodes = g.number_of_nodes()
print("Num of nodes: {}".format(number_of_nodes))
######################
init_time = time.time()
if os.path.exists(matrix_path):
    A = np.load(matrix_path)
    print("Matrix loaded: {}".format(time.time() - init_time))
else:
    A = get_random_walks(g, walk_len, cont_prob, scale)  # get_pmi_matrix(g, walk_len, cont_prob, scale)
    np.save(matrix_path, A)
    print("Matrix computed: {}".format(time.time() - init_time))
del g

######################



temp_emb_file="./temp_emb_{}.txt".format(filename)

srp = SimHashPMI(input_dim=number_of_nodes, output_dim=dim, chunk_size=chunk_size, dtype=np.float, verbose=True)
srp.encodeAll(X=A, emb_file=temp_emb_file)
srp.transpose_embeddings(embedding_file=output_path, format="word2vec")



