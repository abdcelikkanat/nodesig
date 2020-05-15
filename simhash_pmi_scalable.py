################################
#
################################


import hashlib
import numbers
import collections
from zlib import crc32
import time
import os
import sys
import numpy as np
import csv
from scipy.stats import ortho_group

class SimHashPMI:

    g = None
    dim = None
    w = None
    chunk_size = None
    dtype = None
    verbose = None
    temp_weight_file = "./hash_weights.txt"
    emb_file = "./temp_embedding.txt"

    def __init__(self, input_dim, output_dim, chunk_size=64, dtype=np.float, verbose=False, proj_met="random"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.w = None
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.verbose = verbose
        self.proj_met = proj_met
        self._generate_projection_matrix()

    def _generate_projection_matrix(self):

        if self.verbose:
            weight_size = self.input_dim * self.output_dim * np.dtype(self.dtype).itemsize / 8 / 1024. **3
            print("The size of projection matrix: {:.3f} Gb".format( weight_size) )

        if self.chunk_size == self.output_dim:

            if self.proj_met  == "random":
                self.w = np.random.normal(loc=0.0, scale=1.0, size=(self.input_dim, self.output_dim))
            elif self.proj_met == "orthogonal":
                print("Orthogonal matrix!")
                self.w = ortho_group.rvs(self.input_dim)
                self.w = self.w[:, :self.output_dim]
            else:
                raise ValueError("Invalid method")

        else:

            if os.path.exists(self.temp_weight_file):
                os.remove(self.temp_weight_file)

            num_of_chunks = int(self.output_dim / self.chunk_size)
            with open(self.temp_weight_file, 'a') as f:
                csvwriter = csv.writer(f, delimiter=' ')
                for c in range(num_of_chunks):
                    w = np.random.normal(loc=0.0, scale=1.0, size=(self.input_dim, self.chunk_size))
                    csvwriter.writerows(w.T)
            f.close()
            self.w = None

    def encode(self, x):

        if self.w is not None:

            emb = np.dot(self.w, x)

        else:

            emb = np.zeros(shape=(self.output_dim, ), dtype=self.dtype)
            with open(self.temp_weight_file, 'r') as f:
                csvreader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)

                for d, row in enumerate(csvreader.readrows()):
                    if np.dot(row, x) > 0:
                        emb[d] = 1
                    else:
                        emb[d] = -1

        return emb

    def encodeAll(self, X, emb_file=None):

        init_time = time.time()

        if emb_file is not None:
            self.emb_file = emb_file
        if os.path.exists(self.emb_file):
            os.remove(self.emb_file)

        if self.w is not None:

            emb = X.dot(self.w)
            emb = np.sign(emb).astype(np.int8)

            with open(self.emb_file, 'a') as fw:
                writer = csv.writer(fw, delimiter=' ')
                writer.writerows(emb.T)
            fw.close()

        else:

            num_of_chunks = int(self.output_dim / self.chunk_size)

            with open(self.temp_weight_file, 'r') as f:
                csvreader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)

                for chunk_id in range(num_of_chunks):
                    print("{}/{}".format(chunk_id+1, num_of_chunks))
                    w = np.zeros(shape=(self.input_dim, self.chunk_size), dtype=self.dtype)
                    for c in range(self.chunk_size):
                        w[:, c] = next(csvreader)
                        #print(row)
                        #w[:, c] = row

                    emb = X.dot(w)
                    emb = np.sign(emb).astype(np.int8)

                    with open(self.emb_file, 'a') as fw:
                        writer = csv.writer(fw, delimiter=' ')
                        for c in range(self.chunk_size):
                            writer.writerow(emb[:,c])
                    fw.close()

        if self.verbose:
            print("All embeddings have been computed in {} secs.".format(time.time() - init_time))

    def transpose_embeddings(self, embedding_file, format='npy'):

        emb = np.zeros(shape=(self.input_dim, self.output_dim), dtype=np.int8)
        with open(self.emb_file, 'r') as f:

            for d in range(self.output_dim):
                line = f.readline()
                for idx, val in enumerate(line.strip().split(' ')):
                    if val == '1':
                        emb[idx, d] = 1
        f.close()

        if format == "word2vec":
            with open(embedding_file, 'w') as f:
                csvwriter = csv.writer(f, delimiter=' ')
                csvwriter.writerow([emb.shape[0], emb.shape[1]])
                for i in range(emb.shape[0]):
                    csvwriter.writerow([i] + emb[i, :].tolist() )
            f.close()

        elif format == "npy":

            np.save(embedding_file, emb)

        else:

            raise ValueError("Invalid file format!")
