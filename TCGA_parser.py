import pickle
import os
import pandas as pd
import numpy as np


class TCGA_parser:

    def __init__(self, pickles_dir="", verbose=True, cTypes=None):
        self.pickles_dir = pickles_dir  # dir with all the pickles
        self.verbose = verbose
        self.cTypes = cTypes            # cancer type
        self.translator = {}            # translate label to cancer type
        # self.load_pickles2df()

    def load_pickles2df(self):
        """ load all the pickle tables in the directory to a single DataFrame """

        # load tables and merge all data to a dictionary, res_dic
        res_dic = {}
        for cancer in self.cTypes:
            pk = self.load_pickle(os.path.join(self.pickles_dir, cancer + '.pickle'))
            res_dic.update(pk)

        # make a large DataFrame from all the tables, df
        df = pd.DataFrame()
        for cancer in self.cTypes:
            df = pd.concat([df, res_dic[cancer]])
            if self.verbose:
                l = np.abs(int(res_dic[cancer]['label'][0]))
                print("{} label={}, examples: {}".format(cancer, l, res_dic[cancer].shape[0]))
                self.translator[l] = cancer
                self.translator[-l] = cancer + " Normal"

        # clean the table and return
        df = df.reset_index(drop=True)
        df = df.dropna(axis=1)
        return df

    def load_pickle(self, pickle_path):
        """ Load a pickle from 'pickle_path' and return it """
        if not os.path.exists(pickle_path):
            print("No such file or directory:\n", pickle_path)
            return
        if self.verbose:
            print("loading '{}'...".format(pickle_path[pickle_path.rfind('/') + 1:]))
        with open(pickle_path, 'rb') as handle:
            return pickle.load(handle)