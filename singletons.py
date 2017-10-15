#!/usr/bin/python3

# from learn_trees import Cpg_trees
from multi_class_learn import CpG_DTrees
from TCGA_parser import TCGA_parser
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from Df_manager import Df_manager
from sklearn.tree import DecisionTreeClassifier
import matplotlib.mlab as mlab
import matplotlib
from multi_class_learn import ctypes
matplotlib.style.use('ggplot')

__author__ = 'nloyfer'


singles_dir = "/cs/usr/nloyfer/netanel/TCGA/pickle/singles/"


def gen_table(ctype):
    """ find the accuracy of every feature, for a single cancer type.
        generate a table of results of shape (nr_features, 2), and dump it to pickle """

    if ctype not in ctypes:
        print("invalid cancer type")
        return

    print("Generating singletons table for cancer ", ctype)
    c = CpG_DTrees(cTypes=[ctype],
                   test_rate=.2,
                   verbose=True,
                   depth=1)

    features = list(c.df.columns[:-1])
    M = len(features)

    # init results table:
    res_arr = pd.DataFrame(features)
    res_arr = res_arr.assign(score=pd.Series(np.zeros(M)))
    res_arr.columns.values[0] = 'name'

    labels = c.df['label'].unique()
    print("labels:", labels)
    classes = [c.df[c.df['label'] == l] for l in labels]
    exit(0)
    for j in range(3):
        df_train, df_test = Df_manager(c.df, {}, c.test_rate, 0, 0)._split_train_test(classes)
        print("shapes:")
        print(df_train.shape, df_test.shape)

        for i in range(M):
            if i % ((M) // 20) == 0:
                print("[{} / {}]".format(i, M))

            train_i = pd.concat([df_train[features[i]], df_train['label']], axis=1)
            test_i = pd.concat([df_test[features[i]], df_test['label']], axis=1)
            dt = DecisionTreeClassifier(min_samples_split=20, random_state=c.rand_state, max_depth=1)
            dt.fit(train_i[features[i]].reshape(-1, 1), train_i['label'].reshape(-1, 1))
            score = dt.score(test_i[features[i]].reshape(-1, 1), test_i['label'].reshape(-1, 1))

            # cur_score.append(self.learn_tree())
            res_arr.set_value(i, 'score', res_arr['score'][i] + score / 3.0)

    # KIRP
    # cg24865495
    #
    # KIRC
    # cg14695378


    # dump results:
    pickle_dest = os.path.join(singles_dir, "{}_singles.pickle".format(c.cTypes))
    print("dumping singletons pickle at:\n", pickle_dest)
    with open(pickle_dest, 'wb') as handle:
        pickle.dump(res_arr, handle)


def load_table(ctype):
    return TCGA_parser().load_pickle(os.path.join(singles_dir, ctype + "_singles.pickle"))
    # return TCGA_parser().load_pickle(os.path.join(singles_dir, "singletons.pickle"))


def analize_singles(ctype):
    res_arr = load_table(ctype)[:-1]
    mean = np.mean(res_arr['score'])
    std = np.std(res_arr['score'])
    print(res_arr.shape)
    print(mean, std)

    # res_arr.plot.hist()
    # pd.DataFrame.hist(res_arr, column='score', bins=120, weights=np.ones_like(res_arr[res_arr.columns[0]]) * 100. / len(res_arr))
    pd.DataFrame.hist(res_arr, column='score', bins=160, normed=True)
    # res_arr.plot.hist()
    # print(res_arr.ix[res_arr['score'].idxmax()])
    print(res_arr.ix[res_arr['score'].idxmin()])
    x = np.linspace(0.8, 1, 500)
    plt.plot(x, mlab.normpdf(x, mean, std))
    plt.show()

gen_table('BRCA')
# gen_table('BLCA')
# analize_singles('BLCA')
# analize_singles('BRCA')