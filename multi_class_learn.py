
from Df_manager import Df_manager
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import numpy as np
import subprocess
import os
from TCGA_parser import TCGA_parser
import matplotlib.pyplot as plt
import matplotlib.image as img

pick_dir = "/cs/usr/nloyfer/netanel/TCGA/pickle/float16/"
fig_dir = "/cs/usr/nloyfer/netanel/TCGA/figures/trees/"

ctypes = ['BLCA', 'BRCA', 'COAD', 'ESCA', 'HNSC', 'KIRC', 'KIRP', 'LC', 'LIHC', 'LUAD', 'LUSC', 'PRAD', 'THCA', 'UCEC']

class CpG_DTrees:

    def __init__(self, cTypes, reps=1, test_rate=.2, rand_state=99, verbose=True,
                 depth=1, vis=False, sites=None, display=False):
        self.test_rate = test_rate          # test rate
        self.rand_state = rand_state        # random state
        self.verbose = verbose              # verbose
        self.depth = depth                  # max depth of the decision tree
        self.vis = vis                      # generate tree visualization iff vis==True
        self.display = display              # display tree visualization iff display==True
        self.reps = reps                    # repeat training process 'reps' times
        self.sites = sites                  # filter CpG sites to list in file, or a single site
        self.cTypes = list(set(cTypes))     # list of cancer type/s. remove duplicates.
        self.df_man = None                  # DF manager

        # init a parser, and load pickles to DataFrame
        self.parser = TCGA_parser(pick_dir, self.verbose, self.cTypes)
        self.df = self.parser.load_pickles2df()

        # prints
        if self.verbose:
            print("\n === learning decision trees === ")
            print("\tcancer type\s: {}\n".format(self.cTypes))

    def _gen_fig_path(self, score):
        """ generate informative file name. for example: dt_BRCA_LUAD_LIHC_0.927 """
        cancer = "_".join(self.cTypes) if len(self.cTypes) > 1 else self.cTypes[0]
        fig_name = "dt_" + cancer + "_%0.3f" % score
        full_path = fig_dir + fig_name + ".png"
        return full_path

    def visualize_tree(self, tree, feature_names, score):
        """Create tree png using graphviz. display it if requested"""
        classes_names = [self.parser.translator[l] for l in sorted(self.df_man.labels)]

        # generate file name and path
        pic_path = self._gen_fig_path(score)            # output figure path (.png)
        dot_path = pic_path.replace(".png", ".dot")     # temporary file path (.dot)

        # generate and save tree figure
        with open(dot_path, 'w') as f:
            export_graphviz(tree,
                            out_file=f,
                            feature_names=feature_names,
                            class_names=classes_names,
                            leaves_parallel=True,
                            rounded=True,
                            impurity=False,
                            # label='root',
                            filled=True)

        subprocess.check_call(["dot", "-Tpng", dot_path, "-o", pic_path])

        # remove the temporary dot file:
        os.remove(dot_path)

        # display tree figure
        if self.display:
            plt.imshow(img.imread(pic_path))
            plt.title("Decision tree for cancer types: {}\nScore: %0.3f".format(self.cTypes) % score)
            plt.axis('off')
            plt.show()
            # subprocess.check_call(["xdg-open", pic_path])

    def learn_tree(self):

        # init a DataFrame Manager:
        self.df_man = Df_manager(self.df, self.test_rate, self.parser.translator, self.verbose)

        # filter sites in the DF:
        if self.sites is not None:
            self.df_man.pick_sites(self.sites)

        # load and shuffle data from DF to datasets:
        datasets = {}
        self.df_man.df2datasets(datasets)

        # init tree model
        dt = DecisionTreeClassifier(min_samples_split=20, random_state=self.rand_state, max_depth=self.depth)

        # train, test, and score:
        print("training...", end="\t") if self.verbose else None
        dt.fit(datasets['x_train'], datasets['y_train'])
        print("testing...", end="\t") if self.verbose else None
        score = dt.score(datasets['x_test'], datasets['y_test'])
        print("score: ", score) if self.verbose else None

        # visualize and display tree
        if self.vis:
            features = list(datasets['x_train'].columns[:])
            self.visualize_tree(dt, features, score)

        # extract tree root
        feature = dt.tree_.feature[0]
        print("tree root:\n", feature, features[feature])

        from sklearn import metrics
        y = datasets['y_test'][:100]
        scores = dt.predict(datasets['x_test'])[:100]
        fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=11)

        # plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % 0)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

        return score

    def mul_tree_learn(self):
        """ perform learning multiple ('reps') times """
        # learn and test:
        trees_scores = []
        for i in range(self.reps):
            # print progress
            if self.reps > 1 and self.verbose:
                print("[ {} / {} ]".format(i + 1, self.reps))

            res = self.learn_tree()
            trees_scores.append(res)

        print("Average score: %0.4f" % np.mean(trees_scores))



