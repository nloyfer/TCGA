from sklearn.utils import shuffle
import os.path
import pandas as pd


class Df_manager():
    """
    API:
    pick_sites(path) - filter self.df to subset of sites
    df2datasets(datasets): - make datasets a dict of datasets (x_train, y_test etc.),
                             from self.df
    """
    def __init__(self, df, test_rate, translator, verbose):
        self.df = df                             # the DataFrame
        self.test_rate = test_rate               # test rate
        self.translator = translator             # translator from label to cancer type
        self.verbose = verbose                   # verbose
        self.labels = self.df['label'].unique()  # unique labels in df

    #################################
    #                               #
    #   Sites filtering logic       #
    #                               #
    #################################

    def _filter_df(self, sites):
        # Filter the DataFrame self.df to contain only the sites from the list 'sites'
        small_df = pd.concat([self.df[site] for site in sites + ['label']], axis=1)
        self.df = small_df

    def pick_sites_from_file(self, path):
        """ given a path to a file of sites, update self.df to contain only data on
            that subset of sites """

        if self.verbose:
            print("Parsing sites file...")

        # read file:
        sites = [site.rstrip('\n') for site in open(path)]

        # validate list of sites:
        features = list(self.df.columns[:-1])
        valid_sites = []
        for site in sites:
            if site in features:
                valid_sites.append(site)

        if self.verbose:
            print("{} valid sites read ({} removed)".format(len(valid_sites), len(sites) - len(valid_sites)))

        # update the DataFrame
        self._filter_df(valid_sites)
        return

    def pick_sites(self, sites):
        """ given a string 'sites', filter self.df accordingly """
        if os.path.isfile(sites):               # if 'sites' is a file
            self.pick_sites_from_file(sites)
        elif sites in self.df.columns[:-1]:     # if 'sites' is a single valid CpG site
            self._filter_df([sites])
        else:
            print("Invalid site/s source:", sites, "\nAbort.")
            exit(0)

    ##################################
    #                                #
    #   DataFrame -> datasets logic  #
    #                                #
    ##################################

    def _split_train_test(self):
        """ split df to test and train, keep train:test ratio similar across classes  """

        # split df to classes by labels:
        classes = [self.df[self.df['label'] == l] for l in self.labels]

        # split each class to train and test, and concat all train and all test:
        df_train, df_test = pd.DataFrame(), pd.DataFrame()
        for class_df in classes:
            test_size = int(self.test_rate * class_df.shape[0])
            class_train, class_test = class_df[test_size:], class_df[:test_size]
            df_train = df_train.append(class_train)
            df_test = df_test.append(class_test)

        # shuffle datasets:
        df_train = shuffle(df_train)
        df_test = shuffle(df_test)

        # print sizes of classes and datasets:
        if self.verbose:
            self._print_shapes(df_train, "training")
            self._print_shapes(df_test, "test")

        return df_train, df_test

    def _print_shapes(self, df, name):
        """ print name and size of a dataframe, and # of each label in it """
        print("\n{} set size: {}".format(name, df.shape[0]))
        print("#\tlabel")
        for l in self.labels:
            print("{}\t{} ({})".format(df[df['label'] == l].shape[0], self.translator[l], int(l)))

    def df2datasets(self, datasets):
        """ split df to train and test, and update the datasets dictionary """

        # split df to training and testing tables
        df_train, df_test = self._split_train_test()

        # return (by updating the dictionary self.datasets):
        datasets['y_train'] = df_train['label']
        del df_train['label']
        datasets['x_train'] = df_train

        datasets['y_test'] = df_test['label']
        del df_test['label']
        datasets['x_test'] = df_test

