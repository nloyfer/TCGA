#!/usr/bin/python3
import argparse

# from learn_trees import Cpg_trees
from multi_class_learn import CpG_DTrees, ctypes
import os

__author__ = 'nloyfer'





def get_args():
    parser = argparse.ArgumentParser(description='Tree Classifier')

    # Cancer types
    parser.add_argument('-c', '--ctype',
                        help='Input cancer type or types',
                        required=True,
                        default=[],
                        action='append',
                        dest='cancers',
                        choices=ctypes)

    # Max depth of the tree
    parser.add_argument('-d', '--depth',
                        help='Max depth of the tree. Default=3',
                        default=3,
                        type=int)

    # repetitions
    parser.add_argument('-r', '--reps',
                        help='Repetitions. Default=1',
                        default=1,
                        type=int)

    # test rate
    parser.add_argument('-tst','--test_rate',
                        help='test rate. Default=0.2',
                        default=0.2,
                        type=float)

    # input CpG sites
    parser.add_argument('-s','--input_sites',
                        help='Input a file with list of sites (cp), or a single site name')

    # verbose
    parser.add_argument('-ver', '--verbose',
                        help='Increase output verbosity.',
                        action='store_true')

    # visualize tree
    parser.add_argument('-vis', '--visualize',
                        help='Generate and save tree visializations.',
                        action='store_true')

    # display tree visualization
    parser.add_argument('-disp', '--display',
                        help='display figures.',
                        action='store_true')

    args = parser.parse_args()
    return args


def display_args(args):
    if args.verbose:
        # show values:
        print("Cancer type\s: %s" % args.cancers)
        print("Max tree depth: %s" % args.depth)
        print("repetitions: %s" % args.reps)
        print("Test dataset rate: %s" % args.test_rate)
        print("Verbosity: %s" % args.verbose)
        print("Visualization: %s" % args.visualize)
        print("Display: %s" % args.display)
        if args.input_sites is not None:
            sites_type = "path" if os.path.isfile(args.input_sites) else "name"
            print("Input sites {}: %s".format(sites_type) % args.input_sites)
        else:
            print("Sites: all sites")
    else:
        print("Verbosity off")

args = get_args()
display_args(args)

c = CpG_DTrees(cTypes=args.cancers,
               reps=args.reps,
               test_rate=args.test_rate,
               rand_state=99,
               verbose=args.verbose,
               depth=args.depth,
               vis=args.visualize,
               sites=args.input_sites,
               display=args.display)

c.mul_tree_learn()
