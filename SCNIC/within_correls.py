# TODO: Add parameters log output file to output folder
# TODO: Testing correls as list vs as pandas dataframe for speed

from __future__ import division

import os
import shutil

import networkx as nx

from biom import load_table
from sparcc_fast import sparcc_correlation, sparcc_correlation_w_bootstraps
from scipy.stats import spearmanr, pearsonr, kendalltau
from scipy.spatial.distance import jaccard, braycurtis, euclidean, canberra
from operator import itemgetter

import general
import correlation_analysis as ca
import distance_analysis as da
import module_maker as mm


def within_correls(args):

    logger = general.Logger("SCNIC_log.txt")
    logger["SCNIC analysis type"] = "within"

    # correlation and p-value adjustment methods
    correl_methods = {'spearman': spearmanr, 'pearson': pearsonr, 'sparcc': None, 'jaccard': jaccard,
                      'cscore': da.cscore, 'braycurtis': braycurtis, 'euclidean': euclidean, 'kendall': kendalltau,
                      'canberra': canberra}
    p_methods = {'bh': general.bh_adjust, 'bon': general.bonferroni_adjust}
    correl_method = correl_methods[args.correl_method.lower()]
    if args.p_adjust is not None:
        p_adjust = p_methods[args.p_adjust]
    else:
        p_adjust = None

    # get features to be correlated and extract metadata
    table = load_table(args.input)
    logger["input table"] = args.input
    metadata = general.get_metadata_from_table(table)
    print "Table loaded: " + str(table.shape[0]) + " observations"
    print ""
    logger["number of samples in input table"] = table.shape[1]
    logger["number of observations in input table"] = table.shape[0]

    # check if output directory already exists and if it does delete it
    #TODO: change this so it only deletes things used by SCNIC within or overwrites
    if args.force:
        shutil.rmtree(args.output, ignore_errors=True)

    # make new output directory and change to it
    if args.output is not None:
        os.makedirs(args.output)
        os.chdir(args.output)
        logger["output directory"] = args.output

    # filter
    if args.min_sample is not None:
        table_filt = general.filter_table(table, args.min_sample)
        print "Table filtered: " + str(table_filt.shape[0]) + " observations"
        print ""
        logger["min samples present"] = args.min_sample
        logger["number of observations present after filter"] = table_filt.shape[0]
    elif args.sparcc_filter is True:
        table_filt = general.sparcc_paper_filter(table)
        print "Table filtered: " + str(table_filt.shape[0]) + " observations"
        print ""
        logger["sparcc paper filter"] = True
        logger["number of observations present after filter"] = table_filt.shape[0]

    else:
        table_filt = table

    logger["number of processors used"] = args.procs

    # correlate feature
    if correl_method in [spearmanr, pearsonr, kendalltau]:
        # use outlier removal
        logger["correlation method used"] = args.correl_method
        logger["p-value adjustment method"] = args.p_adjust
        if args.outlier_removal:
            print "Correlating with outlier removal."
            logger["outlier removal used"] = True
            # remove outlier observations
            # first attempt with just looking at individual otu's
            good_samples = general.remove_outliers(table_filt)
            print "Outliers removed: " + str(len(good_samples)) + " observations"
            print ""

            correls = ca.paired_correlations_from_table_with_outlier_removal(table_filt, good_samples,
                                                                                            correl_method, p_adjust)
        # calculate correlations normally
        else:
            print "Correlating with " + args.correl_method
            # correlate feature
            correls = ca.paired_correlations_from_table(table_filt, args.correl_method, p_adjust,
                                                                       nprocs=args.procs)

    # calculate distances
    elif correl_method in [jaccard, braycurtis, euclidean]:
        table_filt_rar = table_filt.subsample(args.rarefaction_level)
        print "Computing pairwise distances with " + args.correl_method
        logger["distance metric used"] = args.correl_method
        if args.min_p is not None:
            logger["number of bootstraps"] = args.bootstraps
            logger["p-value adjustment method"] = args.p_adjust
            correls = da.bootstrap_distance_vals(table_filt_rar, args.correl_method, nprocs=args.procs,
                                                                bootstraps=args.bootstraps, p_adjust=p_adjust)
        else:
            correls = da.paired_distances_from_table(table_filt_rar, args.correl_method)

    else:
        print "Correlating using sparcc"

        if args.min_p is None:
            correls = sparcc_correlation(table_filt)
        else:
            correls = sparcc_correlation_w_bootstraps(table_filt, args.procs, args.bootstraps)
            logger["number of bootstraps"] = args.bootstraps
            if p_adjust is not None:
                correls['p-adj'] = p_adjust(correls.p)
            logger["p-value adjustment method"] = args.p_adjust

    correls.sort_values([correls.columns[-1], correls.columns[2]], inplace=True)
    correls.to_csv(open('correls.txt', 'w'), sep='\t', index=False)

    print "Features Correlated"

    # make correlation network
    logger["network making minimum p-value"] = args.min_p
    logger["network making minimum r value"] = args.min_r
    net = general.correls_to_net(correls, conet=True, metadata=metadata, min_p=args.min_p, min_r=args.min_r)
    print "Network Generated"
    print "number of nodes: " + str(net.number_of_nodes())
    print "number of edges: " + str(net.number_of_edges())
    print ""
    logger["number of nodes"] = net.number_of_nodes()
    logger["number of edges"] = net.number_of_edges()

    # make modules
    logger["clique percolation method k-size"] = args.k_size
    net, cliques = mm.make_modules(net, args.k_size)
    logger["number of modules created"] = len(cliques)
    print "Modules Formed"
    print "number of modules: " + str(len(cliques))
    print ""

    # print network
    nx.write_gml(net, 'conetwork.gml')

    # collapse modules
    coll_table = mm.collapse_modules(table, cliques, args.prefix)
    logger["number of observations in output table"] = coll_table.shape[0]
    print "Table Collapsed"
    print "Collapsed Table Observations: " + str(coll_table.shape[0])

    # print new table
    coll_table.to_json('make_modules.py', open('collapsed.biom', 'w'))

    logger.output_log()
    print('\a')
