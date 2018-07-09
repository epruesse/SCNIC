import multiprocessing
import subprocess
import tempfile
import warnings
from functools import partial
from itertools import combinations, chain
from os import path

from biom.table import Table

import pandas as pd

from scipy.stats import spearmanr

from tqdm import tqdm

from SCNIC.general import p_adjust


def corr_vector(i, j, corr_method):
    res = [
        tuple(corr_method(i, n)) for n in j
    ]
    return res


def df_to_correls(cor, col_label='r'):
    """takes a square correlation dataframe and turns it into a long form dataframe"""
    correls = pd.DataFrame(cor.stack().loc[list(combinations(cor.index, 2))], columns=[col_label])
    return correls


def calculate_correlations(table: Table, corr_method=spearmanr, p_adjustment_method: str = 'fdr_bh', nprocs=1) -> pd.DataFrame:
    # TODO: multiprocess this. Casey mimed Mike's pool implementation for betweeen_correls_from_tables()
    index = list()
    data = list()

    if nprocs > multiprocessing.cpu_count():
        warnings.warn("nprocs greater than CPU count, using all avaliable CPUs")
        nprocs = multiprocessing.cpu_count()

    multiproc = 2  # alternate multiproc implementations
    nobs = len([1 for n in table.iter(axis="observation")])

    if nprocs == 1:
        for (val_i, id_i, _), (val_j, id_j, _) in tqdm(table.iter_pairwise(axis='observation'), total=nobs**2):
            r, p = corr_method(val_i, val_j)
            index.append((id_i, id_j))
            data.append((r, p))
        correls = pd.DataFrame(data, index=index, columns=['r', 'p'])
        correls.index = pd.MultiIndex.from_tuples(correls.index)  # Turn tuple index into actual multiindex
    elif multiproc == 1:
        with multiprocessing.Pool(nprocs) as pool:
            for val_i, id_i, _ in tqdm(table.iter(axis="observation"), total=nobs):
                vals_j = (val_j for val_j, id_j, _ in table.iter(axis="observation"))
                corr = partial(corr_method, b=val_i)
                corrs = pool.map(corr, vals_j)
                data += [(id_i, table.ids(axis="observation")[n], corrs[n][0], corrs[n][1])
                         for n in range(len(corrs))]

        correls = pd.DataFrame(data, columns=['feature1', 'feature2', 'r', 'p'])
        # NEEDS REVIEW: Workaround to the multiindex issue in between_two_correls_from_tables()
        index = list(zip(correls['feature1'], correls['feature2']))
        correls.index = pd.MultiIndex.from_tuples(index)
        correls.drop(columns=['feature1','feature2'])
    elif multiproc == 2:
        corr = partial(corr_vector,
                       j=[i[0] for i in table.iter(axis="observation")],
                       corr_method=corr_method)
        with multiprocessing.Pool(nprocs) as pool:
            data = [res for res in tqdm(
                pool.imap(corr, (i[0] for i in table.iter(axis="observation"))),
                total=nobs
            )]
        data = chain(data)
        index = ((i[1],j[1])
                 for i in table.iter(axis="observation")
                 for j in table.iter(axis="observation"))
        
        correls = pd.DataFrame(data, index=index, columns=['r', 'p'])
        correls.index = pd.MultiIndex.from_tuples(index)
    if p_adjustment_method is not None:
        correls['p_adjusted'] = p_adjust(correls.p, method=p_adjustment_method)
    return correls


def fastspar_correlation(table: Table, nprocs=1, verbose: bool=False) -> pd.DataFrame:
    # TODO: multiprocess support
    with tempfile.TemporaryDirectory(prefix='fastspar') as temp:
        table.to_dataframe().to_dense().to_csv(path.join(temp, 'otu_table.tsv'), sep='\t', index_label='#OTU ID')
        if verbose:
            stdout = None
        else:
            stdout = subprocess.DEVNULL
        subprocess.run(['fastspar', '-c',  path.join(temp, 'otu_table.tsv'), '-r',
                        path.join(temp, path.join(temp, 'correl_table.tsv')), '-a',
                        path.join(temp, 'covar_table.tsv'), '-t', str(nprocs)], stdout=stdout)
        cor = pd.read_table(path.join(temp, 'correl_table.tsv'), index_col=0)
        return df_to_correls(cor)


def between_correls_from_tables(table1, table2, correl_method=spearmanr, nprocs=1):
    """Take two biom tables and correlation"""
    correls = list()

    if nprocs == 1:
        for data_i, otu_i, _ in table1.iter(axis="observation"):
            for data_j, otu_j, _ in table2.iter(axis="observation"):
                corr = correl_method(data_i, data_j)
                correls.append([otu_i, otu_j, corr[0], corr[1]])
    else:
        import multiprocessing
        if nprocs > multiprocessing.cpu_count():
            warnings.warn("nprocs greater than CPU count, using all avaliable CPUs")
            nprocs = multiprocessing.cpu_count()

        pool = multiprocessing.Pool(nprocs)
        for data_i, otu_i, _ in table1.iter(axis="observation"):
            datas_j = (data_j for data_j, _, _ in table2.iter(axis="observation"))
            corr = partial(correl_method, b=data_i)
            corrs = pool.map(corr, datas_j)
            correls += [(otu_i, table2.ids(axis="observation")[i], corrs[i][0], corrs[i][1])
                        for i in range(len(corrs))]
        pool.close()
        pool.join()

    correls = pd.DataFrame(correls, columns=['feature1', 'feature2', 'r', 'p'])
    return correls.set_index(['feature1', 'feature2'])  # this needs to be fixed, needs to return multiindex
