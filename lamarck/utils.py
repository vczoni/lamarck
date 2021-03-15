import itertools
import numpy as np
import pandas as pd


def idfunc(x):
    return hash(x)


def column_aggregator(df):
    for col in df.columns:
        yield df[col]


def create_id(df):
    df = df.copy()
    colgen = column_aggregator(df)
    cols = tuple(colgen)
    idcol = pd.Series(tuple(zip(*cols))).apply(idfunc)
    return df.assign(id=idcol).set_index('id')


def get_id(genome):
    genomevals = tuple(genome.values())
    return idfunc(genomevals)


def genome_already_exists(genome, pop):
    return get_id(genome) in pop.datasets.index


def creature_already_exists(creature, pop):
    return creature.id in pop.datasets.index


def is_objective_ascending(objective):
    if objective.lower() in ['min', 'minimize']:
        ascending = True
    elif objective.lower() in ['max', 'maximize']:
        ascending = False
    else:
        raise Exception(":objective: must be either 'min' or 'max'.")
    return ascending


def compare_to_value(val1, oper, val2):
    if oper.lower() == 'eq':
        return val1 == val2
    elif oper.lower() == 'neq':
        return val1 != val2
    elif oper.lower() == 'lt':
        return val1 < val2
    elif oper.lower() == 'le':
        return val1 <= val2
    elif oper.lower() == 'gt':
        return val1 > val2
    elif oper.lower() == 'ge':
        return val1 >= val2
    else:
        raise Exception(f'Invalid operator "{oper}".')


def deterministic_linear(start, stop, n, dtype):
    return np.linspace(start, stop, n, dtype=dtype)


def deterministic_log(start, stop, n, dtype):
    return np.geomspace(start, stop, n, dtype=dtype)


def vectorial_distribution(n, length, domain):
    vectors = [domain] * n
    perm_gen = itertools.product(*vectors)
    end = length ** len(domain)
    step = end // n
    dist_gen = itertools.islice(perm_gen, 0, end, step)
    return tuple(dist_gen)


def vectorial_distribution_set(n, length, domain):
    perm_gen = itertools.permutations(domain, length)
    end = np.math.perm(len(domain), length)
    step = end // n
    dist_gen = itertools.islice(perm_gen, 0, end, step)
    return tuple(dist_gen)


def random_linear(start, stop):
    return np.random.uniform(start, stop)


def random_linear_dist(start, stop, n, dtype):
    return np.random.uniform(start, stop, n).astype(dtype)


def random_log_dist(start, stop, n, dtype):
    dist = [random_log(start, stop) for _ in n]
    return np.array(dist, dtype=dtype)


def random_log(start, stop):
    lstart = np.log(start)
    lstop = np.log(stop)
    val = np.random.uniform(lstart, lstop)
    return np.exp(val)
