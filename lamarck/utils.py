import hashlib
import pandas as pd


# Objective map
objective_ascending_map = {'min': True, 'max': False}


# functions
def hash_tuple(obj: tuple) -> str:
    """
    Custom hashing function.
    """
    s = str(obj).encode()
    return hashlib.sha256(s).hexdigest()


def hash_cols(df: pd.DataFrame) -> pd.Series:
    """
    Transform row data in Tuples and hash them.
    """
    cols = (df[col] for col in df.columns)
    return pd.Series(tuple(zip(*cols)), name='id').apply(hash_tuple)


def get_outputs(func):
    outputs = func.__code__.co_consts
    if isinstance(outputs[-1], tuple):
        return outputs[-1]
    else:
        return (outputs[-1],)


# Exceptions
class VectorialOverloadException(Exception):
    """
    Exception is raised when an impossible vector is declared in the gene specs.
    """

    def __init__(self, set_length, domain_lenght):
        message = (
            f"`SetGene`'s length ({set_length}) is greater than the domain's "
            f"({domain_lenght}), therefore its impossible to assign a SetGene with "
            "those specifications."
        )
        super().__init__(message)


class ParentOverloadException(Exception):
    """
    Exception is raised when an impossible number of parents is somehow solicited.
    """

    def __init__(self, total_parents, solicited_parents):
        message = (
            f"Total number of parents ({total_parents}) is lower than the number of different "
            "parents required to fulfill the specifications (in that case, a number of "
            f"{solicited_parents} is required)."
        )
        super().__init__(message)
