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


# Exceptions
class VectorialOverloadException(Exception):
    """
    Exception is raised when an impossible vector is declared in the gene specs.
    """

    def __init__(self, vector_length, domain_lenght):
        message = f"""
        Vector's length ({vector_length}) is greater than the domain's ({domain_lenght}),
        therefore its impossible to assign a vector with those specifications since the
        :replacement: parameter is set to `False`.
        """
        super().__init__(message)
