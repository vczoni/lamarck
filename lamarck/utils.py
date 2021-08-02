
# Objective map
objective_ascending_map = {'min': True, 'max': False}


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
