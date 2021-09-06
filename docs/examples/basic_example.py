import numpy as np
from lamarck import BlueprintBuilder, Optimizer


# Defining the Process
def my_process(x, y):
    val = np.sin(x)*x + np.sin(y)*y
    return {'val': val}

# Building the Blueprint
builder = BlueprintBuilder()
# (note that 'x' and 'y' are declared, which are the exact same names as the
# parameters of the function 'my_process' - *this is required*)
builder.add_float_gene(name='x',
                       domain=(0, 12*np.pi))

builder.add_float_gene(name='y',
                       domain=(0, 12*np.pi))

blueprint = builder.get_blueprint()

# Building the Population
popdet = blueprint.populate.deterministic(n=20)
poprnd = blueprint.populate.random(n=600)
pop = popdet + poprnd

# Setting up the Optimizer
opt = Optimizer(pop, my_process)

# Simulate (this will return an optimized population)
opt.simulate.single_criteria(output='val', objective='max')

# Check the best solution
print(opt.datasets.get_best_criature())
