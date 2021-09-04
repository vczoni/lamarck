import numpy as np
import time
from lamarck import BlueprintBuilder, Optimizer

# Defining the Process
def my_process(x, y):
    val = np.sin(x)*x + np.sin(y)*y
    time.sleep(0.1)
    return {'val': val}

# Building the Blueprint
builder = BlueprintBuilder()
# Adding gene specs
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
bestopt = opt.simulate.single_criteria(output='val', objective='max')

# Check the best solution
print(bestopt.datasets.get_best_criature())
