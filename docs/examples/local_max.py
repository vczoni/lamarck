import numpy as np
from lamarck import Optimizer

# Defining the Process
def my_process(x, y):
    val = np.sin(x)*x + np.sin(y)*y
    return {'val': val}

# Setting up the Optimizer
opt = Optimizer()

# Genome Blueprint
# (note that 'x' and 'y' are declared, which are the exact same names as the
# parameters of the function 'process' - *this is required*)
opt.genome_creator.add_gene_specs.numeric(name='x',
                                          domain=float,
                                          range=[0, 12*np.pi])

opt.genome_creator.add_gene_specs.numeric(name='y',
                                          domain=float,
                                          range=[0, 12*np.pi])

# Creating the Population
opt.create_population(n_det=20, n_rand=600)

# Setting up the Environment
opt.set_process(my_process)

# Simulate (this will return an optimized population)
optpop = opt.run.single_objective(output='val', objective='max')

# Check the best solution
print(optpop.get_creature.best())