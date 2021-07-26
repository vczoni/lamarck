# Lamarck Optimizer: A Genetic Algorithm Prototype.

The Lamarck package is a simple optimization tool that operates by creating Populations of different solutions, running them through a specific Process (an user-defined function) and selecting, mixing and tweaking them to get to the best of a wide range of possibilities.

The Process must be a Python `function` with one or more input parameters and must return a `dict` of outputs. The Solution Space is defined by the "Genome Blueprint", which is a `dict` that determines how those input parameters (genes) will configure a solution (genome) by specifying what are the ranges of those variables and how they behave.

##### Basic Flow

> ```raw
> Genome Blueprint --> Population --> Process --> Results --> Selection --> Mutation
>                         ^                                                      |
>                         +--(repeat)--------------------------------------------+
> ```

## Features
- Creation of **very diverse** Populations
- **Multiple types** of Input Variables (Genes)
    - **Numeric**
    - **Categorical**
    - **Vectorial**
    - **Boolean**
- **Versatile process modelling** (it's just a normal Python function)
- Optimization for **Single** or **Multiple Objectives**
- Simulation control
    - Maximum **number of Generations**
    - Maximum **Stall** (to halt if the simulation is not finding better solutions)
    - **Mutation** probability
    - **Selection** proportion
- **Reproduction** control
    - by **Tournament**
    - by **Elitism**
- **Constraint** addition
- **Visualization** Tools
    - Variable pair
    - Evolution of the solutions
    - Pareto fronts

## Examples
### Basic Example #1

```python
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
# [Creature <###id###> - genome: {'x': 33.0503879157277, 'y': 33.075952331006285}]

# So it found x=33.05 and y=33.08 as the best Solution.
```

### Basic Example #2 - Travelling Salesman
(This one uses the module `docs/examples/toymodules/salesman.py`)
```python
from toymodules.salesman import TravelSalesman
from lamarck import Optimizer

# Defining the Process
# In order to persist the TravelSalesman object for the Process, we need to
# embed it in a function 
def process_deco(travel_salesman):
    def wrapper(route):
        return {'distance': travel_salesman.get_route_distance(route)}
    return wrapper

# Setting up the Optimizer
opt = Optimizer()

# Genome Blueprint
# (this will only have one variable that is a "Vector" of the particular order
# of cities that the salesman should vist)
number_of_cities = 20
cities = tuple(range(number_of_cities))
opt.genome_creator.add_gene_specs.vectorial(name='route',
                                            domain=cities,
                                            replace=False,
                                            length=number_of_cities)

# Creating the Population (5000 randomly generated 'route's)
opt.create_population(n_rand=5000)

# Setting up the Environment
trav_salesman = TravelSalesman(number_of_cities, seed=123)
process = process_deco(trav_salesman)
opt.set_process(process)
# Activate MultiThreading (may speed things up)
opt.env.config.set_multi(True)

# Simulate (this will return an optimized population)
optpop = opt.run.single_objective(output='distance', objective='min')

# Check the best solution
print(optpop.get_creature.best())
# [Creature <###id###> - genome: {'route': (1, 19, 4, 17, 14, 8, 5, 15, 10, 11, 2, 9, 18, 3, 0, 6, 13, 7, 12, 16)]

# So The best Sequence it found (minimum 'distance' travelled) is:
# (1, 19, 4, 17, 14, 8, 5, 15, 10, 11, 2, 9, 18, 3, 0, 6, 13, 7, 12, 16)

# Just so you know, there are 1.216.451.004.088.320.000 different Routes in
# this problem (of 20 cities that are all interconnected), so THE best solution
# is REALLY HARD to find but the algorithm will get very close very fast
```

### Genome Specifications

1. Numeric
    1.1. Domain: `<class> {int, float}`
    1.2. Range: `list or tuple: [min, max] or (min, max)`
>

2. Categorical
    2.1. Domain: `list or tuple`
>

3. Vectorial
    3.1. Domain: `list or tuple`
    3.2. Replacement: `bool {True, False}`
    3.3. Length: `int`
>

4. Boolean
>
#### Genome Example
```python
genome_blueprint = {
    'num_var': {
        'type': 'numeric',
        'domain': int,
        'range': [0, 10] # [min, max]
    },
    'cat_var': {
        'type': 'categorical',
        'domain': ['A', 'B', 'C', 'D', 'E'],
    },
    'vec_var': {
        'type': 'vectorial',
        'domain': [0, 1, 2, 3, 4, 5],
        'replacement': False,
        'length': 3, # lenght CANNOT be greater than the domain's length
    },
    'vec_var_replace': {
        'type': 'vectorial',
        'domain': ['i', 'j', 'k'],
        'replacement': True,
        'length': 5, # lenght CAN be greater than the domain's length because of the replacement
    },
    'bool_var': {
        'type': 'boolean',
    }
}

# This genome blueprint will help build the population with multiple values for
# the variables `num_var`, `cat_var`, `vec_var` and `vec_var_replace`

# In this case, the "Process" must be a Function that has those variables as
# parameters... oh and the output MUST ALWAYS be a `dict` with all the desired
# outputs.
def some_process(num_var, cat_var, vec_var, vec_var_replace, bool_var):
    return {'output_1': ...}
```

##### For more examples and use cases, check out the `docs/examples` directory.

