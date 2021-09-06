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
- **Versatile process modelling** (it's just a standard Python function)
- Optimization for **Single** or **Multiple Objectives**
- Simulation control
    - Maximum **number of Generations**
    - Maximum **Stall** (halt if the simulation is not finding better solutions)
    - **Multithreading**
    - **Mutation** proportion
    - **Selection** proportion
- **Reproduction** control
- **Constraint** addition
- **Visualization** Tools
    - Variable pair
    - Evolution of the solutions
    - Pareto fronts

## Examples
### Basic Example #1 - Simple optimization run

```python
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
# x           33.008920
# y           33.023696
# val         66.001915
# Criteria    66.001915
# Rank         1.000000

# So it found x=33.01 and y=33.02 as the best Solution.
```

### Basic Example #2 - Travelling Salesman
(Using the module from the `docs/examples` folder: *`toymodules/salesman.py`*)
```python
from toymodules.salesman import TravelSalesman
from lamarck import BlueprintBuilder, Optimizer

builder = BlueprintBuilder()

# Defining the Process
# In order to persist the TravelSalesman object for the Process, we need to wrap it in a function
def process_deco(travel_salesman):
    def wrapper(route):
        distance = travel_salesman.get_route_distance(route)
        return {'distance': distance}
    return wrapper

# Genome Blueprint
# (this will only have one variable that is a "Set" of the particular order of cities that the
# salesman should vist)
number_of_cities = 20
cities = tuple(range(number_of_cities))
builder.add_set_gene(name='route',
                     domain=cities,
                     length=number_of_cities)

# Creating the Population (5000 randomly generated routes)
blueprint = builder.get_blueprint()
pop = blueprint.populate.random(n=5000)

# Checking the cities "map"
trav_salesman = TravelSalesman(number_of_cities, seed=123)
trav_salesman.plot()
```
<img src="docs/img/salesman_10c_space.png"/>

```python
# Setting up the Process
process = process_deco(trav_salesman)

# Setting up the Optimizer
opt = Optimizer(population=pop, process=process)

# Activate MultiThreading (for better performance)
opt.config.multithread = True

# peek best solution during runtime
opt.config.peek_champion_variables = ['route', 'distance']
```
##### - "Light" Simulation:
```python
# Simulate
opt.config.max_generations = 5
opt.simulate.single_criteria(output='distance', objective='min')

# Lets check this route
best_creature = opt.datasets.get_best_criature()
best_route = best_creature['route']
trav_salesman.plot_route(best_route)
```
<img src="docs/img/salesman_20c_light_best.png"/>

##### - "Heavier" Simulation:
```python
# Now lets amp up the number of generations
opt.config.max_generations = 40
opt.config.max_stall = 10
opt.simulate.single_criteria(output='distance', objective='min')

# Lets check this route
best_creature = opt.datasets.get_best_criature()
best_route = best_creature['route']
trav_salesman.plot_route(best_route)
```
<img src="docs/img/salesman_20c_heavy_best.png"/>

```python
# So The best Sequence it found (minimum 'distance' travelled) is:
# (9, 18, 3, 0, 6, 16, 12, 7, 13, 11, 15, 10, 5, 8, 14, 2, 17, 4, 19, 1)

# Just so you know, there are 1.216.451.004.088.320.000 different Routes in
# this problem (of 20 cities that are all interconnected), so THE best solution
# is REALLY HARD to find but the algorithm will get very close very fast
```
###### Evolution of the Species:
<img src="docs/img/salesman_20c_evolution.gif"/>


### Basic Example #3 - Local Maximum
```python
import numpy as np
from matplotlib import pyplot as plt
from lamarck import Optimizer, BlueprintBuilder, HistoryExplorer

def process(x, y):
    val = np.sin(x)*x + np.sin(y)*y
    return {'val': val}

# Checking the solution space...
maxrange = 12
x = np.linspace(0, maxrange*np.pi, 100)
y = np.linspace(0, maxrange*np.pi, 100)
Xi, Yi = np.meshgrid(x, y)
Z = process(Xi, Yi)['val']

fig, ax = plt.subplots(constrained_layout=True)
CS = ax.contourf(x, y, Z, 15, cmap=plt.cm.bone)
cbar = fig.colorbar(CS)
ax = cbar.ax.set_ylabel('val')
```
<img src="docs/img/local_max_space.png"/>

```python
# Genome Creation
builder = BlueprintBuilder()
builder.add_float_gene(name='x',
                       domain=(0, maxrange*np.pi))
builder.add_float_gene(name='y',
                       domain=(0, maxrange*np.pi))
blueprint = builder.get_blueprint()

# Population
pop = blueprint.populate.deterministic(20) + blueprint.populate.random(600)
opt = Optimizer(pop, process)

# Create Optimizer
opt = Optimizer(population=pop, process=process) 

# Populated solution space:
```
<img src="docs/img/local_max_space_pop.png"/>

```python
# simulate
opt.simulate.single_criteria(output='val', objective='max')

# Evolution of the Species:
```
<img src="docs/img/local_max_evolution.gif"/>


## Starting away from optimum space:
```python
constraint = lambda x, y: (x < 10) & (y < 10)
blueprint.add_constraint(constraint)

new_pop = blueprint.populate.deterministic(20) + blueprint.populate.random(600)
```

#### Simulating WITHOUT MUTATION
```python
opt = Optimizer(new_pop, process)
opt.config.p_mutation = 0
opt.simulate.single_criteria(output='val', objective='max')
```
<img src="docs/img/local_max_no_mutation_evolution.gif"/>

###### Creatures get stuck in a local maximum

#### Simulating WITH MUTATION
```python
opt.config.p_mutation = 0.1
opt.simulate.single_criteria(output='val', objective='max')
```
<img src="docs/img/local_max_mutation_evolution.gif"/>

###### Creatures now gravitate towards the global maximum

#### What if there's a gap between the global optimum and the starting area?
```python
def new_process(x, y):
    if not (((x < 10) & (y < 10)) | ((x > 30) & (y > 30))):
        val = -80
    else:
        val = process(x, y)['val']
    return {'val': val}

opt = Optimizer(new_pop, new_process)
opt.simulate.single_criteria(output='val', objective='max', quiet=True)
```
<img src="docs/img/local_max_isolated_1gene_evolution.gif"/>

###### Creatures can't reach the best solution because only one gene is mutated

###### ... so we need to mutate both 'x' and 'y' genes. We do that just by changing the 'max_mutated_genes' config:
```python
opt.config.max_mutated_genes = 2
opt.simulate.single_criteria(output='val', objective='max', quiet=True)
```
<img src="docs/img/local_max_isolated_2gene_evolution.gif"/>

###### Now it eventually reach the other area and find the global max =)


## Simulation Configs
###### Default values:
```python
max_generations: int = 20 # maximum number of Generations
max_stall: int = 5 # How many times the simulation will run with no new best Creature
p_selection: float = 0.5 # Proportion of the Population that will survive to the next Generation
n_dispute: int = 2 # How many Creatures will be randomly selected to dispute for each Parent spot
n_parents: int = 2 # Amount of Parents that will provide genomes to mix and form a new Creature
children_per_relation: int = 2 # How many children the same group of parents will generate
p_mutation: float = 0.1 # Proportion of New Population that will be generated by Mutation
max_mutated_genes: int = 1 # Maximum amount of genes the can mutate
children_per_mutation: int = 1 # How many children the same Creature will generate by Mutating
multithread: bool = True # Using Multithread to simulate multiple Creatures concurrently
max_workers: int | None = None # Multithread Workers
peek_champion_variables: list | None = None # Select some variables to show during
```


### Genome Specifications

1. Scalar

    1. Integer
        1. Name: `str`
        2. Domain: `list or tuple: [min, max] or (min, max)`

    2. Float
        1. Name: `str`
        2. Domain: `list or tuple: [min, max] or (min, max)`

    3. Categorical
        1. Name: `str`
        2. Domain: `list or tuple`
    
    4. Boolean
        1. Name: `str`
>

2. Vectorial
    
    1. Array
        1. Name: `str`
        2. Domain: `list or tuple`
        3. Length: `int`
    
    1. Set
        1. Name: `str`
        2. Domain: `list or tuple`
        3. Length: `int`

#### Genome Example
```python
from lamarck import Blueprint

genome_blueprint_dict = {
    'num_var_int': {
        'type': 'integer',
        'domain':  (0, 10) # [min, max]
    },
    'num_var_float': {
        'type': 'float',
        'domain':  (2.5, 7.5) # [min, max]
    },
    'cat_var': {
        'type': 'categorical',
        'domain': ('A', 'B', 'C', 'D', 'E'),
    },
    'vec_var': {
        'type': 'set',
        'domain': (0, 1, 2, 3, 4, 5),
        'length': 3, # lenght CANNOT be greater than the domain's length
    },
    'vec_var_replace': {
        'type': 'array',
        'domain': ('i', 'j', 'k'),
        'length': 5, # lenght CAN be greater than the domain's length because of the replacement
    },
    'bool_var': {
        'type': 'boolean',
    }
}

blueprint = Blueprint(genome_blueprint_dict)
```
##### Using the BlueprintBuilder to assist the creation of this (fairly complicated) object:
```python
from lamarck import BlueprintBuilder

builder = BlueprintBuilder()

# num_var_int
builder.add_integer_gene(name='num_var_int',
                         domain=(0, 10))

# num_var_float
builder.add_float_gene(name='num_var_float',
                       domain=(2.5, 7.5))

# cat_var
builder.add_categorical_gene(name='cat_var',
                             domain=('A', 'B', 'C', 'D', 'E'))

# vec_var
builder.add_set_gene(name='vec_var',
                     domain=(0, 1, 2, 3, 4, 5),
                     length=3)

# vec_var_replace
builder.add_array_gene(name='vec_var_replace',
                       domain=('i', 'j', 'k'),
                       length=5)

# bool_var
builder.add_bool_gene(name='bool_var')

# generate blueprint
blueprint = builder.get_blueprint()
```

##### Defining the process
This genome blueprint will help build the population with multiple values for the variables `num_var_int`, `num_var_float`, `cat_var`, `vec_var`, `vec_var_replace` and `bool_var`.

In this case, the `Process` must be a `Function` that has those variables as parameters and the output MUST ALWAYS be a `dict` with all the desired outputs.

Also, it is important that all the outputs are defined **within the function and NOT within the output `dict`**. The reason for this is that in order to get the *outputs*'s names, the algorithm will look for them in the function's `__code__.co_consts` and it **may** create a **problem** with its ordering. 

- Example of a **VALID** process output:
```python
def some_process(num_var_int, num_var_float, cat_var, vec_var, vec_var_replace, bool_var):
    out1 = func1(num_var_int, num_var_float)
    if bool_var:
        out2 = func2(cat_var, vec_var_replace)
    else:
        out2 = func3(cat_var, vec_var)
    return {'output_1': out1, 'output_2': out2}
```

- Example of a **PROBLEMATIC** process output (**may work but avoid doing this**):
```python
def some_process(num_var_int, num_var_float, cat_var, vec_var, vec_var_replace, bool_var):
    return {
        'output_1': func1(num_var_int, num_var_float),
        'output_2': func2(cat_var, vec_var_replace) if bool_var else func3(cat_var, vec_var)
    }
```


##### For more examples and use cases, check out the `docs/examples` directory.

