
# Lamarck: A genetyc algorithm for general-purpose optimizations

### Featuring:

1. Population generator
    - Genome blueprint creation
    - Population creation
    - Constraint definition

2. Process creation
    - The process is defined by a Python Function

3. Selection criteria
    - Single max/min objective
    - Multiple Ranked max/min objectives
    - Multiple objective defined by pareto fronts

4. Simulation control
    - Max generations
    - Max stall count
    - Target reached


## 1. Population Creation

### 1.1. Genome Bluepint Creation
```python
from lamarck import BlueprintBuilder

builder = BlueprintBuilder()

# Adding gene specs
builder.add_numeric_gene(name='w',
                         domain=int,
                         range=[8, 15])

builder.add_numeric_gene(name='x',
                         domain=float,
                         range=[0, 10])

builder.add_categorical_gene(name='y',
                             domain=['A', 'B', 'C'])

builder.add_vectorial_gene(name='z',
                           domain=(0, 1, 2, 3, 4),
                           replacement=False,
                           length=5)

builder.add_boolean_gene(name='flag')

# getting the blueprint (which is just a Python `dict`)
blueprint = builder.get_blueprint()

print(blueprint._dict)
```
>```
>{
>    'w': {
>        'type': 'numeric',
>        'specs': {'domain': <class 'int'>,
>                  'range': [8, 15]}},
>    'x': {
>        'type': 'numeric',
>        'specs': {'domain': <class 'float'>,
>                  'range': [0, 10]}},
>    'y': {
>        'type': 'categorical',
>        'specs': {'domain': ['A', 'B', 'C']}},
>    'z': {
>        'type': 'vectorial',
>        'specs': {'domain': [0, 1, 2, 3, 4],
>                  'replacement': False,
>                  'length': 5}},
>    'flag': {
>        'type': 'boolean',
>        'specs': {}}
>}
>```

### 1.2. Adding Constraints
```python
# adding constraints
def w_gt_x(w, x): return w > x
def w_plus_x_gt_3(w, x): return (w + x) > 3
blueprint.add_constraint(w_gt_x)
blueprint.add_constraint(w_plus_x_gt_3)
```

### 1.3. Population Creation
```python
# creating population (they're just Pandas DataFrames!)
import pandas as pd

popdet = blueprint.populate.deterministic(n=3)
poprand = blueprint.populate.random(n=200)
pop = pd.concat((popdet, poprand))

pop.head()
```
|    |   w |   x | y   | z               | flag   |
|---:|----:|----:|:----|:----------------|:-------|
|  0 |   8 |   0 | A   | (0, 1, 2, 3, 4) | False  |
|  1 |   8 |   0 | A   | (0, 1, 2, 3, 4) | True   |
|  2 |   8 |   0 | A   | (2, 1, 4, 3, 0) | False  |
|  3 |   8 |   0 | A   | (2, 1, 4, 3, 0) | True   |
|  4 |   8 |   0 | A   | (4, 3, 2, 1, 0) | False  |

## 2. Setting the Process
### 2.1. Python Function
```python
# setting up the function (must have input parameters matching the genes)
def my_process(w, x, y, z, flag):
    var1 = x*w + (hash(str(tuple(y)+z)) % 100)
    var2 = w - x if flag else w - hash(y)%25 + 10
    return {'power': var1, 'diff': var2}
```

### 2.2. Setting the Environment (Optimizer)
```python
from lamarck import Optimizer

opt = Optimizer(population=pop,
                process=my_process)
```

### 2.3. Run the Environment (just for testing one population)
```python
output = opt.run()
# will run the function for each gene in the current population and return its Output
# data as a Pandas DataFrame.
output.head()
```
| id                                                               |   w |   x | y   | z               | flag   |   power |   diff |
|:-----------------------------------------------------------------|----:|----:|:----|:----------------|:-------|--------:|-------:|
| 5b3817cc550623ddfa7ad6ab0712d217540a48ddb85d19eaa07773dbb11ab296 |   8 |   0 | A   | (0, 1, 2, 3, 4) | False  |      84 |     -6 |
| 3fd076410604c57dbd90282b4f469004560d317f08e0c682666ae275579a406c |   8 |   0 | A   | (0, 1, 2, 3, 4) | True   |      84 |      8 |
| 0cc50247f93271d6946ed8baba45d77f09873e4d6fbe1a7a3aa21f60405ee740 |   8 |   0 | A   | (2, 1, 4, 3, 0) | False  |      48 |     -6 |
| 8b103f0471c9a2ff8cf416be4b7dc88fb033d41b457c37bab565c59a8f804fa9 |   8 |   0 | A   | (2, 1, 4, 3, 0) | True   |      48 |      8 |
| aeb8ffb0c1457d4392ff6a3ffe5ebfdc113cc5ee14e565a0b4b9a0a304ceb618 |   8 |   0 | A   | (4, 3, 2, 1, 0) | False  |       5 |     -6 |

## 3. Run Simulation

### 3.1. Configure simulation control vars
```python
opt.config.max_generations = 50
opt.config.max_stall = 5
opt.config.p_selection = 0.5
opt.config.p_elitism = 0.1
opt.config.p_tournament = 0.4
opt.config.n_dispute = 2
opt.config.p_mutation= 0.05
opt.config.max_mutated_genes = 1
```

### 3.2. Define objectives and selection criteria for the simulations
```python
# Single objective
optpop = opt.simulate.single_criteria(output='power', objective='max')

# Multi objective
## Ranked
optpop = opt.simulate.multi_criteria.ranked(outputs=['power', 'diff'],
                                            objectives=['max', 'min'])
## Pareto
optpop = opt.simulate.multi_criteria.pareto(outputs=['power', 'diff'],
                                            objectives=['max', 'min'])
```
