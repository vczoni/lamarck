
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

builder.add_array_gene(name='z',
                       domain=(0, 1, 2, 3, 4),
                       length=5)

builder.add_set_gene(name='w',
                     domain=(0, 1, 2, 3, 4),
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
>        'type': 'array',
>        'specs': {'domain': [0, 1, 2, 3, 4],
>                  'length': 5}},
>    'w': {
>        'type': 'set',
>        'specs': {'domain': [0, 1, 2, 3, 4],
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
Will run the function for each gene in the current population and return its Output data as a Pandas DataFrame.
```python
opt.run(return_data=True)
opt.datasets.results.head()
```
| id                   |   power |   diff |
|:---------------------|--------:|-------:|
| 5b3817cc550623ddfa...|      37 |     15 |
| 3fd076410604c57dbd...|      37 |      8 |
| bd2b6f57b8ad1b57e9...|       9 |     15 |
| c4db4ccc569b41bd7a...|       9 |      8 |
| 2e9ca363a8f09621ac...|      12 |     15 |

### 2.4. Run some Fitness Criteria
Will apply the selected criteria to the results data and return a Pandas DataFrame.
#### 2.4.1. Single Objective
```python
opt.apply_fitness.single_criteria(output='power', objective='max')
opt.datasets.fitness.head()
```
| id                   |   Criteria |   Rank |
|:---------------------|-----------:|-------:|
| 5b3817cc550623ddfa...|         37 |   1850 |
| 3fd076410604c57dbd...|         37 |   1850 |
| bd2b6f57b8ad1b57e9...|          9 |   2079 |
| c4db4ccc569b41bd7a...|          9 |   2079 |
| 2e9ca363a8f09621ac...|         12 |   2042 |

#### 2.4.2. Multi Objective
##### 2.4.2.1. Ranked Objectives
```python
opt.apply_fitness.multi_criteria_ranked(outputs=['power', 'diff'],
                                        objectives=['max', 'min'])
opt.datasets.fitness.head()
```
| id                   |   Criteria1 |   Criteria2 |   Rank |
|:---------------------|------------:|------------:|-------:|
| 5b3817cc550623ddfa...|          37 |          15 |   1861 |
| 3fd076410604c57dbd...|          37 |           8 |   1855 |
| bd2b6f57b8ad1b57e9...|           9 |          15 |   2084 |
| c4db4ccc569b41bd7a...|           9 |           8 |   2079 |
| 2e9ca363a8f09621ac...|          12 |          15 |   2052 |

##### 2.4.2.2. Pareto Fronts
```python
opt.apply_fitness.multi_criteria_pareto(outputs=['power', 'diff'],
                                        objectives=['max', 'min'])
opt.datasets.fitness.head()
```
| id                   |   Front |     Crowd |   Rank |
|:---------------------|--------:|----------:|-------:|
| 5b3817cc550623ddfa...|      66 | 0.265356  |   1446 |
| 3fd076410604c57dbd...|      66 | 0         |   1716 |
| bd2b6f57b8ad1b57e9...|      66 | 1.81679   |   1083 |
| c4db4ccc569b41bd7a...|      66 | 0         |   1716 |
| 2e9ca363a8f09621ac...|      66 | 0.0679169 |   1647 |

## 3. Reproduction

### 3.1. Selection
```python
from lamarck.optimizer import select_fittest

ranked_pop = opt.datasets.simulation
fittest_pop = select_fittest(ranked_pop=ranked_pop, p=0.5, rank_col='Rank')
fittest_pop.head()
```

### 3.2. Reproduce (Sexual Relations)
```python
from lamarck.reproduce import Populator

populator = Populator(blueprint=blueprint)
offspring = populator.sexual(ranked_pop=fittest_pop,
                             n_offspring=5,
                             n_dispute=2,
                             n_parents=2,
                             children_per_relation=2,
                             seed=42)
offspring
```

### 3.2. Mutate (Asexual Relations)
```python
mutated_offspring = populator.asexual(ranked_pop=fittest_pop,
                                      n_offspring=5,
                                      seed=42)
mutated_offspring
```

## 4. Run Simulation

### 4.1. Configure simulation control vars
```python
opt.config.max_generations = 50
opt.config.max_stall = 5
opt.config.p_selection = 0.5
opt.config.p_tournament = 0.4
opt.config.n_dispute = 2
opt.config.n_parents = 2
opt.config.n_children_per_relation = 2
opt.config.p_mutation= 0.05
```

### 4.2. Define objectives and selection criteria for the simulations
```python
# Single objective
optpop = opt.simulate.single_criteria(output='power', objective='max')

# Multi objective
## Ranked
optpop = opt.simulate.multi_criteria_ranked(outputs=['power', 'diff'],
                                            objectives=['max', 'min'])
## Pareto
optpop = opt.simulate.multi_criteria_pareto(outputs=['power', 'diff'],
                                            objectives=['max', 'min'])
```
