
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
builder.add_integer_gene(name='w',
                         domain=(8, 15))

builder.add_float_gene(name='x',
                       domain=(0, 10))

builder.add_categorical_gene(name='y',
                             domain=('A', 'B', 'C'))

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
>        'type': 'integer',
>        'specs': {'domain': (8, 15)}},
>    'x': {
>        'type': 'numeric',
>        'specs': {'domain': (0, 10)}},
>    'y': {
>        'type': 'categorical',
>        'specs': {'domain': ('A', 'B', 'C')}},
>    'z': {
>        'type': 'array',
>        'specs': {'domain': (0, 1, 2, 3, 4),
>                  'length': 5}},
>    'w': {
>        'type': 'set',
>        'specs': {'domain': (0, 1, 2, 3, 4),
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
# creating the population
import pandas as pd

popdet = blueprint.populate.deterministic(n=3)
poprand = blueprint.populate.random(n=200)
pop = popdet + poprand

pop.head()
```
|    |   w |   x | y   | z               | flag   |
|---:|----:|----:|:----|:----------------|:-------|
|  0 |   8 |   0 | A   | (0, 1, 2, 3, 4) | False  |
|  1 |   8 |   0 | A   | (0, 1, 2, 3, 4) | True   |
|  2 |   8 |   0 | A   | (0, 4, 3, 2, 1) | False  |
|  3 |   8 |   0 | A   | (0, 4, 3, 2, 1) | True   |
|  4 |   8 |   0 | A   | (1, 4, 3, 2, 0) | False  |

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
| id                     |   power |   diff |
|:-----------------------|--------:|-------:|
| 5b3817cc550623ddfa7a...|      66 |     -2 |
| 3fd076410604c57dbd90...|      66 |      8 |
| bd2b6f57b8ad1b57e9dd...|      68 |     -2 |
| c4db4ccc569b41bd7a8a...|      68 |      8 |
| 2e9ca363a8f09621acaf...|      90 |     -2 |


### 2.4. Run some Fitness Criteria
Will apply the selected criteria to the results data and return a Pandas DataFrame.
#### 2.4.1. Single Objective
```python
opt.apply_fitness.single_criteria(output='power', objective='max')
opt.datasets.fitness.head()
```
| id                     |   Criteria |   Rank |
|:-----------------------|-----------:|-------:|
| 5b3817cc550623ddfa7a...|         66 |   1575 |
| 3fd076410604c57dbd90...|         66 |   1575 |
| bd2b6f57b8ad1b57e9dd...|         68 |   1526 |
| c4db4ccc569b41bd7a8a...|         68 |   1526 |
| 2e9ca363a8f09621acaf...|         90 |   1176 |
#### 2.4.2. Multi Objective
##### 2.4.2.1. Ranked Objectives
```python
opt.apply_fitness.multi_criteria_ranked(outputs=['power', 'diff'],
                                        objectives=['max', 'min'])
opt.datasets.fitness.head()
```
| id                     |   Criteria1 |   Criteria2 |   Rank |
|:-----------------------|------------:|------------:|-------:|
| 5b3817cc550623ddfa7a...|          66 |          -2 |   1575 |
| 3fd076410604c57dbd90...|          66 |           8 |   1585 |
| bd2b6f57b8ad1b57e9dd...|          68 |          -2 |   1526 |
| c4db4ccc569b41bd7a8a...|          68 |           8 |   1534 |
| 2e9ca363a8f09621acaf...|          90 |          -2 |   1176 |

##### 2.4.2.2. Pareto Fronts
```python
opt.apply_fitness.multi_criteria_pareto(outputs=['power', 'diff'],
                                        objectives=['max', 'min'])
opt.datasets.fitness.head()
```
| id                     |   Front |   Crowd |   Rank |
|:-----------------------|--------:|--------:|-------:|
| 5b3817cc550623ddfa7a...|      10 |       0 |   1366 |
| 3fd076410604c57dbd90...|      10 |       0 |   1366 |
| bd2b6f57b8ad1b57e9dd...|      10 |       1 |    170 |
| c4db4ccc569b41bd7a8a...|      10 |       0 |   1366 |
| 2e9ca363a8f09621acaf...|      10 |       0 |   1366 |


## 3. Reproduction

### 3.1. Selection
```python
from lamarck.optimizer import select_fittest

ranked_pop = opt.datasets.simulation
fittest_pop = select_fittest(ranked_pop=ranked_pop, p=0.5, rank_col='Rank')
fittest_pop.data.head()
```
| id       |   w |        x | y   | z               | flag   |   power |   diff |   Front |    Crowd |   Rank |
|:---------|----:|---------:|:----|:----------------|:-------|--------:|-------:|--------:|---------:|-------:|
| 26443c...|  15 | 10       | A   | (1, 4, 3, 2, 0) | True   | 240     |      5 |       1 | inf      |      1 |
| 5c70c2...|   8 |  6       | A   | (1, 4, 3, 2, 0) | False  | 138     |     -2 |       1 | inf      |      1 |
| ae0979...|  11 |  9.56247 | A   | (1, 2, 0, 4, 3) | False  | 187.187 |      1 |       1 |  42      |      3 |
| f704a6...|   9 |  8       | A   | (1, 4, 3, 2, 0) | False  | 162     |     -1 |       1 |  34      |      4 |
| 7a07d9...|  10 |  8       | A   | (1, 4, 3, 2, 0) | False  | 170     |      0 |       1 |  27.1872 |      5 |

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
offspring.data.head()
```
|    |   w |       x | y   | z               | flag   |
|---:|----:|--------:|:----|:----------------|:-------|
|  0 |  10 | 1.54473 | C   | (0, 2, 3, 1, 4) | False  |
|  1 |  10 | 1.54473 | C   | (0, 2, 1, 3, 4) | False  |
|  2 |  12 | 2       | C   | (0, 1, 2, 3, 4) | False  |
|  3 |  12 | 2       | B   | (2, 1, 0, 3, 4) | False  |
|  4 |  13 | 8       | C   | (2, 0, 3, 4, 1) | False  |
|  5 |  10 | 2.70767 | C   | (2, 4, 3, 0, 1) | False  |


### 3.3. Mutate (Asexual Relations)
```python
mutated_offspring = populator.asexual(ranked_pop=fittest_pop,
                                      n_offspring=5,
                                      n_mutated_genes=1,
                                      children_per_creature=5,
                                      seed=42)
mutated_offspring.data.head()
```
|    |   w |       x | y   | z               | flag   |
|---:|----:|--------:|:----|:----------------|:-------|
|  0 |   8 | 7.08767 | B   | (2, 1, 4, 3, 0) | False  |
|  1 |   8 | 7.68636 | B   | (4, 3, 0, 2, 1) | False  |
|  2 |   8 | 1.16158 | B   | (2, 1, 4, 3, 0) | False  |
|  3 |   8 | 7.68636 | B   | (2, 4, 0, 1, 3) | False  |
|  4 |   8 | 7.68636 | B   | (2, 1, 4, 3, 0) | True   |


## 4. Run Simulation

### 4.1. Configure simulation control vars
```python
opt.config.max_generations = 20
opt.config.max_stall = 5
opt.config.p_selection = 0.5
opt.config.n_dispute = 2
opt.config.n_parents = 2
opt.config.children_per_relation = 2
opt.config.p_mutation = 0.1
opt.config.max_mutated_genes = 1
opt.config.children_per_mutation = 1
opt.config.multithread = True
opt.config.max_workers = None
```

### 4.2. Define objectives and selection criteria for the simulations
#### 4.2.1 Single objective
```python
bestopt = opt.simulate.single_criteria(output='power', objective='max')
```

#### 4.2.2 Multi objective - Ranked objectives
```python
bestopt = opt.simulate.multi_criteria.ranked(outputs=['power', 'diff'],
                                             objectives=['max', 'min'])
```
#### 4.2.3 Multi objective - Pareto fronts
```python
bestopt = opt.simulate.multi_criteria.pareto(outputs=['power', 'diff'],
                                             objectives=['max', 'min'])
```
