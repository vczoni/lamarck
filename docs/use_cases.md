
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

4. Simulation controle
    - Max generations
    - Max stall count
    - Target reached


## 1. Population Creation

### 1.1. Genome Bluepint Creation
```python
from lamarck import GenomeBlueprintBuilder

builder = GenomeBlueprintBuilder()

# Adding gene specs
builder.add_numeric_gene(name='w',
                         domain=float,
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
```
output:
{
   'w': {
       'type': 'numeric',
       'domain': float,
       'range': [8, 15]},
   'x': {
       'type': 'numeric',
       'domain': float,
       'range': [0, 10]},
   'y': {
       'type': 'categorical',
       'domain': ['A', 'B', 'C']},
   'z': {
       'type': 'vectorial',
       'domain': [0, 1, 2, 3, 4],
       'replacement': False,
       'length': 5},
   'flag': {
       'type': 'boolean'}
}
```

### 1.2. Adding Constraints
```python
# adding constraints
def w_gt_x(w, x): w > x
def w_plus_x_gt_3(w, x): (w + x) > 3
blueprint.add_constraint(name='w_gt_x', func=w_gt_x)
blueprint.add_constraint(name='w_plus_x_gt_3', func=w_plus_x_gt_3)
```

### 1.3. Population Creation
```python
# creating population
popdet = blueprint.populate.deterministic(n=3)
poprand = blueprint.populate.random(n=200)
pop = popdet + poprand
```
