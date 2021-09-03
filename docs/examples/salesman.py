from toymodules.salesman import TravelSalesman
from lamarck import BlueprintBuilder, Optimizer


builder = BlueprintBuilder()

# Defining the Process
# In order to persist the TravelSalesman object for the Process, we need to
# wrap it in a function
def process_deco(travel_salesman):
    def wrapper(route):
        return {'distance': travel_salesman.get_route_distance(route)}
    return wrapper

# Genome Blueprint
# (this will only have one variable that is a "Vector" of the particular order
# of cities that the salesman should vist)
number_of_cities = 20
cities = tuple(range(number_of_cities))
builder.add_set_gene(name='route',
                     domain=cities,
                     length=number_of_cities)

# Creating the Population (5000 randomly generated 'route's)
blueprint = builder.get_blueprint()
pop = blueprint.populate.random(n=5000)

# Setting up the Environment
trav_salesman = TravelSalesman(number_of_cities, seed=123)
process = process_deco(trav_salesman)

# Setting up the Optimizer
opt = Optimizer(population=pop, process=process)

# Activate MultiThreading (for better performance)
opt.config.multithread = True

# Simulate (this will return an optimized population)
bestopt = opt.simulate.single_criteria(output='distance', objective='min')

# Check the best solution
best_creature = bestopt.datasets.get_best_criature(outputs='distance', objectives='min')
print([print(f'{k}: {x}') for k, x in best_creature.iteritems()])

print(bestopt.datasets.history.sort_values('distance').head())
