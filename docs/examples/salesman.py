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