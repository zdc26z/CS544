#Kholby Lawson, CS544, Fall 2015

#imports
from random import Random
from time import time
from math import sin
from math import sqrt
from inspyred import ec
from inspyred.ec import terminators

#generator function
def generate_schwefel(random, args):
    #default to 2 inputs
    size = args.get('num_inputs', 2)
    #pick value randomly between -500 and 500
    return [random.uniform(-500, 500) for i in range(size)] 

#evaluator function
def evaluate_schwefel(candidates, args):
    fitness = []
    for cs in candidates:
        #schweful evaluation
        #f(x) = 418.9829n - [sum from i to n](-x[i]sin(sqrt(abs(x[i]))))
        fit = 418.9829 * len(cs) - sum([(-x * sin(sqrt(abs(x)))) for x in cs])
        fitness.append(fit)
    return fitness

#ec
rand = Random()
rand.seed(int(time()))
es = ec.ES(rand)
es.terminator = terminators.evaluation_termination
final_pop = es.evolve(generator=generate_schwefel,
                      evaluator=evaluate_schwefel,
                      pop_size=100,
                      maximize=False,
                      bounder=ec.Bounder(-500, 500),
                      max_evaluations=20000,
                      mutation_rate=0.25,
                      num_inputs=2)
# Sort and print the best individual, who will be at index 0.
final_pop.sort(reverse=True)
print(final_pop[0])








