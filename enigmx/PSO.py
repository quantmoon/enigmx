"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""

import numpy as np
import time
import math
import random as rand
from operator import attrgetter
from copy import deepcopy
import matplotlib.pyplot as plt

np.random.seed(0)

#%load_ext autoreload
#%autoreload 2

#%matplotlib inline
#%config InlineBackend.figure_format = "retina"

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler


class Particle(object):
    """Clase Particle para almacenar informacion de una particula en PSO """
    #una particula representa una solucion en nuestro espacio de búsqueda
    #una solución es un vector de coordenadas: posición son las coordenadas de la solución
    #sin embargo, una partícula también tiene velocidad.
    #combinamos los dos vectores: posición, velocidad, el "fitness" (dimension que evalúa esa partícula),
    #                             tienen una mejor posición ya vista ("best position")
    #Todoeso guarda la partícula: nunca muere, solo se mueve en el espacio de búsqueda
    #recordando su mejor posición
    #recordar que cada partícula es evaluada en una dimensión de "fitness"
    #nosotros queremos ubicar en dicho rank dimensional de fitness
    def __init__(self, initial_position, initial_velocity, fitness): #constructor recibe posicion y velocidad inicial y fitness
        self.position = initial_position
        self.velocity = initial_velocity
        self.fitness = fitness
        self.best_position = initial_position
        self.best_fitness = fitness
        
        
class PSO_particles:
    """ Clase que implementa el generador de partículas del optimizador PSO. El constructor puede recibir:
        PN: numero de particulas (Particles Number)
        coord_min: vector con los limites inferiores para delimitar el espacio de busqueda
        coord_max: vector con los limites superiores para delimitar el espacio de busqueda """
    
    def __init__(self, PN, coord_min, coord_max, seed=0):
        self.PN = PN
        self.coord_min = np.array(coord_min)
        self.coord_max = np.array(coord_max)
        self.seed = seed
        self.values = self.initialize_particles()

    def create_particle(self):   # Instancia una particula aleatoria dentro de los limites de busqueda
        #iniciamos la búsqueda en un subespacio dentro del recuadro verde
        #para puntualizar la busqueda
        position = self.coord_min + rand.random()*(self.coord_max - self.coord_min)
        Vmin = -1*(self.coord_max - self.coord_min)
        Vmax = (self.coord_max - self.coord_min)
        velocity = Vmin + rand.random()*(Vmax - Vmin)
        return [position, velocity] #definicion de posicion y velocidad de la particula
    
    def initialize_particles(self):  # crea las PN particles de PSO 
        return [self.create_particle() for _ in range(self.PN)]
    
    
class PSO:
    """ Clase que implementa el optimizador PSO. El constructor puede recibir:
        fn: La funcion a ser minimizada
        w: factor de inercia de la particula
        phi1: peso de aprendizaje cognitivo
        phi2: peso de aprendizaje social
        max_iter: número total de iteraciones que ejecutará PSO """

    def __init__(self, fn, particles, w, phi1, phi2, max_iter):  
        self.fn = fn
        self.PN = particles.PN
        self.coord_min = particles.coord_min
        self.coord_max = particles.coord_max
        self.w = w
        self.phi1 = phi1
        self.phi2 = phi2
        self.max_iter = max_iter
        self.particles = self.set_particles(particles.values)
        self.best_position_swarm = []
        self.best_fitness_swarm = []
    
    def set_particles(self, particles):
        return [Particle(solution[0], solution[1], self.fitness(solution[0])) for solution in particles]

    def fitness(self, position):  
      #hace un mapeamiento de maximización a la función de minimización 
        # el fitness para fn positivos es trasladado a valores entre [0,1]: 0 para fn=inf y 1 para fn=0. 
        # el fitness para fn negativos es trasladado a valores entre ]1,inf]: 1 para fn=-0 y inf para fn=-inf    
        result = self.fn(position)
        if result >= 0:
            fitness = 1 / (1 + result)
        else:
            fitness = 1 + abs(result)
        return fitness
    
    def get_bestparticle(self):
      #retorna particula con el fitness mas alto
        best = max(self.particles, key=attrgetter('fitness'))
        return best
        
    def optimize(self):
        start_time = time.time()
        #print ('Iniciando optimizacion con Algoritmo PSO')
        
        history_bestfitness = []
        history_bestsolution = []
        best_particle = self.get_bestparticle()
        self.best_position_swarm, self.best_fitness_swarm = deepcopy(best_particle.position), best_particle.fitness 
        history_bestfitness.append(self.best_fitness_swarm)  # almacena la historia de mejores fitness en cada ciclo
        history_bestsolution.append(self.best_position_swarm)
        #print("Mejor solucion inicial = {}, fitness = {}".format(self.best_position_swarm, self.best_fitness_swarm))

        for g in range(self.max_iter):  # For each cycle
            #garantiza que todas las particulas tengan su mejor valor, posicion y fitness actualizado
            for i in range(self.PN): # por cada particula en el swarm

                # si la particula i es mejor que la mejor posicion que ya vió la particula
                if self.particles[i].fitness > self.particles[i].best_fitness: 
                  #actualiza
                    self.particles[i].best_position = deepcopy(self.particles[i].position)
                    self.particles[i].best_fitness =  self.particles[i].fitness
                #si no es mejor, pasa a la siguietne particula
                # si la mejor posicion que ya vió la particula i es mejor que la mejor position de todo el swarm
                if self.particles[i].best_fitness > self.best_fitness_swarm:
                    self.best_position_swarm = deepcopy(self.particles[i].best_position)
                    self.best_fitness_swarm  = self.particles[i].best_fitness
                    

            #vuelve a recorrer las partículas otra vez para verificar si cambia la mejor partícula
            #la idea es poder generar una mejor posición, en caso lo hubiere
            # Actualiza la velocidad y position de cada particula 1
            for i in range(self.PN): # por cada particula en el swarm
                r1 = rand.random()#numeros random entre 0 y 1
                r2 = rand.random()#numeros random entre 0 y 1
                #'w':parametro de inercia
                self.particles[i].velocity = self.w*self.particles[i].velocity + self.phi1*r1*(self.particles[i].best_position - self.particles[i].position) + self.phi2*r2*(self.best_position_swarm - self.particles[i].position)
                self.particles[i].position = self.particles[i].position + self.particles[i].velocity
                self.particles[i].fitness  = self.fitness(self.particles[i].position)
                

             ## Obtiene la mejor posicion encontrada en este ciclo
            best_particle = self.get_bestparticle()  # mejor posicion del presente ciclo 
            history_bestfitness.append(best_particle.fitness)
            history_bestsolution.append(best_particle.position)
            
            #if (g % 5 == 0): # muestra resultados cada 5 ciclos
            #print("Ciclo {}, Mejor solucion del ciclo = {} (fitness = {}))".format(g, best_particle.position, best_particle.fitness ))
        end_time = time.time()
        #print("Mejor solucion encontrada por PSO: {}, fitness = {}. Tomo {} seg ".format(self.best_position_swarm, self.best_fitness_swarm, end_time-start_time))
        return self.best_position_swarm, self.best_fitness_swarm, history_bestfitness, history_bestsolution
    
    
## Funcion  ackley, Typical coord_min = [-20, -20] , coord_max = [20, 20] , optimum at  [0, 0]
## Estas funciones son gradiente-dependientes: depende de la lectura que haga el algoritmo
def ackley(d, *, a=20, b=0.2, c=2*np.pi):
    sum_part1 = np.sum([x**2 for x in d])
    part1 = -1.0 * a * np.exp(-1.0 * b * np.sqrt((1.0/len(d)) * sum_part1))
    sum_part2 = np.sum([np.cos(c * x) for x in d])
    part2 = -1.0 * np.exp((1.0 / len(d)) * sum_part2)

    return a + np.exp(1) + part1 + part2

## Funcion  rastrigin, Typical coord_min = [-5, -5] , coord_max = [5, 5] , optimum at  [0, 0]
def rastrigin(d):
    sum_i = np.sum([x**2 - 10*np.cos(2 * np.pi * x) for x in d])
    return 10 * len(d) + sum_i

## Funcion  rosenbrock,  Typical  coord_min = [-3, -3] , coord_max = [3, 3] , optimum at  [0, 0]
def rosenbrock(d, a=1, b=100):
    return (a - d[0])**2 + b * (d[1] - d[0]**2)**2

## Funcion  schwefel, Typical coord_min = [-500, -500] , coord_max = [500, 500] , optimum at [420.968, 420.968]
def schwefel(x):
    val = 0
    d = len(x)
    for i in range(d):
        val += x[i] * math.sin(math.sqrt(abs(x[i])))
    val = 418.9829 * d - val
    return val
