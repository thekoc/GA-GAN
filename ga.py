import numpy as np
from bresenham import bresenham
import itertools

class GeneticAlgorithm:
    def __init__(self, population: np.ndarray, offspring_num, crossover_probability, mutation_probability):
        self.population = population
        self.fitness = np.zeros(len(population))
        self.offspring_num = offspring_num
        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability
        self._parent_generator = self.parent_generator()
    
    def calculate_fitness(self, population):
        raise NotImplementedError()
    
    def select_best(self, num):
        indices = np.argsort(self.fitness)[-num:]
        return self.population[indices]

    def parent_generator1(self):
        while True:
            f_max = np.max(self.fitness)
            f_min = np.min(self.fitness)
            if f_max != f_min:
                p = (self.fitness - f_min) / (f_max - f_min) + 0.2
                assert np.sum(p) != 0
                p = p / np.sum(p)
            else:
                p = np.ones(len(self.fitness)) / len(self.fitness)
                assert len(self.fitness) != 0
            # print(p)
            indices = np.random.choice(np.arange(len(self.population)), 2, p=p, replace=False)
            yield self.population[indices]
    
    def parent_generator(self):
        # tournament
        while True:
            tournament_size = 10
            indices = np.random.choice(np.arange(len(self.fitness)), tournament_size, replace=False)
            tournament_fitness = self.fitness[indices]
            fittest_index_in_tournament = np.argsort(tournament_fitness)[-1]
            i1 = indices[fittest_index_in_tournament]
            

            indices = np.random.choice(np.arange(len(self.fitness)), tournament_size, replace=False)
            tournament_fitness = self.fitness[indices]
            fittest_index_in_tournament = np.argsort(tournament_fitness)[-1]
            i2 = indices[fittest_index_in_tournament]

            yield self.population[[i1, i2]]
    
    def next_parents(self, restart=False):
        if restart:
            self._parent_generator = self.parent_generator()
        return next(self._parent_generator)
    
    
    def breed(self, generation=1):
        for _ in range(generation):
            offspring_num = 0
            offspring = []
            self.fitness = self.calculate_fitness(self.population)
            parents = self.next_parents(restart=True)
            offspring += list(self.select_best(3))
            while offspring_num < self.offspring_num:
                if np.random.random() < self.crossover_probability:
                    offspring += list(self.crossover(parents))
                else:
                    offspring += list(parents)
                offspring_num = len(offspring)
                parents = self.next_parents()

                self.fitness = self.calculate_fitness(self.population)
                assert self.fitness.ndim == 1

            offspring = np.array(offspring)
            self.mutate(offspring, self.mutation_probability)
            self.population = np.array(offspring)
            print(np.max(self.fitness))
        return self.population
    
    def crossover(self, parents):
        raise NotImplementedError()

    def mutate(self, polulation, probability=0.1):
        raise NotImplementedError()
        
class PixelGA(GeneticAlgorithm):
    def crossover(self, parents):

        offspring = []
        p1 = np.array(parents[0])
        p2 = np.array(parents[1])
        crossover_point = np.random.randint(1, len(p1) - 1)
        assert crossover_point > 0
        t = np.array(p2[0:crossover_point])
        p2[0:crossover_point] = p1[0:crossover_point]
        p1[0:crossover_point] = t
        offspring.append(p1)
        offspring.append(p2)

        return offspring


    def mutate(self, population, probability=0.1):
        """mutate in place"""
        p_size = population.size
        draw_num = int(p_size * probability)
        mutate_indices = np.random.choice(np.arange(0, p_size), draw_num)
        indices = np.unravel_index(mutate_indices, population.shape)
        population[indices] = 1 - population[indices]

            
            
            
    
    def in_pixel(self, population):
        return population

class LineGA(GeneticAlgorithm):
    def crossover(self, parents):
        offspring = []
        p1 = np.array(parents[0])
        p2 = np.array(parents[1])
        crossover_point = np.random.randint(1, len(p1) - 1)
        assert crossover_point > 0
        t = np.array(p2[0:crossover_point])
        p2[0:crossover_point] = p1[0:crossover_point]
        p1[0:crossover_point] = t
        offspring.append(p1)
        offspring.append(p2)
        return offspring

    def mutate(self, population, probability=0.1):
        for i in range(len(population)):
            lines = np.argwhere(population[i])
            for line in lines:
                if np.random.random_sample() <= 0.01:
                    population[i][tuple(line)] = False
                

            if np.random.random_sample() <= 0.3:
                x1 = np.random.choice(population[i].shape[1])
                y1 = np.random.choice(population[i].shape[0])
                x2 = np.random.choice(population[i].shape[3])
                y2 = np.random.choice(population[i].shape[2])
                population[i][y1][x1][y2][x2] = not population[i][y1][x1][y2][x2]

        
            lines = np.argwhere(population[i])
            for line in lines:
                if np.random.random_sample() <= 0.1:
                    population[i][tuple(line)] = False
                    offset = np.random.randint(-2, 2, 4)
                    newline = (line + offset) % 28
                    population[i][tuple(newline)] = True


    def in_pixel(self, population):
        pixel_population = np.zeros((len(population), 28, 28, 1))
        for i, gene in enumerate(population):
            lines = np.argwhere(gene)
            for line in lines:
                points = list(bresenham(*line))
                for point in points:
                    pixel_population[i][point[1]][point[0]][0] = 1
        return pixel_population



if __name__ == "__main__":
    # def calculate_fitness(population):
    #     return np.sum(population, axis=(1, 2, 3))
    # initial_population = np.random.randint(2, size=(1000, 10, 10, 1))
    # pga = PixelGA(initial_population, 1000, 0.8, 0.1)
    # pga.calculate_fitness = calculate_fitness
    # pga.breed(500)
    # print(pga.select_best(1))
    population_size = 256
    initial_population = np.zeros((population_size, 28, 28, 28, 28)).astype(np.bool)
    generator = LineGA(initial_population, population_size, 0.8, 0.1)
    fitness = np.random.rand(population_size)
    generator.calculate_fitness = lambda x: fitness
    generator.breed(10)