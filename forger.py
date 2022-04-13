import random
import numpy as np
from tqdm import tqdm
import myopencvutil

# Global variables
MIN_CHOICE = 0
MAX_CHOICE = 8

# A list of lambda functions to optimize direction finding
ST = lambda x,y: (x,y)
UL = lambda x,y: (x-1, y-1)
UU = lambda x,y: (x-1, y)
UR = lambda x,y: (x-1, y+1)
RR = lambda x,y: (x, y+1)
DR = lambda x,y: (x+1, y+1)
DD = lambda x,y: (x+1, y)
DL = lambda x,y: (x+1, y-1)
LL = lambda x,y: (x, y-1)

DIRECTION_FINDERS = [ST, UL, UU, UR, RR, DR, DD, DL, LL]

class Forger:
    def __init__(self, filename: str, population_size: int = 5000, generation_num: int = 1000, parent_num: int = 50):
        # set the initialization parameters
        self.population_size = population_size
        self.generation_num = generation_num
        self.parent_num = parent_num
        self.offspring_size = population_size - parent_num

        # set the target image
        self.target_img = myopencvutil.read_binary_img(filename=filename)
        self.target_shape = self.target_img.shape

        path_bounds = calc_path_len(self.target_img)
        self.path_len = path_bounds['max']

        # do the 1s and 0s calculations to avoid re-calculation
        self.target_ones = np.count_nonzero(self.target_img)
        self.target_zeros = self.target_img.size - self.target_ones
        self.ones_ratio = self.target_ones / self.target_img.size
        self.zeros_ratio = self.target_zeros / self.target_img.size

        # init population
        # TODO: use a numpy array instead of list?
        self.population = [generate_random_path(path_bounds['max'], path_bounds['min']) for _ in range(population_size)]
        self.population.sort(reverse = True, key = self.fitness)
    
    
    def execute_path(self, path):
        """
        returns a numpy array with `path` executed on it
        turning each element passed through to True
        """
        blank = np.zeros(self.target_shape, dtype=bool)
        x, y = blank.shape[0]- 1, 0
        blank[x][y] = True
        for s in path:
            x_new, y_new = DIRECTION_FINDERS[s](x, y)
            try:
                blank[x_new][y_new] = True
                x, y = x_new, y_new
            except IndexError:
                continue
        return blank


    def similarity_score(self, src) -> float:
        """
        calculates a score for similarity between the given arrays
        `src` and `self.target_img`

        :param src: the array to compare with `self.target_img`
        """
        
        # count the ones and zeros in the target image
        c1 = self.ones_ratio
        c2 = self.zeros_ratio
        
        # count the ones and zeros matching the target
        matching_ones = np.count_nonzero((src == True) & (src == self.target_img))
        matching_zeros = np.count_nonzero((src == False) & (src == self.target_img))

        return c1 * matching_ones /  self.target_ones + c2 * matching_zeros / self.target_zeros

    # FIXME: design a function that behaves reasonably
    def turning_penalty(self, path) -> float:
        # The absolute differences between consecutive elements of an array
        diffs = np.abs(np.ediff1d(path)).sum()
        return 1 / (diffs + 1)
    

    def fitness(self, path) -> float:
        result_img = self.execute_path(path)
        return self.similarity_score(result_img)


    def crossover(self, parents):

        # crossover index
        c_idx = np.uint8(self.offspring_size/2)
        
        # the best parents mating is not always the best option
        np.random.shuffle(parents)
        num_parents = len(parents)
        
        offspring = [np.concatenate((parents[k % num_parents][0: c_idx], parents[(k+1) % num_parents][c_idx:])) for k in range(self.offspring_size)]
        
        return offspring
    

    def mutate(self, offspring, mu: float = 0.05, max_change: int = 3):
        for indiv_idx in range(len(offspring)):
            indices = random.sample(range(self.path_len), int(self.path_len * mu) + 1)
            for step_idx in indices:
                r = (offspring[indiv_idx][step_idx] + np.random.randint(0, max_change)) % (MAX_CHOICE + 1)
                offspring[indiv_idx][step_idx] = r
        return offspring


    def fabricate(self):
        pbar = tqdm(total=self.generation_num)
        for _ in range(self.generation_num):
            parents = self.population[:self.parent_num]

            # remove the steps where we decided to stop `False`
            for i in range(len(parents)):
                parents[i][ parents[i] == True ].resize(self.path_len)
            
            offspring = self.crossover(parents)
            offspring = self.mutate(offspring, mu = 0.05)
            self.population = parents + offspring
            self.population.sort(reverse = True, key = lambda p: self.fitness(p))
            img = self.execute_path(self.population[0])
            myopencvutil.plot_image(img)
            pbar.update(1)
        pbar.close()


        
def generate_random_path(size, min_steps):
    path_start = np.random.randint(MIN_CHOICE + 1, MAX_CHOICE, size = min_steps)
    path_rest = np.random.randint(MIN_CHOICE, MAX_CHOICE, size = size - min_steps)
    return np.concatenate((path_start, path_rest))

def calc_path_len(tgt):
    """
    calculates the maximum and the minimum path length possible
    to draw the image
    returns a dict with keys `max` and `min`
    """
    ones = np.count_nonzero(tgt)
    diag = max(tgt.shape)
    
    # the legth of the path is the black pixels + the max
    # steps needed to reach start position
    return {
        'max': min(ones * 2 + diag - 3, tgt.size),
        'min': ones
    }