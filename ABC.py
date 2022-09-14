import numpy as np

def calc_food_fx(food):
    x1, x2 = food
    return x1 ** 2 - x1 * x2 + x2 ** 2 + 2 * x1 + 4 * x2 + 3

def calc_food_fitness(f):
    if f >= 0:
        return 1 / (1 + f)
    return 1 + abs(f)

def calc_swarm_fx(food_source):
    fx = np.empty(food_source.shape[0])
    for index, food in enumerate(food_source):
        fx[index] =  calc_food_fx(food)
    return fx        
        
def calc_swarm_fitness(fx):
    fitness = np.empty(fx.shape)
    for index, f in enumerate(fx):
        fitness[index] = calc_food_fitness(f)     
    return fitness        
        
def employeePhase(food_source, fx, fitness, trial):
    
    new_food_source = np.empty(food_source.shape)
    new_fx = np.empty(fx.shape)
    new_fitness = np.empty(fitness.shape)
    
    for index, food in enumerate(food_source):
        
        x_index = np.random.randint(food_source.shape[1]) 
        partner_index = np.random.randint(food_source.shape[0])
        partner = food_source[partner_index]
        
        # x_new = x + phi * (x - x_partner)
        phi = np.random.random()
        food[x_index] = food[x_index] + phi * (food[x_index] - partner[x_index])
        
        # Scaling out of the range value to (-5, 5)
        if food[x_index] > 5.0:
            food[x_index] = 5.0
        if food[x_index] < -5.0:
            food[x_index] = -5.0
            
        food_fx = calc_food_fx(food)    
        food_fitness = calc_food_fitness(food_fx)
        if food_fitness < fitness[index]: # the new solution better than old solution
            new_food_source[index] = food
            new_fx[index] = calc_food_fx(food)
            new_fitness[index] = calc_food_fitness(food_fx)
            trial[index] = 0
            
        else: # the old solution better than the new
            
            new_food_source[index] = food_source[index]
            new_fx[index] = fx[index]
            new_fitness[index] = fitness[index]
            trial[index] += 1    
            
    return new_food_source, new_fx, new_fitness, trial
        
def onLookerPhase(food_source, fx, fitness, trial):
    
    np.seterr(invalid='ignore') # Supress/hide the warning
    probs = fitness / fitness.sum() # calculate the population probability
    
    new_food_source = np.empty(food_source.shape)
    new_fx = np.empty(fx.shape)
    new_fitness = np.empty(fitness.shape)
    
    for index, food in enumerate(food_source):
        
        r = np.random.random()
        if r > probs[index]:
            new_food_source[index] = food_source[index]
            new_fx[index] = fx[index]
            new_fitness[index] = fitness[index]
            continue
        
        x_index = np.random.randint(food_source.shape[1]) 
        partner_index = np.random.randint(food_source.shape[0])
        partner = food_source[partner_index]
        
        # x_new = x + phi * (x - x_partner)
        phi = np.random.random()
        food[x_index] = food[x_index] + phi * (food[x_index] - partner[x_index])
        
        # Scaling out of the range value to (-5, 5)
        if food[x_index] > 5.0:
            food[x_index] = 5.0 
        if food[x_index] < -5.0:
            food[x_index] = -5.0
            
        food_fx = calc_food_fx(food)    
        food_fitness = calc_food_fitness(food_fx)
        if food_fitness < fitness[index]: # the new solution better than old solution
            new_food_source[index] = food
            new_fx[index] = calc_food_fx(food)
            new_fitness[index] = calc_food_fitness(food_fx)
            trial[index] = 0
            
        else: # the old solution better than the new
            
            new_food_source[index] = food_source[index]
            new_fx[index] = fx[index]
            new_fitness[index] = fitness[index]
            trial[index] += 1    
            
    return new_food_source, new_fx, new_fitness, trial
        
        
def scoutPhase(food_source, fx, fitness, trial, scope, limit):
    
    (b, a) = scope #(-5, 5)
    
    for index in range(food_source.shape[0]):
        
        if trial[index] > limit:
            
            food_source[index] = (b - a) * np.random.random((1, food_source.shape[1])) + a
            fx[index] = calc_food_fx(food_source[index])
            fitness[index] = calc_food_fitness(fx[index])
            trial[index] = 0
            
    return food_source, fx, fitness, trial      
        
    
def ABC(swarm_size = 10, dim = 2, limit = 1, iteration = 20):
    # initialize the population
    (a, b) = (-5, 5)
    food_source = (b - a) * np.random.random((swarm_size, dim)) + a
    fx = np.empty(10)
    fitness = np.empty(10)
    trial = np.zeros(10)
    
    best_food = None
    best_fx = None
    
    for _ in range(iteration):
        
        food_source, fx, fitness, trial = employeePhase(food_source, fx, fitness, trial) # Employee Phase
        
        food_source, fx, fitness, trial = onLookerPhase(food_source, fx, fitness, trial) # OnLooker Phase
        
        food_source, fx, fitness, trial = scoutPhase(food_source, fx, fitness, trial, (a, b), limit) # Scout Phase
        
        # Memorize the best solution
        best_fx = fx.max()
        best_food = food_source[fx.argmax()]
        
    return best_food, best_fx 

# test case
food, fx = ABC(swarm_size = 10, dim = 2, limit = 1, iteration = 20)
print(f"the best solution is\nX = {food[0]}\nY = {food[1]}")
print(f"f(x) = x^2 - xy + y^2 + 2x + 4y + 3 = {fx}")