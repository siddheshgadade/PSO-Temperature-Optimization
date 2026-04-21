import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset (Reads directly from the GitHub repository folder)
data = pd.read_csv("TEMP_ANNUAL_SEASONAL_MEAN.csv")
data['ANNUAL'] = pd.to_numeric(data['ANNUAL'], errors='coerce')
clean_data = data.dropna(subset=['ANNUAL'])

# PSO Algorithm
class Particle:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity
        self.best_position = position.copy()
        self.best_fitness = float('inf')

    def update(self, global_best_position, w, c1, c2):
        r1, r2 = np.random.random(2)
        self.velocity = (w * self.velocity +
                         c1 * r1 * (self.best_position - self.position) +
                         c2 * r2 * (global_best_position - self.position))
        self.position += self.velocity

def fitness_function(position, target):
    return np.sum((position - target)**2)

def PSO(target, num_particles=100, max_iter=500, w_start=0.9, w_end=0.4, c1=1.5, c2=1.5):
    particles = [Particle(np.random.uniform(20, 30, len(target)),
                          np.random.uniform(-1, 1, len(target)))
                 for _ in range(num_particles)]

    global_best_position, global_best_fitness = np.random.uniform(20, 30, len(target)), float('inf')
    fitness_history, particle_positions = [], []

    for iteration in range(max_iter):
        w = w_start - (w_start - w_end) * (iteration / max_iter)
        particle_positions.append([p.position.copy() for p in particles])

        for p in particles:
            fitness = fitness_function(p.position, target)
            if fitness < p.best_fitness:
                p.best_fitness = fitness
                p.best_position = p.position.copy()
            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_position = p.position.copy()

        for p in particles:
            p.update(global_best_position, w, c1, c2)

        fitness_history.append(global_best_fitness)

    return global_best_position, global_best_fitness, fitness_history, particle_positions

# Run PSO and plot results
target_temperatures = clean_data['ANNUAL'].to_numpy()
years = clean_data['YEAR'].to_numpy()

best_position, best_fitness, fitness_history, particle_positions = PSO(target_temperatures)

# Fitness history plot
plt.plot(fitness_history)
plt.title('Fitness Optimization using PSO')
plt.xlabel('Iteration')
plt.ylabel('Best Fitness')
plt.grid(True)
plt.show()

# Actual vs Predicted Temperatures plot
plt.plot(years, target_temperatures, label='Actual Temperatures', marker='o')
plt.plot(years, best_position, label='Predicted Temperatures', linestyle='--', marker='x')
plt.title('Actual vs Predicted Temperatures')
plt.xlabel('Year')
plt.ylabel('Temperature')
plt.legend()
plt.grid(True)
plt.show()

print(f"Best Position (Predicted Temperatures): {best_position}")
print(f"Best Fitness: {best_fitness}")
print(f"Normalized Fitness (Mean Squared Error): {best_fitness / len(target_temperatures)}")
