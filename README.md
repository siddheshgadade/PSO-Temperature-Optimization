# 🧬 Particle Swarm Optimization (PSO) from Scratch

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/Library-NumPy-013243.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Library-Pandas-150458.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Library-Matplotlib-ffffff.svg)](https://matplotlib.org/)

An object-oriented, ground-up implementation of the **Particle Swarm Optimization (PSO)** algorithm applied to historical climate data. This project demonstrates the mathematical mechanics of swarm intelligence to minimize Mean Squared Error (MSE) in time-series temperature predictions.

---

## 🧠 Algorithmic Implementation

Instead of relying on pre-built Machine Learning libraries like `scikit-learn`, this project implements the core mathematics of PSO natively in Python.

### The Swarm Mechanics (`Particle` Class)
The algorithm simulates a swarm of particles moving through a multi-dimensional search space to find an optimal solution (the target temperature array). Each particle updates its trajectory based on three factors:
1. **Inertia Weight (`w`):** Maintains the particle's current momentum (decays over time to encourage local fine-tuning).
2. **Cognitive Parameter (`c1`):** The particle's tendency to return to its own historical best position.
3. **Social Parameter (`c2`):** The particle's attraction to the swarm's overall global best position.

**Fitness Function:** Evaluated using the Sum of Squared Errors (SSE) between the particle's position and the actual historical data.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.x
- NumPy, Pandas, Matplotlib

### Execution
1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR-USERNAME/PSO-Temperature-Optimization.git](https://github.com/YOUR-USERNAME/PSO-Temperature-Optimization.git)
