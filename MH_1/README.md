# TSP Genetic Algorithm Solver

A comprehensive implementation of various genetic algorithm operators for solving the Traveling Salesman Problem (TSP). This project explores different crossover methods, mutation strategies, and selection techniques to optimize route finding.

**Original Implementation: 2019**

## Features

### Crossover Operators
- **Partially Mapped Crossover (PMX)** - Advanced crossover preserving relative order
- **Uniform Order-Based Crossover** - Maintains position-based inheritance
- **Order-1 Crossover** - Classic TSP crossover operator

### Mutation Operators
- **Inversion Mutation** - Reverses segments of the tour
- **Reciprocal Exchange** - Swaps two cities in the tour

### Selection Methods
- **Stochastic Universal Sampling (SUS)** - Low-bias selection technique
- **Random Selection** - Uniform random parent selection
- **Elitist Strategy** - Preserves top-performing solutions (configurable)

### Initialization Strategies
- **Random Generation** - Completely randomized initial population
- **Nearest Neighbor Heuristic** - Greedy approach for initial solutions

## Performance Highlights

The implementation includes 8 pre-configured algorithm combinations that achieved notable results in benchmark tests:
- Best performing configuration achieved approximately **16.8** fitness value on 128-city problems
- Stochastic Universal Sampling showed **1-2% improvement** over random selection
- Early convergence detection with most improvements occurring in first 100 iterations

## Usage

### Basic Run
```bash
python tsp_solver.py sample_data.tsp
```

### Configuration Options

The solver supports multiple pre-configured algorithm combinations:

| Config | Initialization | Crossover | Mutation | Selection |
|--------|---------------|-----------|----------|-----------|
| 1 | Random | Uniform | Inversion | Random |
| 2 | Random | PMX | Reciprocal | Random |
| 3 | Random | Uniform | Reciprocal | SUS |
| 4 | Random | PMX | Reciprocal | SUS |
| 5 | Random | PMX | Inversion | SUS |
| 6 | Random | Uniform | Inversion | SUS |
| 7 | Heuristic | PMX | Reciprocal | SUS |
| 8 | Heuristic | Uniform | Inversion | SUS |

To run a specific configuration, modify the flags in `tsp_solver.py`:
```python
run_configuration = True
config_number = 4  # Choose 1-8
```

### Parameters

Default parameters (customizable in code):
- **Population Size**: 300
- **Mutation Rate**: 0.1
- **Iterations**: 500
- **Elite Size**: 20 (when enabled)

## File Structure

```
├── tsp_solver.py      # Main GA implementation
├── individual.py      # Individual/chromosome class
├── sample_data.tsp    # Sample TSP instance (128 cities)
└── requirements.txt   # Dependencies (uses standard library only)
```

## Data Format

The TSP data file format:
- First line: Number of cities
- Following lines: `city_id x_coordinate y_coordinate`

*Note: The sample TSP instance file was provided by Dr. Diarmuid Grimes*

## Algorithm Details

### Fitness Function
Uses Euclidean distance to calculate total tour length (minimization problem)

### Convergence Analysis
The implementation tracks fitness improvements across iterations, showing rapid initial convergence with refinements in later stages.

## Requirements

- Python 3.7+ (tested with Python 3.12)
- No external dependencies - uses only Python standard library

## Installation

```bash
# Clone the repository
git clone [repository-url]

# No installation needed - uses standard library only
python tsp_solver.py sample_data.tsp
```

## Experimental Features

- **Experiment Runner**: Run multiple configurations with statistical analysis
- **Debug Flags**: Extensive debugging options for algorithm analysis
- **Elite Survival**: Optional preservation of best solutions across generations

## Notes

This implementation was developed as an exploration of metaheuristic optimization techniques, specifically focusing on the effectiveness of different GA operators for combinatorial optimization problems.

## License

MIT License - See LICENSE file for details