# SAT Solvers with Local Search

Advanced implementations of GWSAT and WalkSAT algorithms for solving Boolean satisfiability (SAT) problems. Features include Tabu search enhancement, comprehensive performance analysis, and visualization capabilities.

**Original Implementation: 2019**  
**Tested with Python 3.12**

## Key Features

### 🎯 Core Highlights
- **Two solver implementations**: GWSAT and WalkSAT with Tabu search
- **Stochastic local search**: Escape local optima through controlled randomness
- **Tabu search enhancement**: Prevents cycling in WalkSAT implementation
- **Performance analytics**: Detailed runtime statistics and iteration tracking
- **Visualization support**: Iteration distribution histograms and convergence plots

## Performance Results

### GWSAT Performance (uf20 instances - 20 variables, 91 clauses)
- **Success rate**: 100% with optimal parameters
- **Average runtime**: ~0.007s for uf20-01
- **Optimal configuration**: wp=0.4 (60% GSAT, 40% random walk)
- **Key finding**: Balance between greedy and random selection critical

### WalkSAT with Tabu Performance
- **Success rate**: 100% on uf20 instances
- **Average runtime**: ~0.02s with Tabu list
- **Tabu list impact**: Prevents revisiting recent variable flips
- **Adaptive mechanism**: Disables Tabu when only 1 clause remains unsatisfied

### Scalability (uf50 instances - 50 variables, 218 clauses)
- **GWSAT**: 100% success with increased iterations (1000)
- **WalkSAT**: 20% success rate, requires parameter tuning
- **Runtime increase**: ~150x compared to uf20 instances

## Project Structure

```
├── gwsat_solver.py         # GWSAT implementation
├── walksat_solver.py       # WalkSAT with Tabu search
├── uf20-01.cnf            # Sample SAT instance (20 vars)
├── uf20-02.cnf            # Sample SAT instance (20 vars)
├── uf50-01.cnf            # Larger SAT instance (50 vars)
├── requirements.txt        # Dependencies
└── gitignore/             # Non-git files (PDFs, docs)
```

## Implementation Details

### 1. GWSAT Algorithm (`gwsat_solver.py`)
Combines GSAT hill-climbing with random walk:
- **Greedy mode**: Selects variable that maximizes satisfied clauses
- **Random walk**: Randomly selects from unsatisfied clause variables
- **Walk probability**: Controls exploration vs exploitation balance

Key innovations:
- Efficient clause evaluation using dictionaries
- Dynamic tracking of literal frequencies in unsatisfied clauses
- Configurable restart mechanism for escaping local optima

### 2. WalkSAT with Tabu Search (`walksat_solver.py`)
Enhanced WalkSAT with memory mechanism:
- **Zero-gain priority**: First attempts flips with no negative impact
- **Minimum damage**: Selects variable with least clause breakage
- **Tabu list**: Prevents recently flipped variables from re-selection
- **Adaptive Tabu**: Automatically disables when approaching solution

Tabu implementation features:
- Configurable list length (default: 5)
- FIFO queue structure using deque
- Failsafe mechanisms for edge cases
- Temporary disable when blocked

## Usage

### Command Line Interface

Both solvers support command-line arguments:

```bash
# GWSAT Usage
python gwsat_solver.py <cnf_file> <executions> <max_iterations> <max_restarts> <walk_probability>

# Example
python gwsat_solver.py uf20-01.cnf 100 1000 10 0.4

# WalkSAT Usage
python walksat_solver.py <cnf_file> <executions> <max_iterations> <max_restarts> <probability> <tabu_length>

# Example
python walksat_solver.py uf20-01.cnf 100 1000 10 0.4 5
```

### Parameters Explained

**GWSAT Parameters:**
- `cnf_file`: Input file in DIMACS CNF format
- `executions`: Number of independent runs
- `max_iterations`: Maximum flips per restart
- `max_restarts`: Maximum random restarts
- `walk_probability`: Probability of random walk (0-1)

**WalkSAT Parameters:**
- Same as GWSAT, plus:
- `probability`: Random selection probability (0-1)
- `tabu_length`: Size of Tabu list (0 to disable)

### Direct Execution

For quick testing with default parameters:
```python
# In each file, set:
use_console_args = False

# Then run:
python gwsat_solver.py
python walksat_solver.py
```

## Configuration Options

Both solvers include extensive configuration flags:

```python
# Debug and Analysis
check_solution = True        # Verify solution correctness
enable_diagnostics = True    # Collect performance metrics
console_gwsat_debug = False  # Step-by-step algorithm trace
console_solution_info = False # Display solution details
```

## Key Insights

### Algorithm Comparison
- **GWSAT**: Better for smaller instances, consistent performance
- **WalkSAT**: Faster when tuned correctly, benefits from Tabu enhancement
- **Scaling**: Both algorithms struggle with larger instances without parameter tuning

### Optimal Parameter Settings
**For uf20 instances:**
- GWSAT: wp=0.4, iterations=1000
- WalkSAT: p=0.4, tabu=5, iterations=1000

**For uf50 instances:**
- Increase iterations to 10000+
- Adjust walk/random probability based on instance
- Consider multiple restart strategies

### Performance Analysis Features
- Runtime statistics (mean, median, std deviation)
- Iteration distribution histograms
- Convergence visualization
- Restart frequency tracking

## Requirements

- Python 3.7+ (tested with Python 3.12)
- matplotlib (for visualization)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Technical Notes

- CNF parser handles DIMACS format
- Efficient clause evaluation using set operations
- Memory-efficient variable tracking
- Comprehensive error handling and validation
- Modular design for easy algorithm extension

## Experiments and Analysis

The implementation includes extensive experimentation capabilities:
- Automated parameter sweep testing
- Statistical analysis of solution quality
- Performance comparison across instances
- Visualization of algorithm behavior

Results show that the balance between exploration and exploitation is critical for both algorithms, with instance-specific tuning yielding significant improvements.

## Author

[sebieire](https://github.com/sebieire/)

## License

MIT License - See LICENSE file for details