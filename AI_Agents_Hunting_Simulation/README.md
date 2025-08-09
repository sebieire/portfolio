# AI Agents Hunting Simulation

A comprehensive implementation of intelligent agents operating in a 2D grid world environment, demonstrating various AI techniques including reflex agents, model-based agents, utility-based agents, search algorithms, and knowledge-based reasoning.

## Overview

This project simulates a hunting scenario where an intelligent agent (huntsman) navigates a 2D grid world to hunt different types of prey (deer and rabbits) while managing energy levels through rest points (huts). The implementation showcases different AI agent architectures and their comparative performance in achieving survival and hunting goals.

## Features

### Part 1: Intelligent Agent Types
- **Simple Reflex Agent**: Acts based on immediate percepts without memory
- **Model-Based Agent**: Maintains internal state to track visited locations and avoid repetition
- **Utility-Based Agent**: Makes decisions based on utility functions, maintains world knowledge, and plans actions based on energy levels

### Part 2: Search Algorithms
- **Breadth-First Search (BFS)**: Complete search using FIFO approach for optimal pathfinding
- **Depth-First Search (DFS)**: LIFO-based search exploring paths to maximum depth
- **Depth-Limited Search (DLS)**: Constrained DFS with configurable depth limit

### Part 3: Knowledge-Based Reasoning
- **Forward Chaining**: Implements logical inference to deduce deer locations from track patterns
- **Knowledge Base Management**: Dynamic clause generation and entailment checking
- **Goal-State Inference**: Achieves complex goals through logical reasoning

## Performance Highlights

Based on 100 simulation runs in a 10x10 grid with high density:

### Agent Comparison
- **Simple Reflex Agent**: 71% survival rate, basic performance (mostly 10-20 points)
- **Model-Based Agent**: 82% survival rate, balanced performance (50%+ achieving 20+ points)
- **Utility-Based Agent**: 100% survival rate, excellent performance (50%+ achieving 70+ points)

### Search Algorithm Efficiency
- **BFS**: 104 nodes expanded, 5 steps to goal (optimal)
- **DFS**: 156 nodes expanded, 29 steps to goal
- **DLS (limit=10)**: 704 nodes expanded, 10 steps to goal

## Project Structure

```
.
├── part1_agent_simulation.py      # Main file for agent simulations
├── part2_search_algorithms.py     # Search algorithm implementations
├── part3_knowledge_reasoning.py   # Knowledge-based reasoning demo
├── ai_agents_notebook.ipynb       # Jupyter notebook with detailed analysis
├── lib_part1/                     # Agent variant implementations
│   ├── agent_variants.py         # Simple reflex, model-based, utility agents
│   ├── huntsman.py               # Hunter agent class
│   ├── thing_objects.py         # Game objects (Rabbit, Deer, Hut)
│   ├── world.py                  # Environment implementation
│   └── statistical_data_functions.py  # Data analysis utilities
├── lib_part2/                     # Search algorithm components
│   ├── search_functions.py      # BFS, DFS, DLS implementations
│   ├── search_classes.py        # Problem formulation and graph structures
│   └── problem_statements_advanced.py  # Advanced search scenarios
└── lib_part3/                     # Knowledge reasoning components
    ├── aima_KB_logic.py         # Knowledge base and inference engine
    └── agent_variants.py        # Knowledge-based agent implementation
```

## Usage

### Running Agent Simulations (Part 1)

```python
from part1_agent_simulation import *

# Configure simulation parameters
agent = simple_reflex_agent  # or model_based_agent, utility_based_agent
clone = True  # Use same world for each run
world_x = 10
world_y = 10
stepsPerGame = 100
runs = 100
critter_density = 'high'  # options: high, medium, low
hut_density = 'high'

# Initialize and run
settings.init_settings(agent, clone, world_x, world_y, stepsPerGame, runs, critter_density, hut_density)
settings.init_statistics_data()
launch_world(settings.number_of_runs)

# Display results
output_general_statistics()
show_heatmap()
```

### Running Search Algorithms (Part 2)

```python
from part2_search_algorithms import *

# Configure search type
settings.init_settings(search_agent, 'BFS')  # Options: 'BFS', 'DFS', 'DLS'
# For DLS, add depth limit: settings.init_settings(search_agent, 'DLS', 10)

settings.init_statistics_data()
launch_world(settings.number_of_runs)
output_search_statistics()
show_heatmap()
```

### Running Knowledge-Based Reasoning (Part 3)

```python
from part3_knowledge_reasoning import *

# Configure reasoning simulation
maxGameSteps = 500
settings.init_settings(random_movement_agent, gameSteps=maxGameSteps)
settings.init_statistics_data()

# Run simulation
world = launch_world(settings.number_of_runs)
output_general_statistics()
show_heatmap()

# View accumulated knowledge
print(world.agent_knowledge_base.clauses)
```

## Technical Details

### World Configuration
- **Grid Size**: Configurable (default 10x10)
- **Wraparound**: The world is toroidal (edges connect)
- **Energy System**: 
  - Movement: -3 energy
  - Rest: +20 energy per round
  - Hunt Deer: -8 energy, +10 performance
  - Hunt Rabbit: -5 energy, +3 performance

### Agent Decision Making
- **Percepts**: Current tile contents (prey/hut)
- **Actions**: Move (N/S/E/W), Hunt, Rest
- **Performance Metrics**: Survival rate, hunting score, energy management

### Search Problem Formulation
- **States**: Grid positions
- **Actions**: Four-directional movement
- **Goal Test**: Reaching hut location
- **Path Cost**: Number of steps taken

## Requirements

- Python 3.7+
- NumPy
- Matplotlib
- Seaborn
- Jupyter (optional, for notebook)

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy matplotlib seaborn jupyter
```

## Academic Context

This project was developed as part of an AI course assignment in 2019, demonstrating practical implementations of fundamental AI concepts including agent architectures, search strategies, and knowledge representation.

## Author

[sebieire](https://github.com/sebieire/)

## License

MIT License

---

*Note: This is an academic project demonstrating AI agent implementations and algorithms. The code has been refactored for portfolio presentation while maintaining the core algorithmic implementations.*