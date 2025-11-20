# Probabilities in the Sky — Markov Chain Weather Simulator

An interactive learning tool demonstrating weather prediction using Markov Chains. Supports interactive UIs (Streamlit and Tkinter), command-line simulations, configurable transition probabilities, and visualizations.

## Features
- Sunny/Cloudy/Rainy weather states (extensible)
- Custom transition matrix with validation
- Multi-day simulation from an initial state
- Visualization: state trajectory, state distribution over time, transition heatmaps
- Interfaces:
  - Streamlit web app — easiest to use
  - Tkinter desktop app — runs locally without a browser
  - CLI — quick simulations and saving plots

## Project Structure
```
probabilities-in-the-sky/
├── probabilities_in_the_sky/
│   ├── __init__.py
│   ├── cli.py
│   ├── models/
│   │   └── markov_chain.py
│   ├── simulation/
│   │   └── simulator.py
│   └── viz/
│       └── visualize.py
├── ui/
│   ├── streamlit_app.py
│   └── tkinter_app.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Quick Start
### 1) Run the Streamlit app
```bash
streamlit run ui/streamlit_app.py
```

### 2) Run the Tkinter desktop app
```bash
python ui/tkinter_app.py
```

### 3) Run a CLI simulation
```bash
python -m probabilities_in_the_sky.cli \
  --states Sunny Cloudy Rainy \
  --matrix "0.7,0.2,0.1;0.3,0.5,0.2;0.2,0.3,0.5" \
  --initial Sunny \
  --days 30 \
  --plot out.png
```

## Usage Notes
- Transition matrix rows must each sum to 1.
- The order of states corresponds to rows/columns in the matrix.
- You can extend the states beyond Sunny/Cloudy/Rainy by adding labels and matching matrix dimensions.

## Development
Run unit-like smoke tests:
```bash
python - << 'PY'
from probabilities_in_the_sky.models.markov_chain import MarkovChain
from probabilities_in_the_sky.simulation.simulator import simulate_markov_chain

states=["Sunny","Cloudy","Rainy"]
P=[[0.7,0.2,0.1],[0.3,0.5,0.2],[0.2,0.3,0.5]]
mc=MarkovChain(states, P)
trajectory, counts = simulate_markov_chain(mc, initial_state="Sunny", days=10)
print(trajectory)
print(counts)
PY
```

## License
MIT


