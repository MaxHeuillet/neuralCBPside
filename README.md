# Neural active learning meets the partial monitoring framework

This repository contains the implementation of algorithms described in the paper "Neural active learning meets the partial monitoring framework".

### Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.8
- pip

### Installation

Follow these steps to set up your environment and run the experiments:

1. **Create a Virtual Environment**:
   ```bash
   python -m venv env
   source env/bin/activate  

2. **Install Dependencies**:

   ```bash 
   pip install -r requirements.txt
   ```

#### Installation Troubleshooting:

- **Gurobi Alternative**: If you prefer not to use Gurobi, you can use PULP as an alternative optimizer. To do this, install PULP using pip install pulp. We provide code 'geometry_gurobi.py' and 'geometry_pulp.py'.

### Running Experiments

Refer to the 'experiments.ipynb' file. 

### Acknowledgements

Special thanks to Yikun Ban, Yuheng Zhang for the open source implementations neural active learning baselines. 
The codebase also leveraged and adapted game environments from Tanguy Urvoy's pmlib (https://github.com/TanguyUrvoy/pmlib).


