# NeuralUCB

This repository contains our pytorch implementation of NeuralCBP in the paper ''Neural Active Learning meets the Partial Monitoring Framework'' (accepted by UAI 2024). This branch contains the developpers code with slurm submission scripts.

## Prerequisites: 
```bash
  pip install -r requirements.txt
```

## Load datasets:

```bash
  python3 load_data.py
```

## Launch experiment on the cluster:

The number of seeds is an additional parameter, for example to run the experiment on 25 seeds:

```bash
  bash benchmark_meta.sh 25
```



## Launch one specific experiment:

```bash
python3 ./benchmark2.py --case 'case1' --model 'MLP' --horizon 9999 --n_folds 25 --approach 'NeuralCBPside' --context_type 'MNISTbinary' --id 0
```

- **case**: Variable that takes the following values:
  - `case1`: Binary with uniform costs
  - `case1b`: Binary with FP-sensitive costs
  - `case2`: Multiclass with uniform costs

- **mode**: Variable that takes the following values:
  - `MLP`: Multi-Layer Perceptron
  - `LeNet`: LeNet Architecture

- **horizon**: Refers to the number of rounds in the experiments.

- **number of folds**: Corresponds to the number of runs in the experiments.

- **approach**: Variable that takes the following values:
  - `NeuralCBPside`
  - `neuronal6`
  - `neuronal3`
  - `ineural3`
  - `ineural6`
  - `margin`
  - `cesa`

- **context type**: Variable that takes the following values:
  - `MNISTbinary`
  - `adult`
  - `MagicTelescope`
  - `MNIST`
  - `FASHION`
  - `covertype`
  - `shuttle`

- **id**: Corresponds to the seed number chosen for the experiment.


#### Installation Troubleshooting:

- **Gurobi Alternative**: If you prefer not to use Gurobi, you can use PULP as an alternative optimizer. To do this, install PULP using pip install pulp. We provide code 'geometry_gurobi.py' and 'geometry_pulp.py'.

### Running Experiments

Refer to the 'experiments.ipynb' file. 

### Acknowledgements

Special thanks to Yikun Ban, Yuheng Zhang for the open source implementations neural active learning baselines. 
The codebase also leveraged and adapted game environments from Tanguy Urvoy's pmlib (https://github.com/TanguyUrvoy/pmlib).



