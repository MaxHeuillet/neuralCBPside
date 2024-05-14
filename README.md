# NeuralUCB
This repository contains our pytorch implementation of NeuralCBP in the paper ''Neural Active Learning meets the Partial Monitoring Framework'' (accepted by UAI 2024). 

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


## Launch experiment on the cluster:

The number of seeds is an additional parameter, for example to run the experiment on 25 seeds:

```bash
  bash benchmark_meta.sh 25
```

## Launch one specific experiment:

```bash
python3 ./benchmark2.py --case 'case1' --model 'MLP' --horizon 9999 --n_folds 25 --approach 'NeuralCBPside' --context_type 'MNISTbinary' --id 0
```

The variable case takes values in: case1 (i.e. binary with uniform costs), case1b (i.e. binary with FP-sensitive costs), case2 (i.e. multiclass with uniform costs).

The variable mode takes values in: MLP, LeNet. The horizon is the number of rounds of the experiments. 

The number of folds corresponds to the number of runs. The approach variable takes values in: NeuralCBPside, neuronal6, neuronal3, ineural3, ineural6, margin, cesa. 

The variable context type takes values in: MNISTbinary, adult, MagicTelescope, MNIST, FASHION, covertype, shuttle. 

The variable id corresponds to the seed number chosen for the experiment.

