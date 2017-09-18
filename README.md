# TRPO_optimizer
This is a repository for the source codes of zero-order function optimizer based on reinforcement learning. Detailed explanation can be found in Introudction.ipynb.

## Usage:
1. Define your objective function in Env_example.py. In Env_example.py, a Michalewicz function is used as an example. This is a commonly used function for optimization algoritm testing. Your own defined function must support vector operation in numpy. In reset() method, define the starting point for each iteration. For Michalewicz function, since its input range is 0 to pi, I choose pi/2 as the starting point.

2. Run master.py to initiate policy proposal network and save initial parameters.
```
master.py [-h] [--init_params_path INIT_PARAMS_PATH]
                 [--hidden_sizes HIDDEN_SIZES] [--init_std INIT_STD]
                 [--min_std MIN_STD] [--hidden_activation HIDDEN_ACTIVATION]
                 [--output_activation OUTPUT_ACTIVATION] [--seed SEED]

Implement master network, initialize all the parameters and save it. Every
worker will load the initial parameters from disk

optional arguments:
  -h, --help            show this help message and exit
  --init_params_path INIT_PARAMS_PATH
                        the path to save initial parameters
  --hidden_sizes HIDDEN_SIZES
                        the size of hidden layer of the policy network format
                        should be l1,l2, ... ,ln
  --init_std INIT_STD   the initial value of std dev used by gaussian sampling
                        function
  --min_std MIN_STD     the minimum value std dev
  --hidden_activation HIDDEN_ACTIVATION
                        the activation used by hidden layer
  --output_activation OUTPUT_ACTIVATION
                        the activation used by output layer
  --seed SEED           the random seed to initialize network
```

3. Run main.py
```
```
4. The result will be logged in the log file you defined. The default is tf_multiprocess.log. The end of the lines will state
