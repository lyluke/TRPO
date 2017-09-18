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
main.py [-h] [--init_params_path INIT_PARAMS_PATH]
               [--hidden_sizes HIDDEN_SIZES] [--init_std INIT_STD]
               [--min_std MIN_STD] [--hidden_activation HIDDEN_ACTIVATION]
               [--output_activation OUTPUT_ACTIVATION]
               [--max_global_iters MAX_GLOBAL_ITERS] [--batch_size BATCH_SIZE]
               [--max_path_length MAX_PATH_LENGTH] [--discount DISCOUNT]
               [--log_file_name LOG_FILE_NAME] [--plot PLOT]
               [--save_period SAVE_PERIOD]
               [--over_sampling_ratio OVER_SAMPLING_RATIO] [--alpha ALPHA]
               [--beta_zero BETA_ZERO] [--seeds SEEDS]
               [--predict_size PREDICT_SIZE] [--log_std_tol LOG_STD_TOL]
               [--verbose VERBOSE] [--num_gpu_devices NUM_GPU_DEVICES]
               [--vanilla VANILLA]

Implement multi-process TRPO in tensorflow, solving zero-order optimization
problem

optional arguments:
  -h, --help            show this help message and exit
  --init_params_path INIT_PARAMS_PATH
                        the path to saved initial parameters
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
  --max_global_iters MAX_GLOBAL_ITERS
                        the max number of total iterations of all the worker
                        processes
  --batch_size BATCH_SIZE
                        the batch size for each worker used to train
  --max_path_length MAX_PATH_LENGTH
                        the maximum length of each path
  --discount DISCOUNT   discount factor when calculate accumulative rewards
  --log_file_name LOG_FILE_NAME
                        the file name of the log recording each workers
                        activity
  --plot PLOT           the flag indicate to save sampled path or not
  --save_period SAVE_PERIOD
                        the period of global iterations in which worker_0 will
                        save parameters. If plot is set to be true, a sampled
                        path will be saved into a pickled binary file
  --over_sampling_ratio OVER_SAMPLING_RATIO
                        the ratio over the batch_size that data sampler will
                        sampled by
  --alpha ALPHA         the parameter alpha of the exponential distribution
                        that prioritized experience will use
  --beta_zero BETA_ZERO
                        the parameter beta of the importance sampling
                        annealing that prioritized experience will use
  --seeds SEEDS         the random seeds that each worker will recieve, format
                        should be s1,s2,...,sn
  --predict_size PREDICT_SIZE
                        the batch_size that each worker will sampled when
                        predicting the final minimum point
  --log_std_tol LOG_STD_TOL
                        the tolerance of the log standard deviation that
                        gaussian sampling will be parametrized. A too samll
                        value can increase the convergence rate of the policy,
                        leadint to a finer result. However, in this case, the
                        training process will be controlled by
                        max_global_iters. A too large value will lead to an
                        inaccurate result
  --verbose VERBOSE     verbose level of logging. 0 only plot basic info, 1
                        will plot specific training info, 2 will plot detailed
                        policy convergence data
  --num_gpu_devices NUM_GPU_DEVICES
                        The number of gpu devices in your mahcine. I have
                        tried cpu implementation but it is way too slow.
  --vanilla VANILLA     Use vanilla TRPO or prioritized experience TRPO.
                        Vanilla TRPO usually has higher convergence stability
                        but easily get stuck in local minimum.
```
4. The result will be logged in the log file you defined. The default is tf_multiprocess.log. The end of the lines will state
