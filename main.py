from worker import Worker
import logging
import pickle
from multiprocessing import Process, Queue, Array, cpu_count
from time import sleep
from Env_example import Env
from baseline import LinearBaseline
import os
import numpy as np
import argparse
from tensorflow import nn
from tensorflow.python.client import device_lib

#load previously saved global initial parameters
def load_params(path):
    global_params = []
    with open(path,'rb') as f:
            gps = pickle.load(f)
    for gp in gps:
        global_params.append(Array('f', gp.flatten()))
    return global_params

def parse_args():
    parser = argparse.ArgumentParser(description='Implement multi-process TRPO\
                                     in tensorflow, solving zero-order \
                                     optimization problem')
    parser.add_argument('--init_params_path', type=str, default='global_vars.pickle',
                        help='the path to saved initial parameters')
    parser.add_argument('--hidden_sizes', type=str, default="64,64",
                        help='the size of hidden layer of the policy network\
                        format should be l1,l2, ... ,ln')
    parser.add_argument('--init_std', type=float, default=1.0,
                        help='the initial value of std dev used by gaussian\
                        sampling function')
    parser.add_argument('--hidden_activation', type=str, default="relu",
                        help='the activation used by hidden layer')
    parser.add_argument('--output_activation', type=str, default="tanh",
                        help='the activation used by output layer')
    parser.add_argument('--max_global_iters', type=int, default=501,
                        help='the max number of total iterations of all the\
                        worker processes')
    parser.add_argument('--batch_size', type=int, default=5000,
                        help='the batch size for each worker used to train')
    parser.add_argument('--max_path_length', type=int, default=10,
                        help='the maximum length of each path')
    parser.add_argument('--discount', type=float, default=0.0,
                        help='discount factor when calculate accumulative rewards')
    parser.add_argument('--log_file_name', type=str, default="tf_multiprocess.log",
                        help='the file name of the log recording each workers \
                        activity')
    parser.add_argument('--plot', type=int, default=0,
                        help='the flag indicate to save sampled path or not')
    parser.add_argument('--save_period', type=int, default=100,
                        help='the period of global iterations in which worker_0\
                        will save parameters. If plot is set to be true, a sampled\
                        path will be saved into a pickled binary file')
    parser.add_argument('--over_sampling_ratio', type=int, default=40,
                        help='the ratio over the batch_size that data sampler will sampled by')
    parser.add_argument('--alpha', type=float, default=1.35,
                        help='the parameter alpha of the exponential distribution\
                        that prioritized experience will use')
    parser.add_argument('--beta_zero', type=float, default=0.4,
                        help='the parameter beta of the importance sampling\
                        annealing that prioritized experience will use')
    parser.add_argument('--seeds', type=str, default="1018,1120,211,318",
                        help='the random seeds that each worker will recieve,\
                        format should be s1,s2,...,sn')
    parser.add_argument('--predict_size', type=int, default=10,
                        help='the batch_size that each worker will sampled when\
                        predicting the final minimum point')
    parser.add_argument('--log_std_tol', type=float, default=-5.0,
                        help='the tolerance of the log standard deviation that\
                        gaussian sampling will be parametrized. A too samll value\
                        can increase the convergence rate of the policy, leadint to\
                        a finer result. However, in this case, the training process\
                        will be controlled by max_global_iters. A too large value\
                        will lead to an inaccurate result')
    parser.add_argument('--verbose', type=int, default=0,
                        help='verbose level of logging. 0 only plot basic info, \
                        1 will plot specific training info, 2 will plot detailed \
                        policy convergence data')
    parser.add_argument('--num_gpu_devices', type=int, default=1,
                        help='The number of gpu devices in your mahcine. I have tried\
                        cpu implementation but it is way too slow.')
    parser.add_argument('--vanilla', type=int, default=0,
                        help='Use vanilla TRPO or prioritized experience TRPO. Vanilla TRPO usually has higher convergence stability but easily get stuck in local minimum.')
    return parser.parse_args()

def get_activation(act_type="relu"):
    if act_type == "relu":
        return nn.relu
    elif act_type == "tanh":
        return nn.tanh
    else:
        raise NotImplementedError

def getKey(path):
    return path[0]

if __name__ == "__main__":
    args = parse_args()
        
    #parsing some options
    init_path = args.init_params_path
    hidden_sizes_str = args.hidden_sizes.split(',')
    hidden_sizes = []
    for h in hidden_sizes_str:
        hidden_sizes.append(int(h))
    init_std = args.init_std
    hidden = args.hidden_activation
    output = args.output_activation
    hidden_activation = get_activation(hidden)
    output_activation = get_activation(output) 
    max_global_iters = args.max_global_iters 
    batch_size = args.batch_size
    max_path_length = args.max_path_length
    discount = args.discount
    plot = args.plot
    if plot is not 0:
        plot = 1
    save_period = args.save_period
    over_sampling_ratio = args.over_sampling_ratio
    alpha = args.alpha
    beta_zero = args.beta_zero
    seeds = []
    seeds_str = args.seeds.split(',')
    for s in seeds_str:
        seeds.append(int(s))
    predict_size = args.predict_size
    log_std_tol = args.log_std_tol
    verbose = args.verbose
    
    #logging setting
    log_file_name = args.log_file_name
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%d %b %H:%M:%S',
                        filename=log_file_name,
                        filemode='w')     
    
    global_params = load_params(init_path)
    results = []
    num_gpu_devices = args.num_gpu_devices
    vanilla = args.vanilla
    worker_count = 0
    while(len(seeds) > 0):
        #object lists to record each worker
        workers = []
        envs = []
        baselines = []
        # Create worker classes
        for i in range(num_gpu_devices):
            if(i >= len(seeds)):
                break;
            envs.append(Env(seeds[i]))
            baselines.append(LinearBaseline())
            workers.append(Worker(
                                envs[-1],
                                worker_count,
                                i,       
                                baselines[-1],
                                hidden_sizes = hidden_sizes,
                                init_std = init_std,
                                hidden_activation = hidden_activation,
                                output_activation = output_activation,
                                batch_size=batch_size,
                                max_path_length=max_path_length,
                                max_global_iters = max_global_iters,
                                discount=discount,
                                save_period = save_period,
                                plot = plot,
                                over_sampling_ratio = over_sampling_ratio,
                                alpha = alpha,
                                beta_zero = beta_zero,
                                seed = seeds[i],
                                predict_size = predict_size,
                                log_std_tol = log_std_tol,
                                verbose = verbose,
                                vanilla = vanilla
                            ))
            worker_count += 1
        #Start worker process using python multiprocessing wrapper
        worker_processes = []
        results_queue = Queue()
        for worker in workers:
            p = Process(target=worker.work, args=(global_params, results_queue))
            p.start()
            sleep(0.5)
            worker_processes.append(p)
        for process in worker_processes:
            process.join()
        for i in range(num_gpu_devices):
            results.append(results_queue.get())
        if len(seeds) > num_gpu_devices:
            seeds = seeds[num_gpu_devices:]
        else:
            seeds = []
    #Gathering and sorting results from each child process
    logging.info("The global minimum is:")
    logging.info(np.array2string(sorted(results,key=getKey)[0][1], max_line_width=10000))
