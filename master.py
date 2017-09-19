import argparse
from tensorflow import global_variables_initializer, get_collection, GraphKeys, Session, nn, reset_default_graph, set_random_seed
from policy import Policy_Network
from Env_example import Env
import os
import pickle

'''
This program will initialize step proposal network parameters. Then the parameters will be saved and distributed to all the workers.
'''

os.environ["CUDA_VISIBLE_DEVICES"] = ""  
def init_params(path, hidden_sizes, init_std, hidden_activation, output_activation):
    master_network = Policy_Network(Env(1),        
                                    hidden_sizes = hidden_sizes,
                                    init_std = init_std,
                                    hidden_activation = hidden_activation,
                                    output_activation = output_activation,
                                    scope='global')
    with Session() as sess:
        sess.run(global_variables_initializer())
        gps = sess.run(get_collection(GraphKeys.TRAINABLE_VARIABLES, scope='global'))
    with open(path,'wb') as f:
        pickle.dump(gps, f)
        
def get_activation(act_type="relu"):
    if act_type == "relu":
        return nn.relu
    elif act_type == "tanh":
        return nn.tanh
    else:
        raise NotImplementedError

def parse_args():
    parser = argparse.ArgumentParser(description='Implement master network, \
                        initialize all the parameters and save it. Every worker\
                        will load the initial parameters from disk')
    parser.add_argument('--init_params_path', type=str, default='global_vars.pickle',
                        help='the path to save initial parameters')
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
    parser.add_argument('--seed', type=int, default=1988,
                        help='the random seed to initialize network')

    return parser.parse_args()
        
if __name__ == "__main__":
    reset_default_graph()
    args = parse_args()
    
    seed = args.seed
    set_random_seed(seed)
    
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
    
    #run master network once to generate global initial parameters
    init_params(init_path, hidden_sizes, init_std,
                       hidden_activation, output_activation)