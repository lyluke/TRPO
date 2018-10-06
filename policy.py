import numpy as np
import tensorflow as tf
import itertools
    
class Policy_Network():
    """
    Definition of step proposal policy network. Default is a fully connected network with relu as internal activation and tanh as the ouput activation to limit the maximum step size less than 1.
    """
    def __init__(
        self,
        env,
        hidden_sizes = [64,64],
        init_std = 1.0,
        hidden_activation = tf.nn.relu,
        output_activation = tf.nn.tanh,
        scope='global',
    ):
        
        self.env = env
        self.size = env.spec()['size']
        self.hidden_sizes = hidden_sizes
        self.init_std = init_std
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.scope=scope    
        
        with tf.variable_scope(self.scope):
            #define policy
            self.__init_policy()
            
            #Only the worker network need ops for loss functions and gradient updating.
            if self.scope != 'global':
                #complete trpo compute graph based on policy
                self.__init_kl()

                #calculate loss based on importance sampling
                self.__init_loss()

                #conjugate gradients
                self.__init_cg()
                
    def __init_policy(self):
        #input state s -> relu internal layers -> tanh -> mean of normal distribution
        self.state = tf.placeholder(tf.float32, [None,self.size[0]], "state")
        #w here for importance sampling compensation
        self.w     = tf.placeholder(tf.float32, [None,1], "w")
        net = self.state
        for i, num in enumerate(self.hidden_sizes):
            net = tf.contrib.layers.fully_connected(
            inputs = net,
            num_outputs = num,
            activation_fn = self.hidden_activation,
            weights_initializer = tf.contrib.layers.variance_scaling_initializer(),
            biases_initializer = tf.zeros_initializer())
        self.mean = tf.contrib.layers.fully_connected(
            inputs = net,
            num_outputs = self.size[0],
            activation_fn = self.output_activation,
            weights_initializer = tf.random_uniform_initializer(minval=-3e-3,maxval=3e-3),
            biases_initializer = tf.random_uniform_initializer(minval=-3e-3,maxval=3e-3))
        
        self.log_std = tf.Variable(np.log(self.init_std)*tf.ones([1,self.size[0]]),name='std')
        self.log_std_tile = tf.tile(self.log_std, tf.stack([tf.shape(self.mean)[0], 1]))

    def __init_kl(self):
        self.old_mean = tf.placeholder(tf.float32, [None,self.size[0]], "old_mean")
        self.old_log_std = tf.placeholder(tf.float32, [None,self.size[0]], "old_log_std")
        # KL divergence formula:
        # { (\mu_1 - \mu_2)^2 + \sigma_1^2 - \sigma_2^2 } / (2\sigma_2^2) + ln(\sigma_2/\sigma_1)}
        self.old_std = tf.exp(self.old_log_std)
        self.new_std = tf.exp(self.log_std_tile)
        self.mean_kl = tf.reduce_mean(tf.multiply(tf.reduce_sum(tf.div(tf.square(self.old_mean - self.mean) + \
                    tf.square(self.old_std) - tf.square(self.new_std), 
                    2*tf.square(self.new_std)+1e-8) + \
                    self.log_std_tile - self.old_log_std, 1),self.w))

    def __init_loss(self):
        #Expectaion of likelihood ratio weighted by advantages
        self.actions = tf.placeholder(tf.float32, [None,self.size[0]], "actions")
        new_zs = tf.div(self.actions - self.mean, self.new_std)
        logli_new = -tf.reduce_sum(self.log_std_tile, axis=1,keepdims=True) - \
                    0.5 * tf.reduce_sum(tf.square(new_zs), axis=1, keepdims=True)
        old_zs = tf.div(self.actions - self.old_mean, self.old_std)
        logli_old = -tf.reduce_sum(self.old_log_std, axis=1, keepdims=True) - \
                    0.5 * tf.reduce_sum(tf.square(old_zs), axis=1, keepdims=True)
        self.advantage = tf.placeholder(tf.float32, [None,1], "advantage")
        self.likelihood_ratio = tf.exp(logli_new - logli_old)
        self.weighted_likelihood_ratio = tf.multiply(self.likelihood_ratio, self.w)
        self.surr_loss = - tf.reduce_mean(tf.multiply(self.likelihood_ratio,self.advantage)) - 5e-3 * tf.reduce_sum(self.log_std)

    def __init_cg(self):
        #Define symbolic experssion for conjugate gradient calculation
        self.get_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
        self.grads = tf.gradients(self.surr_loss, self.get_params)
        self.grads_flatten = tf.concat([tf.reshape(p, [-1]) for p in self.grads], 0)
        self.kl_grads = tf.gradients(self.mean_kl, self.get_params)
        self.xs_flatten = tf.placeholder(tf.float32, tf.Tensor.get_shape(self.grads_flatten))
        self.xs = []
        begin = 0
        for g in self.grads:
            size = np.prod(np.array(tf.Tensor.get_shape(g).as_list()))
            self.xs.append(tf.reshape(tf.slice(self.xs_flatten, list([begin]), list([size])),tf.shape(g)))
            begin = begin + size

        self.fisher_prod_x = tf.gradients(
        tf.add_n([tf.reduce_sum(tf.multiply(g, x)) for g, x in itertools.izip(self.kl_grads, self.xs)]),
        self.get_params)
        self.fisher_prod_x_flatten = tf.concat([tf.reshape(p, [-1]) for p in self.fisher_prod_x], 0)

        self.var_list = self.get_params
        self.assign_op = []
        self.assign_value = []
        for var in self.var_list:
            self.assign_value.append(tf.placeholder(tf.float32,var.get_shape()))
            self.assign_op.append(tf.assign(var,self.assign_value[-1]))
