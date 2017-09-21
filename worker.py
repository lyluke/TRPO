import numpy as np
import tensorflow as tf
import pickle
import itertools
import logging
from krylov import cg
import scipy.signal
from Experience import Experience
from policy import Policy_Network
import os

class Worker():
    """
    Define worker class for multiple process implementation. Every worker's work() method will be executed on a seperated child process and paired with one CUDA device.
    """
    def __init__(
        self,
        env,
        name,
        visible_gpu_device,
        baseline,
        model_path = './',
        hidden_sizes = [128,64],
        init_std = 1.0,
        hidden_activation = tf.nn.relu,
        output_activation = tf.nn.tanh,
        batch_size=5000,
        max_path_length=10,
        discount=0.,
        plot=False,
        cg_iterations = 10,
        kl_step_size = 0.01,
        backtrack_ratio=0.8,
        max_backtracks=15,
        cg_reg_coef=1e-5,
        reward_scale=1.,
        max_global_iters=101,
        log_file_name = None,
        save_period = 100,
        over_sampling_ratio = 40,
        alpha = 1.35,
        beta_zero = 0.4,
        seed = None,
        predict_size = 10,
        log_std_tol = -5,
        verbose = 0,
        vanilla = 0
    ):
        #set only one cuda device visible to one worker
        self.visible_gpu_device = visible_gpu_device
        self.name = "worker_" + str(name)
        self.number = name
        #each worker's policy network has its own dedicated scope
        self.scope = self.name
        self.model_path = model_path       
        self.env = env
        self.baseline = baseline
        #size of hidden layers of policy network
        self.hidden_sizes = hidden_sizes
        #initial value of standard deviation of normal distribution
        self.init_std = init_std
        #hidden layer activation, default is relu
        self.hidden_activation = hidden_activation
        #output layer activation, default is tanh to limit step within (-1,1)
        self.output_activation = output_activation
        #batch size of policy network training data
        self.batch_size = batch_size
        #max length of one path
        self.max_path_length = max_path_length
        #discounted factor, default is 0
        self.discount = discount
        self.plot = plot
        #max number of iteration of conjugate gradient
        self.cg_iterations = cg_iterations
        #how much kl divergence is allowed between new and old distribution
        self.kl_step_size = kl_step_size
        #when performing line search, shrinking factor for each iteration of trial
        self.backtrack_ratio = backtrack_ratio
        #max number of back track line search
        self.max_backtracks = max_backtracks
        self.log_file_name = log_file_name
        #regularization coefficient of conjugate gradient, preventing singular matrix inverse problem
        self.cg_reg_coef = cg_reg_coef
        self.reward_scale = reward_scale
        #max number of iteration of trpo updating
        self.max_global_iters = max_global_iters
        self.save_period = save_period
        #how much over sampling will be performed on top of batch size
        self.over_sampling = over_sampling_ratio
        #exponential coefficient of priority distribution, the larger the alpha is, the more high priority path will be sampled
        self.alpha = alpha
        #bias annealing coefficient starting value
        self.beta_zero = beta_zero
        #the batch size when step proposal network performs prediction
        self.predict_size = predict_size
        if seed is None:
            raise NotImplementedError
        self.seed = seed
        #how much standard deviation is good enough to terminate trpo updating, sigma = exp(log_std_tol)
        self.log_std_tol = log_std_tol
        self.verbose = verbose
        #calculate the over sampled size of paths
        self.scheduled_size = int(float(self.batch_size) / float(self.max_path_length) * float(self.over_sampling))
        conf = {
        'alpha':self.alpha,
        'size': self.scheduled_size,
        'learn_start': 0,
        'partition_num': 5,
        'beta_zero':self.beta_zero,
        'steps': self.max_global_iters,
        'batch_size': self.batch_size / self.max_path_length,
        'seed': self.seed
           }
        self.experience = Experience(conf)
        #whether use vanilla trpo or prioritized experience trpo
        self.vanilla = vanilla
        
    def demo(self, global_params, period):
        results = []
        os.environ["CUDA_VISIBLE_DEVICES"]="{0}".format(self.visible_gpu_device)
        np.random.seed(self.seed)
        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        with tf.device('/gpu:0'):
            self.local_policy = Policy_Network(        
                self.env,
                hidden_sizes = self.hidden_sizes,
                scope=self.name,
                init_std = self.init_std,
                hidden_activation = self.hidden_activation,
                output_activation = self.output_activation
            )
        global_iters = 0
        
        with tf.Session() as sess: 
            sess.run(tf.global_variables_initializer())
            logging.info("Starting worker " + str(self.number))
            #copy global params to local network
            self.update_local_ops(sess, global_params)
            with sess.as_default(), sess.graph.as_default():
                while global_iters < self.max_global_iters:
                    
                    if global_iters % period == 0:
                        paths = self.obtain_samples(sess, self.predict_size)
                        results.append(np.array([path['observations'] for path in paths]))
                        
                    logging.info("{0} at global epiode #{1} start:".format(self.name, global_iters))
                    cur_step, log_std = self.train(sess, global_iters)
                    global_iters += 1
                    if np.max(log_std) < self.log_std_tol:
                        paths = self.obtain_samples(sess, self.predict_size)
                        results.append(np.array([path['observations'] for path in paths]))
                        break
        tf.reset_default_graph()
        return results
    
    def work(self, global_params, outputs):
        os.environ["CUDA_VISIBLE_DEVICES"]="{0}".format(self.visible_gpu_device)
        np.random.seed(self.seed)
        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        with tf.device('/gpu:0'):
            self.local_policy = Policy_Network(        
                self.env,
                hidden_sizes = self.hidden_sizes,
                scope=self.name,
                init_std = self.init_std,
                hidden_activation = self.hidden_activation,
                output_activation = self.output_activation
            )
        global_iters = 0
        
        with tf.Session() as sess: 
            sess.run(tf.global_variables_initializer())
            logging.info("Starting worker " + str(self.number))
            #copy global params to local network
            self.update_local_ops(sess, global_params)
            with sess.as_default(), sess.graph.as_default():
                while global_iters < self.max_global_iters:
                    
                    logging.info("{0} at global epiode #{1} start:".format(self.name, global_iters))
                    cur_step, log_std = self.train(sess, global_iters)
                    global_iters += 1
                    # Periodically save model parameters
                    if global_iters % self.save_period == 0:
                        saver = tf.train.Saver(max_to_keep=5)
                        saver.save(sess,self.model_path+'/model-'+str(global_iters)+'.cptk')
                        logging.info(str(self.name)+" "+"Saved Model")
                    
                    if np.max(log_std) < self.log_std_tol:
                        break
            outputs.put(self.predict(sess))
        tf.reset_default_graph()
        
    def getKey(self, path):
        return path[0]
    
    def predict(self, sess):
        #collect self.predict_size paths to use as the prediction of global minimum
        paths = self.obtain_samples(sess, self.predict_size)
        for i in range(len(paths)):
            paths[i] = (self.env.function(paths[i]['observations'][-1,:]+paths[i]['actions'][-1,:]),paths[i]['observations'][-1,:]+paths[i]['actions'][-1,:])
        p = sorted(paths,key=self.getKey)[0]
        return p
    
    def update_local_ops(self, sess, global_params):
        #global params is a list of one-dimension arrays, need to reconstruct the shape.
        sess.run(self.local_policy.assign_op, 
                 feed_dict=dict({i:np.array(d).reshape(i.get_shape().as_list()) for i, d in itertools.izip(self.local_policy.assign_value, global_params)}))
            
    #all the functions below is training related
    def train(self,sess, global_iters):
        #first rollout new paths, then at train newtork with them.
        cur_steps = []
        
        if self.vanilla is 0:
            #prioritized experience replay
            self.experience.record_size = 0
            self.experience._experience = []
            self.experience.exp_idx = []
            self.experience.priority_queue = []

            if self.verbose >= 1:
                logging.info(str(self.name)+" "+'Roll out to collect samples:')
            paths = self.obtain_samples(sess, self.scheduled_size)

            p_stored = []
            #store sampled path in experience in descending order
            for i in range(len(paths)):
                p_stored.append(-self.env.function(paths[i]['observations'][-1,:]+paths[i]['actions'][-1,:]))
                self.experience.store(paths[i], p_stored[-1])
            if self.verbose >= 1:
                logging.info(str(self.name)+" "+'all paths are collected')  
            #sample batch_size data based on pre-defined prioritized distribution
            paths, rank_list, w, p = self.experience.sample(global_iters, logging)
        else:
            #vanilla trpo only need to sample self.batch_size data and no need for bias compensation
            paths = self.obtain_samples(sess, self.batch_size / self.max_path_length)
            w = np.ones(self.batch_size / self.max_path_length)
        
        #procoess raw data
        samples_data= self.process_samples(paths, w)
        
        if self.verbose >= 1:
            logging.info(str(self.name)+" "+'Optimize policy:')
        loss_before, loss_after, mean_kl, cur_step, log_std = self.optimize_policy(sess, samples_data)
        cur_steps.append(cur_step)
        
        if self.vanilla is 0 and self.verbose >= 2:
            #for me to debug Michalewicz function
            logging.info(str(self.name)+" "+"stored priority distribution:")            
            hist = np.histogram(np.array(p_stored), bins=[0,0.5,1.0,1,1.5,2.0,2.5,2.8,2.86,2.9])
            logging.info(np.array2string(hist[0], max_line_width=10000))
            logging.info(np.array2string(hist[1], max_line_width=10000))

            logging.info(str(self.name)+" "+"sampled priority distribution:")            
            hist = np.histogram(p, bins=[0,0.5,1.0,1,1.5,2.0,2.5,2.8,2.86,2.9])
            logging.info(np.array2string(hist[0], max_line_width=10000))
            logging.info(np.array2string(hist[1], max_line_width=10000))

            logging.info(str(self.name)+" "+"sampled point 1st dimension distribution:")
            hist = np.histogram(np.array([path['observations'][-1,0]+path['actions'][-1,0] for path in paths]), bins=[-10,0,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,10])
            logging.info(np.array2string(hist[0], max_line_width=10000))
            logging.info(np.array2string(hist[1], max_line_width=10000))
            logging.info(str(self.name)+" "+"stored point 1st dimension distribution:")
            hist = np.histogram(np.array([path['observations'][-1,0]+path['actions'][-1,0] for path in self.experience._experience]), bins=[-10,0,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,10])
            logging.info(np.array2string(hist[0], max_line_width=10000))
            logging.info(np.array2string(hist[1], max_line_width=10000))

            logging.info(str(self.name)+" "+"sampled point 2nd dimension distribution:")
            hist = np.histogram(np.array([path['observations'][-1,1]+path['actions'][-1,1] for path in paths]), bins=[-10,0,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,10])
            logging.info(np.array2string(hist[0], max_line_width=10000))
            logging.info(np.array2string(hist[1], max_line_width=10000))
            logging.info(str(self.name)+" "+"stored point 2nd dimension distribution:")
            hist = np.histogram(np.array([path['observations'][-1,1]+path['actions'][-1,1] for path in self.experience._experience]), bins=[-10,0,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,10])            
            logging.info(np.array2string(hist[0], max_line_width=10000))
            logging.info(np.array2string(hist[1], max_line_width=10000))

            logging.info(str(self.name)+" "+"sampled top 2 experience:")
            logging.info("1st priority:{0}, final point:{1}, start point:{2}".format(p[0],paths[0]['observations'][-1,:]+paths[0]['actions'][-1,:],paths[0]['observations'][0,:]))
            logging.info("2nd priority:{0}, final point:{1}, start point:{2}".format(p[1],paths[1]['observations'][-1,:]+paths[1]['actions'][-1,:],paths[1]['observations'][0,:]))
            logging.info("last priority:{0}, final point:{1}, start point:{2}".format(p[-1],paths[-1]['observations'][-1,:]+paths[-1]['actions'][-1,:],paths[-1]['observations'][0,:]))
        logging.info('\n')
        
        if (self.plot == True) and (global_iters % self.save_period == 0):
            with open(self.model_path+'/path-'+str(global_iters)+'.dat', 'wb') as f:
                pickle.dump((paths[0]['observations'], paths[0]['rewards']), f)
                
        return np.mean(np.stack(cur_steps, axis=0), axis=0), log_std
        
    def obtain_samples(self, sess, limit=None):
        #parallely sampling path using CUDA
        if limit is None:
            limit = self.batch_size * self.over_sampling / self.max_path_length
        #initial position reset
        o = self.env.reset(limit)
        observations = np.zeros((self.max_path_length, o.shape[0], o.shape[1]))
        actions = np.zeros((self.max_path_length, o.shape[0], o.shape[1]))
        rewards = np.zeros((self.max_path_length, o.shape[0]))
        means = np.zeros((self.max_path_length, o.shape[0], o.shape[1]))
        log_stds= np.zeros((self.max_path_length, o.shape[0], o.shape[1]))
        path_length = 0
        while path_length < self.max_path_length:
            #roll out the path by self.max_path_length steps
            actions[path_length,:,:], means[path_length,:,:], log_stds[path_length,:,:] = self.get_action(limit, o, sess)
            next_o, r, d = self.env.step(actions[path_length,:,:])
            observations[path_length,:,:]=o
            rewards[path_length,:]=r
            o = next_o
            path_length += 1
        
        #group all paths in a list
        paths = []
        for i in range(limit):
            path = dict(
            observations=observations[:,i,:],
            actions=actions[:,i,:],
            rewards=self.reward_scale * rewards[:,i],
            agent_infos=np.array([{'mean':means[j,i,:],'log_std':log_stds[j,i,:]} for j in range(path_length)]))
            paths.append(path)
        return paths

    def process_samples(self, paths, w):
        #calculate advantage first than fit baseline for next iteration
        baselines = []
        returns = []
        #calculate discounted rewards and subtract baseline from rewards to get advantages
        for path in paths:
            path_baselines = np.append(self.baseline.predict(path), 0)
            deltas = path["rewards"] + self.discount * path_baselines[1:] - \
                     path_baselines[:-1]
            path["advantages"] = self.discount_cumsum(
                deltas, self.discount)
            path["returns"] = self.discount_cumsum(path["rewards"], self.discount)
            baselines.append(path_baselines[:-1])
            returns.append(path["returns"])
        l = path["rewards"].shape[0]
        ws = np.repeat(w, l).reshape((-1,1))
        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        rewards = np.concatenate([path["rewards"] for path in paths])
        advantages = np.concatenate([path["advantages"] for path in paths])
        advantages = advantages.reshape((-1,1))
        agent_infos = np.concatenate([path["agent_infos"] for path in paths])

        average_discounted_return = np.mean([path["returns"][0] for path in paths])

        undiscounted_returns = [sum(path["rewards"]) for path in paths]
        samples_data = dict(
            ws = ws,
            observations=observations,
            actions=actions,
            rewards=rewards,
            advantages=advantages,
            agent_infos=agent_infos,
            paths=paths,
        )
        if self.verbose >= 1:
            logging.info(str(self.name)+" "+"fitting baseline...")
            
        self.baseline.fit(paths)
        
        if self.verbose >= 1:
            logging.info(str(self.name)+" "+"fitted")
            logging.info(str(self.name)+" "+'AverageDiscountedReturn: {0}'.format(average_discounted_return))
            logging.info(str(self.name)+" "+'AverageReturn: {0}'.format(np.mean(undiscounted_returns)))
            logging.info(str(self.name)+" "+'NumTrajs: {0}'.format(len(paths)))
            logging.info(str(self.name)+" "+'StdReturn: {0}'.format(np.std(undiscounted_returns)))
            logging.info(str(self.name)+" "+'MaxReturn: {0}'.format(np.max(undiscounted_returns)))
            logging.info(str(self.name)+" "+'MinReturn: {0}'.format(np.min(undiscounted_returns)))
            logging.info(str(self.name)+" "+"Stopping point is:{0}".format(observations[-1,:]+actions[-1,:]))
        return samples_data
    
    def get_action(self, batch_size, observation, sess):
        resize_obs = observation.reshape((batch_size,self.local_policy.size[0]))
        
        #using CDUA capability, parallely sample new steps
        [mean, log_std] = sess.run([self.local_policy.mean, self.local_policy.log_std], 
                                   feed_dict = {self.local_policy.state:resize_obs})
        #sample from normal distribution
        rnd = np.random.normal(size=mean.shape)
        action = rnd * np.exp(log_std) + mean

        return action, mean, log_std
    
    def optimize_policy(self, sess, samples_data):
        samples_data['log_std'] = np.stack([agent_info['log_std'] for agent_info in samples_data["agent_infos"]])
        samples_data['mean'] = np.stack([agent_info['mean'] for agent_info in samples_data["agent_infos"]])
        
        loss_before, loss_after, mean_kl, cur_step, log_std = self.optimize(sess, samples_data)
        
        if self.verbose >= 1:
            logging.info(str(self.name)+" "+'LossBefore: {0}'.format(loss_before))
            logging.info(str(self.name)+" "+'LossAfter: {0}'.format(loss_after))
            logging.info(str(self.name)+" "+'MeanKL: {0}'.format(mean_kl))
            logging.info(str(self.name)+" "+'dLoss: {0}'.format(loss_before - loss_after))
        return loss_before, loss_after, mean_kl, cur_step, log_std

    def optimize(self, sess, samples_data):
        feed_dict = dict({self.local_policy.state:samples_data['observations'],
                          self.local_policy.actions:samples_data['actions'],
                          self.local_policy.advantage:samples_data['advantages'],
                          self.local_policy.old_mean:samples_data['mean'],
                          self.local_policy.old_log_std:samples_data['log_std'],
                          self.local_policy.w:samples_data['ws']
                         })
        if self.verbose >= 1:
            logging.info(str(self.name)+" "+"computing loss before")
            logging.info(str(self.name)+" "+"performing update")
            logging.info(str(self.name)+" "+"computing descent direction")
        
        #calculate loss gradient g using symbolic experssion
        [self.flat_gradient,loss_before] = sess.run([self.local_policy.grads_flatten,self.local_policy.surr_loss], 
                                                    feed_dict=feed_dict)
        #calculate s = A^-1*g by A*g
        descent_direction = cg(self.local_policy, self.flat_gradient, feed_dict, sess, 
                               reg_coef = self.cg_reg_coef,cg_iters=self.cg_iterations,verbose=False)
        #calculate A*s
        A_dot_descent_direction = sess.run(self.local_policy.fisher_prod_x_flatten, 
                                           feed_dict = dict(feed_dict, **{self.local_policy.xs_flatten:descent_direction})) +\
                                           self.cg_reg_coef * descent_direction
        #calculate line search step = sqrt(2kl/sAs)
        initial_step_size = np.sqrt(
            2.0 * self.kl_step_size * (1. / (np.abs(descent_direction.dot(A_dot_descent_direction)) + 1e-8))
        )
        
        #initial descent step for line search
        flat_descent_step = initial_step_size * descent_direction
        
        if self.verbose >= 1:
            logging.info(str(self.name)+" "+"descent direction computed")
        
        prev_param = sess.run(self.local_policy.get_params, feed_dict=feed_dict)
        
        if self.verbose >= 1:
            logging.info(str(self.name)+" "+"current log_std: {0}".format(prev_param[-1]))
        
        #perform line search along flat_descent_step direction to ensure kl divergence smaller than kl_step_size
        #shrink descent step by self.backtrack_ratio
        loss_after=0 
        kl_after = 0
        for n_iter, ratio in enumerate(self.backtrack_ratio ** np.arange(self.max_backtracks)):
            cur_step = ratio * flat_descent_step
            start = 0
            cur_param = []
            for param in prev_param:
                size = param.flatten().shape[0]
                cur_param.append(param - cur_step[start:start+size].reshape(param.shape))
                start += size
            sess.run(self.local_policy.assign_op,feed_dict={i: d for i, d in zip(self.local_policy.assign_value, cur_param)})    
            loss_after, kl_after, log_std = sess.run([self.local_policy.surr_loss, self.local_policy.mean_kl, self.local_policy.log_std],feed_dict=feed_dict)
            if np.isnan(kl_after):
                import ipdb; ipdb.set_trace()
            if loss_after < loss_before and kl_after <= self.kl_step_size:
                break
                
        if self.verbose >= 1:
            logging.info(str(self.name)+" "+"backtrack iters: %d" % n_iter)
            logging.info(str(self.name)+" "+"optimization finished")
        
        return loss_before, loss_after, kl_after, ratio * flat_descent_step, log_std
        
    def discount_cumsum(self, x, discount):
        # See https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html#difference-equation-filtering
        return scipy.signal.lfilter([1], [1, -discount], x[::-1], axis=0)[::-1]
