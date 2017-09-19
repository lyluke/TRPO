import sys
import math
import numpy as np
import bisect

class Experience(object):

    def __init__(self, conf):
        self.size = conf['size']
        self.replace_flag = conf['replace_old'] if 'replace_old' in conf else True
        self.priority_size = conf['priority_size'] if 'priority_size' in conf else self.size

        self.alpha = conf['alpha'] if 'alpha' in conf else 0.7
        self.beta_zero = conf['beta_zero'] if 'beta_zero' in conf else 0.5
        self.batch_size = conf['batch_size'] if 'batch_size' in conf else 32
        self.learn_start = conf['learn_start'] if 'learn_start' in conf else 1000
        self.total_steps = conf['steps'] if 'steps' in conf else 100000
        # partition number N, split total size to N part
        self.partition_num = conf['partition_num'] if 'partition_num' in conf else 100
        self.seed = conf['seed'] if 'seed' in conf else 1
        self.index = 0
        self.record_size = 0
        self.isFull = False

        self._experience = []
        self.exp_idx = []
        self.priority_queue = []
        self.distributions = self.build_distributions()

        self.beta_grad = (1 - self.beta_zero) / (self.total_steps - self.learn_start)
        np.random.seed(self.seed)
    def build_distributions(self):
        """
        preprocess pow of rank
        (rank i) ^ (-alpha) / sum ((rank i) ^ (-alpha))
        :return: distributions, dict
        """
        res = {}
        n_partitions = self.partition_num
        partition_num = 1
        # each part size
        partition_size = int(math.floor(self.size / n_partitions))

        for n in range(int(partition_size), self.size + 1, int(partition_size)):
            if self.learn_start <= n <= self.priority_size:
                distribution = {}
                # P(i) = (rank i) ^ (-alpha) / sum ((rank i) ^ (-alpha))
                pdf = list(
                    map(lambda x: math.pow(x, -self.alpha), range(1, n + 1))
                )
                pdf_sum = math.fsum(pdf)
                distribution['pdf'] = list(map(lambda x: x / pdf_sum, pdf))
                # split to k segment, and than uniform sample in each k
                # set k = batch_size, each segment has total probability is 1 / batch_size
                # strata_ends keep each segment start pos and end pos
                cdf = np.cumsum(distribution['pdf'])
                strata_ends = {1: 0, self.batch_size + 1: n}
                step = 1.0 / self.batch_size
                index = 1
                for s in range(2, self.batch_size + 1):
                    while cdf[index] < step:
                        index += 1
                    strata_ends[s] = index
                    step += 1.0 / self.batch_size

                distribution['strata_ends'] = strata_ends

                res[partition_num] = distribution

            partition_num += 1

        return res

    def fix_index(self,priority):
        """
        get next insert index
        :return: index, int
        """
        #if self.record_size < self.size:
        self.record_size += 1
            
        if self.record_size > self.size:

            '''
            if self.replace_flag:
                self.index = bisect.bisect_right(self.priority_queue, priority)
                self.priority_queue.insert(self.index, priority)
                self.priority_queue.pop(0)
                return self.index
            else:
            '''
            sys.stderr.write('Experience replay buff is full and replace is set to FALSE!\n')
            self.index = -10
            return self.index
        else:
            self.index = bisect.bisect_right(self.priority_queue, priority)
            self.priority_queue.insert(self.index, priority)
            return self.index

    def store(self, experience, priority):
        """
        store experience, suggest that experience is a tuple of (s1, a, r, s2, t)
        so each experience is valid
        :param experience: maybe a tuple, or list
        :return: bool, indicate insert status
        """
        self._experience.append(experience)
        insert_index = self.fix_index(priority)
        if insert_index >= 0:
            self.exp_idx.insert(insert_index, len(self._experience) - 1)
            if(self.record_size > self.size):
                #self._experience.pop(0)
                sys.stderr.write("Experience overflow!")
            return True

        elif insert_index == -10:
            sys.stderr.write('Insert failed\n')
            return False

    def retrieve(self, indices):
        """
        get experience from indices
        :param indices: list of experience id
        :return: experience replay sample
        """
        '''
        try:
            a = [self._experience[self.exp_idx[-v]] for v in indices]
            b = [self.priority_queue[-v] for v in indices]
        except IndexError:
            sys.stderr.write("a: {0}, b: {1}".format(len(a),len(b)))
            sys.stderr.write("v: {0}, {1}, {2}".format(v[0], v[1], v[-1]))
        '''
        return ([self._experience[self.exp_idx[-v]] for v in indices],[self.priority_queue[-v] for v in indices])
                       
    
    def sample(self, global_step, logging):
        """
        sample a mini batch from experience replay
        :param global_step: now training step
        :return: experience, list, samples
        :return: w, list, weights
        :return: rank_e_id, list, samples id, used for update priority
        """
        if self.record_size < self.learn_start:
            sys.stderr.write('Record size less than learn start! Sample failed\n')
            return False, False

        dist_index = int(math.floor(float(self.record_size) / float(self.size) * float(self.partition_num)))  
        partition_size = int(math.floor(self.size / self.partition_num))
        partition_max = dist_index * partition_size
        distribution = self.distributions[dist_index]
        rank_list = []
        # sample from k segments
        for n in range(1, self.batch_size + 1):
            if(distribution['strata_ends'][n] + 1 < distribution['strata_ends'][n + 1]):
                index = np.random.randint(distribution['strata_ends'][n] + 1,
                                       distribution['strata_ends'][n + 1])
            else:
                index = distribution['strata_ends'][n + 1]
                
            rank_list.append(index)

        
        # beta, increase by global_step, max 1
        #beta = min(self.beta_zero + (global_step - self.learn_start - 1) * self.beta_grad, 1)
        beta = self.beta_zero + (1.0 - self.beta_zero) / 2 + (1.0 - self.beta_zero) / 2 * np.tanh((global_step - self.total_steps/2) / (self.total_steps/6.0))
        #beta = (1.0 - self.beta_zero) * np.exp(float(global_step) / float(self.total_steps)) / (np.exp(1) - 1) + (self.beta_zero * np.exp(1) - 1) / (np.exp(1) - 1)
        # find all alpha pow, notice that pdf is a list, start from 0
        alpha_pow = [distribution['pdf'][v - 1] for v in rank_list]
        # w = (N * P(i)) ^ (-beta) / max w
        w = np.power(np.array(alpha_pow) * partition_max, -beta)
        w_max = max(w)
        w = np.divide(w, w_max)
        
        logging.info("current beta is: {0}".format(beta))

        # get experience id according rank_list
        experience, priority = self.retrieve(rank_list)
        return experience, rank_list, w, priority
