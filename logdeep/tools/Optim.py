'''A wrapper class for scheduled optimizer '''
import numpy as np

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, n_warmup_steps, lr_max, decay_rate, decay_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.lr_max = lr_max
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.n_steps = 0

    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()

    def step(self):
        self._optimizer.step()

    def state_dict(self):
        return self._optimizer.state_dict()

    def load_state_dict(self,state_dict):
        self._optimizer.load_state_dict(state_dict)

    def load_n_steps(self, n_steps):
        self.n_steps = n_steps

    def _get_lr_scale(self):
        if self.n_steps <= self.n_warmup_steps:
            lr = self.lr_max * self.n_steps / (self.n_warmup_steps)
        else:
            lr = self.lr_max * self.decay_rate ** (self.n_steps / self.decay_steps)
        return lr

    def update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

