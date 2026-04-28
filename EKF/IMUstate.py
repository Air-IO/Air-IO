import torch

import pypose as pp
from torch.autograd.functional import jacobian


class IMUstate(pp.module.NLS):
    def __init__(self):
        self.dt = None
        self.gravity = torch.tensor([0., 0., 9.81007])
        super().__init__()

    @property
    def A(self):
        '''
        '''
        func = lambda x: self.state_transition(x, self._ref_input, self._ref_dt, self._ref_t)
        return jacobian(func, self._ref_state, **self.jacargs)

    @property
    def B(self):
        func = lambda x: self.state_transition(self._ref_state, x, self._ref_dt, self._ref_t)
        return jacobian(func, self._ref_input, **self.jacargs)

    @property
    def C(self):
        r'''
        Linear/linearized system output matrix.

        .. math::
            \mathbf{C} = \left. \frac{\partial \mathbf{g}}{\partial \mathbf{x}} \right|_{\chi^*}
        '''
        func = lambda x: self.observation(x, self._ref_input, self._ref_dt, self._ref_t)
        return jacobian(func, self._ref_state, **self.jacargs)

    @property
    def D(self):
        r'''
        Linear/Linearized system observation matrix.

        .. math::
            \mathbf{D} = \left. \frac{\partial \mathbf{g}}
                                {\partial \mathbf{u}} \right|_{\chi^*}
        '''
        func = lambda x: self.observation(self._ref_state, x, self._ref_dt, self._ref_t)
        return jacobian(func, self._ref_input, **self.jacargs)

    def forward(self, state, input, dt):
        self.dt = torch.atleast_1d(dt)
        self.state, self.input = torch.atleast_1d(state), torch.atleast_1d(input)
        state = self.state_transition(self.state, self.input, self.dt, self.systime)
        obs = self.observation(self.state, self.input, self.dt, self.systime)
        return state, obs

    def set_refpoint(self, state=None, input=None, dt=None, t=None):
        self._ref_state = self.state if state is None else torch.atleast_1d(state)
        self._ref_input = self.input if input is None else torch.atleast_1d(input)
        self._ref_dt = self.dt if dt is None else torch.atleast_1d(dt)
        self._ref_t = self.systime if t is None else torch.atleast_1d(t)
        self._ref_f = self.state_transition(self._ref_state, self._ref_input, self._ref_dt, self._ref_t)
        self._ref_g = self.observation(self._ref_state, self._ref_input, self._ref_dt, self._ref_t)
        return self

