""" Module with examples of common pruning patterns
"""
from .abstract import Pruning
from .utils import get_activations, get_param_gradients
from torch import nn as nn


class ActivationMixin(Pruning):

    def update_activations(self):
        assert self.inputs is not None, \
            "Inputs must be provided for activations"
        self._activations = get_activations(self.model, self.inputs)

    def activations(self, only_prunable=True):
        if not hasattr(self, '_activations'):
            self.update_activations()
        if only_prunable:
            return {module: self._activations[module] for module in self.prunable}
        else:
            return self._activations

    def module_activations(self, module,next_iter=False):
        if not hasattr(self, '_activations') or next_iter:
            print("Updating activations")
            self.update_activations()
        return self._activations[module]


class GradientMixin(Pruning):

    def update_gradients(self, include_smoothing = False):
        assert self.inputs is not None and self.outputs is not None, \
            "Inputs and Outputs must be provided for gradients"
        loss_func = None
        if self.dataset_name == 'STSBDATA':
            loss_func = nn.MSELoss()
        else:
            loss_func = None
        self._param_gradients = get_param_gradients(self.model, self.inputs, self.outputs, include_smoothing = include_smoothing, loss_func=loss_func)

    def param_gradients(self, only_prunable=True, include_smoothing = False):
        if not hasattr(self, "_param_gradients"):
            self.update_gradients(include_smoothing)
        if only_prunable:
            return {module: self._param_gradients[module] for module in self.prunable}
        else:
            return self._param_gradients

    def module_param_gradients(self, module, next_iter = False, include_smoothing = False):

        if not hasattr(self, "_param_gradients") or next_iter:
            print('updating gradients')
            self.update_gradients(include_smoothing)
        return self._param_gradients[module]

    def input_gradients(self):
        raise NotImplementedError("Support coming soon")

    def output_gradients(self):
        raise NotImplementedError("Support coming soon")
