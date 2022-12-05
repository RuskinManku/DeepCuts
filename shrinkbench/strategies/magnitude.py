"""Magnitude based pruning

Implements pruning strategy that prunes as follows

For a given tensor, sort by absolute value and
keep only the _fraction_ with highest magnitudes
so that overall desired compression is achieved
"""

import numpy as np

from ..pruning import (LayerPruning,
                       VisionPruning,
                       GradientMixin,
                       ActivationMixin)
from .utils import (fraction_threshold,
                    fraction_mask,
                    map_importances,
                    flatten_importances,
                    importance_masks,
                    activation_importance,
                    activation_importance_no_weight
                    )


class GlobalMagWeight(VisionPruning):

    def model_masks(self,make_mask=False,next_iter=False):
        importances = map_importances(np.abs, self.params())
        flat_importances = flatten_importances(importances)
        threshold = fraction_threshold(flat_importances, self.fraction)
        masks = importance_masks(importances, threshold)
        return masks


class LayerMagWeight(LayerPruning, VisionPruning):

    def layer_masks(self, module):
        params = self.module_params(module)
        importances = {param: np.abs(value) for param, value in params.items()}
        masks = {param: fraction_mask(importances[param], self.fraction)
                 for param, value in params.items() if value is not None}
        return masks


class GlobalMagGrad(GradientMixin, VisionPruning):

    def model_masks(self,make_mask=False,next_iter=False):
        params = self.params()
        grads = self.param_gradients()
        importances = {mod:
                       {p: np.abs(params[mod][p]*grads[mod][p])
                        for p in mod_params}
                       for mod, mod_params in params.items()}
        flat_importances = flatten_importances(importances)
        threshold = fraction_threshold(flat_importances, self.fraction)
        masks = importance_masks(importances, threshold)
        return masks


class LayerMagGrad(GradientMixin, LayerPruning, VisionPruning):

    def layer_masks(self, module):
        params = self.module_params(module)
        grads = self.module_param_gradients(module)
        importances = {param: np.abs(grads[param]) for param, value in params.items()}
        masks = {param: fraction_mask(importances[param], self.fraction)
                 for param, value in params.items() if value is not None}
        return masks


class GlobalMagAct(ActivationMixin, VisionPruning):

    def model_masks(self,make_mask=False,next_iter=False):
        params = self.params()
        activations = self.activations()
        # [0] is input activation
        importances = {mod:
                       {p: np.abs(activation_importance(params[mod][p], activations[mod][0]))
                        for p in mod_params}
                       for mod, mod_params in params.items()}
        flat_importances = flatten_importances(importances)
        threshold = fraction_threshold(flat_importances, self.fraction)
        masks = importance_masks(importances, threshold)
        return masks


class LayerMagAct(ActivationMixin, LayerPruning, VisionPruning):

    def layer_masks(self, module):
        params = self.module_params(module)
        input_act, _ = self.module_activations(module)
        importances = {param: np.abs(activation_importance(value, input_act))
                       for param, value in params.items()}
        masks = {param: fraction_mask(importances[param], self.fraction)
                 for param, value in params.items() if value is not None}
        return masks

class LayerGradCAM(ActivationMixin, GradientMixin, LayerPruning, VisionPruning):

    def layer_masks(self, module, next_iter = False):
        # print(f'layer mask called with {module}')
        params = self.module_params(module)
        output_act = self.module_activations(module, next_iter = next_iter)
        # if output_act.shape == input_act.shape:
        #     output_act += input_act
        grads = self.module_param_gradients(module, next_iter = next_iter)
        output_act = output_act.mean(axis=0)
        output_act = output_act.mean(axis=0)
        # print('output_act:',output_act.shape)
        if module not in self.importances:
            self.importances[module] = {
            'weight': np.abs(params['weight']*grads['weight']*np.repeat(output_act.reshape(-1,1),grads['weight'].shape[1],axis=1)),
            'bias': np.abs(params['bias']*grads['bias']*output_act)
            }
        else:
            self.importances[module]['weight'] += np.abs(params['weight']*grads['weight']*np.repeat(output_act.reshape(-1,1),grads['weight'].shape[1],axis=1))
            self.importances[module]['bias'] += np.abs(params['bias']*grads['bias']*output_act)
            
        # importances = {param: np.abs(output_act*grads[param])
        #                for param, value in params.items()}
        # print(self.fraction)
        # print(importances)
        # masks = {param: fraction_mask(importances[param], self.fraction)
        #          for param, value in params.items() if value is not None}
        # return masks
    
    def model_masks(self, prunable=None, make_mask = False, next_iter = False):
        """Compute masks using the said strategy for every module
        This is a straight forward implementation that supports
        strategies that prune each module independently
        """
        if prunable is None:
            prunable = self.prunable_modules()
        for module in prunable:
            self.layer_masks(module, next_iter)
            next_iter = False
        if make_mask:
            masks = dict()
            for module in prunable:
                masks[module] = {param: fraction_mask(self.importances[module][param], self.fraction)
                                for param, value in self.module_params(module).items() if value is not None}

            # print(self.importances)
            return masks

class GlobalGradCAMShift(ActivationMixin, GradientMixin, VisionPruning):

    def layer_masks(self, module, next_iter = False):
        # print(f'layer mask called with {module}')
        params = self.module_params(module)
        output_act = self.module_activations(module, next_iter = next_iter)
        # if output_act.shape == input_act.shape:
        #     output_act += input_act
        grads = self.module_param_gradients(module, next_iter = next_iter)
        output_act = output_act.mean(axis=0)
        output_act = output_act.mean(axis=0)
        # print('output_act:',output_act.shape)
        if module not in self.importances:
            self.importances[module] = {
            'weight': np.abs(10+params['weight']*grads['weight']*np.repeat(output_act.reshape(-1,1),grads['weight'].shape[1],axis=1)),
            'bias': np.abs(10+params['bias']*grads['bias']*output_act)
            }
        else:
            self.importances[module]['weight'] += np.abs(10+params['weight']*grads['weight']*np.repeat(output_act.reshape(-1,1),grads['weight'].shape[1],axis=1))
            self.importances[module]['bias'] += np.abs(10+params['bias']*grads['bias']*output_act)
    
    def model_masks(self, prunable=None, make_mask = False, next_iter = False):
        """Compute masks using the said strategy for every module
        This is a straight forward implementation that supports
        strategies that prune each module independently
        """
        if prunable is None:
            prunable = self.prunable_modules()
        for module in prunable:
            self.layer_masks(module, next_iter)
            next_iter = False
        if make_mask:
            importances = self.importances
            flat_importances = flatten_importances(importances)
            threshold = fraction_threshold(flat_importances, self.fraction)
            masks = importance_masks(importances, threshold)
            return masks


class GlobalGradCAM(ActivationMixin, GradientMixin, VisionPruning):

    def layer_masks(self, module, next_iter = False):
        # print(f'layer mask called with {module}')
        params = self.module_params(module)
        output_act = self.module_activations(module, next_iter = next_iter)
        # if output_act.shape == input_act.shape:
        #     output_act += input_act
        grads = self.module_param_gradients(module, next_iter = next_iter)
        output_act = output_act.mean(axis=0)
        output_act = output_act.mean(axis=0)
        # print('output_act:',output_act.shape)
        if module not in self.importances:
            self.importances[module] = {
            'weight': np.abs(params['weight']*grads['weight']*np.repeat(output_act.reshape(-1,1),grads['weight'].shape[1],axis=1)),
            'bias': np.abs(params['bias']*grads['bias']*output_act)
            }
        else:
            self.importances[module]['weight'] += np.abs(params['weight']*grads['weight']*np.repeat(output_act.reshape(-1,1),grads['weight'].shape[1],axis=1))
            self.importances[module]['bias'] += np.abs(params['bias']*grads['bias']*output_act)
    
    def model_masks(self, prunable=None, make_mask = False, next_iter = False):
        """Compute masks using the said strategy for every module
        This is a straight forward implementation that supports
        strategies that prune each module independently
        """
        if prunable is None:
            prunable = self.prunable_modules()
        for module in prunable:
            self.layer_masks(module, next_iter)
            next_iter = False
        if make_mask:
            importances = self.importances
            flat_importances = flatten_importances(importances)
            threshold = fraction_threshold(flat_importances, self.fraction)
            masks = importance_masks(importances, threshold)
            return masks


class LayerSmoothGrad(GradientMixin, LayerPruning, VisionPruning):

    def layer_masks(self, module, next_iter = False):
        # print(f'layer mask called with {module}')
        params = self.module_params(module)
        # output_act = self.module_activations(module, next_iter = next_iter)
        # if output_act.shape == input_act.shape:
        #     output_act += input_act
        grads = self.module_param_gradients(module, next_iter = next_iter, include_smoothing = True)
        # output_act = output_act.mean(axis=0)
        # output_act = output_act.mean(axis=0)
        # print('output_act:',output_act.shape)
        if module not in self.importances:
            self.importances[module] = {
            'weight': np.abs(params['weight']*grads['weight']),
            'bias': np.abs(params['bias']*grads['bias'])
            }
        else:
            self.importances[module]['weight'] += np.abs(params['weight']*grads['weight'])
            self.importances[module]['bias'] += np.abs(params['bias']*grads['bias'])
            
        # importances = {param: np.abs(output_act*grads[param])
        #                for param, value in params.items()}
        # print(self.fraction)
        # print(importances)
        # masks = {param: fraction_mask(importances[param], self.fraction)
        #          for param, value in params.items() if value is not None}
        # return masks
    
    def model_masks(self, prunable=None, make_mask = False, next_iter = False):
        """Compute masks using the said strategy for every module
        This is a straight forward implementation that supports
        strategies that prune each module independently
        """
        if prunable is None:
            prunable = self.prunable_modules()
        for module in prunable:
            self.layer_masks(module, next_iter)
            next_iter = False
        if make_mask:
            masks = dict()
            for module in prunable:
                masks[module] = {param: fraction_mask(self.importances[module][param], self.fraction)
                                for param, value in self.module_params(module).items() if value is not None}

            # print(self.importances)
            return masks

class LayerSmoothGradCAM(ActivationMixin, GradientMixin, LayerPruning, VisionPruning):

    def layer_masks(self, module, next_iter = False):
        # print(f'layer mask called with {module}')
        params = self.module_params(module)
        output_act = self.module_activations(module, next_iter = next_iter)
        # if output_act.shape == input_act.shape:
        #     output_act += input_act
        grads = self.module_param_gradients(module, next_iter = next_iter, include_smoothing = True)
        output_act = output_act.mean(axis=0)
        output_act = output_act.mean(axis=0)
        # print('output_act:',output_act.shape)
        if module not in self.importances:
            self.importances[module] = {
            'weight': np.abs(params['weight']*grads['weight']*np.repeat(output_act.reshape(-1,1),grads['weight'].shape[1],axis=1)),
            'bias': np.abs(params['bias']*grads['bias']*output_act)
            }
        else:
            self.importances[module]['weight'] += np.abs(params['weight']*grads['weight']*np.repeat(output_act.reshape(-1,1),grads['weight'].shape[1],axis=1))
            self.importances[module]['bias'] += np.abs(params['bias']*grads['bias']*output_act)
            
        # importances = {param: np.abs(output_act*grads[param])
        #                for param, value in params.items()}
        # print(self.fraction)
        # print(importances)
        # masks = {param: fraction_mask(importances[param], self.fraction)
        #          for param, value in params.items() if value is not None}
        # return masks
    
    def model_masks(self, prunable=None, make_mask = False, next_iter = False):
        """Compute masks using the said strategy for every module
        This is a straight forward implementation that supports
        strategies that prune each module independently
        """
        if prunable is None:
            prunable = self.prunable_modules()
        for module in prunable:
            self.layer_masks(module, next_iter)
            next_iter = False
        if make_mask:
            masks = dict()
            for module in prunable:
                masks[module] = {param: fraction_mask(self.importances[module][param], self.fraction)
                                for param, value in self.module_params(module).items() if value is not None}

            # print(self.importances)
            return masks

class GlobalSmoothGradCAM(ActivationMixin, GradientMixin, VisionPruning):

    def layer_masks(self, module, next_iter = False):
        # print(f'layer mask called with {module}')
        params = self.module_params(module)
        output_act = self.module_activations(module, next_iter = next_iter)
        # if output_act.shape == input_act.shape:
        #     output_act += input_act
        grads = self.module_param_gradients(module, next_iter = next_iter,  include_smoothing = True)
        output_act = output_act.mean(axis=0)
        output_act = output_act.mean(axis=0)
        # print('output_act:',output_act.shape)
        if module not in self.importances:
            self.importances[module] = {
            'weight': np.abs(params['weight']*grads['weight']*np.repeat(output_act.reshape(-1,1),grads['weight'].shape[1],axis=1)),
            'bias': np.abs(params['bias']*grads['bias']*output_act)
            }
        else:
            self.importances[module]['weight'] += np.abs(params['weight']*grads['weight']*np.repeat(output_act.reshape(-1,1),grads['weight'].shape[1],axis=1))
            self.importances[module]['bias'] += np.abs(params['bias']*grads['bias']*output_act)
    
    def model_masks(self, prunable=None, make_mask = False, next_iter = False):
        """Compute masks using the said strategy for every module
        This is a straight forward implementation that supports
        strategies that prune each module independently
        """
        if prunable is None:
            prunable = self.prunable_modules()
        for module in prunable:
            self.layer_masks(module, next_iter)
            next_iter = False
        if make_mask:
            importances = self.importances
            flat_importances = flatten_importances(importances)
            threshold = fraction_threshold(flat_importances, self.fraction)
            masks = importance_masks(importances, threshold)
            return masks

class LayerGradCAMShift(ActivationMixin, GradientMixin, LayerPruning, VisionPruning):

    def layer_masks(self, module, next_iter = False):
        # print(f'layer mask called with {module}')
        params = self.module_params(module)
        output_act = self.module_activations(module, next_iter = next_iter)
        # if output_act.shape == input_act.shape:
        #     output_act += input_act
        grads = self.module_param_gradients(module, next_iter = next_iter)
        output_act = output_act.mean(axis=0)
        output_act = output_act.mean(axis=0)
        # print('output_act:',output_act.shape)
        if module not in self.importances:
            self.importances[module] = {
            'weight': np.abs(params['weight']*grads['weight']*np.repeat(10+output_act.reshape(-1,1),grads['weight'].shape[1],axis=1)),
            'bias': np.abs(params['bias']*grads['bias']*(10+output_act))
            }
        else:
            self.importances[module]['weight'] += np.abs(params['weight']*grads['weight']*np.repeat(10+output_act.reshape(-1,1),grads['weight'].shape[1],axis=1))
            self.importances[module]['bias'] += np.abs(params['bias']*grads['bias']*(10+output_act))
            
        # importances = {param: np.abs(output_act*grads[param])
        #                for param, value in params.items()}
        # print(self.fraction)
        # print(importances)
        # masks = {param: fraction_mask(importances[param], self.fraction)
        #          for param, value in params.items() if value is not None}
        # return masks
    
    def model_masks(self, prunable=None, make_mask = False, next_iter = False):
        """Compute masks using the said strategy for every module
        This is a straight forward implementation that supports
        strategies that prune each module independently
        """
        if prunable is None:
            prunable = self.prunable_modules()
        for module in prunable:
            self.layer_masks(module, next_iter)
            next_iter = False
        if make_mask:
            masks = dict()
            for module in prunable:
                masks[module] = {param: fraction_mask(self.importances[module][param], self.fraction)
                                for param, value in self.module_params(module).items() if value is not None}

            # print(self.importances)
            return masks

class LayerSmoothGradCAMShift(ActivationMixin, GradientMixin, LayerPruning, VisionPruning):

    def layer_masks(self, module, next_iter = False):
        # print(f'layer mask called with {module}')
        params = self.module_params(module)
        output_act = self.module_activations(module, next_iter = next_iter)
        # if output_act.shape == input_act.shape:
        #     output_act += input_act
        grads = self.module_param_gradients(module, next_iter = next_iter, include_smoothing = True)
        output_act = output_act.mean(axis=0)
        output_act = output_act.mean(axis=0)
        # print('output_act:',output_act.shape)
        if module not in self.importances:
            self.importances[module] = {
            'weight': np.abs(params['weight']*grads['weight']*np.repeat(10+output_act.reshape(-1,1),grads['weight'].shape[1],axis=1)),
            'bias': np.abs(params['bias']*grads['bias']*(10+output_act))
            }
        else:
            self.importances[module]['weight'] += np.abs(params['weight']*grads['weight']*np.repeat(10+output_act.reshape(-1,1),grads['weight'].shape[1],axis=1))
            self.importances[module]['bias'] += np.abs(params['bias']*grads['bias']*(10+output_act))
            
        # importances = {param: np.abs(output_act*grads[param])
        #                for param, value in params.items()}
        # print(self.fraction)
        # print(importances)
        # masks = {param: fraction_mask(importances[param], self.fraction)
        #          for param, value in params.items() if value is not None}
        # return masks
    
    def model_masks(self, prunable=None, make_mask = False, next_iter = False):
        """Compute masks using the said strategy for every module
        This is a straight forward implementation that supports
        strategies that prune each module independently
        """
        if prunable is None:
            prunable = self.prunable_modules()
        for module in prunable:
            self.layer_masks(module, next_iter)
            next_iter = False
        if make_mask:
            masks = dict()
            for module in prunable:
                masks[module] = {param: fraction_mask(self.importances[module][param], self.fraction)
                                for param, value in self.module_params(module).items() if value is not None}

            # print(self.importances)
            return masks
