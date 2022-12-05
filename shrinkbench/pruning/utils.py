"""Auxiliary utils for implementing pruning strategies
"""

from collections import OrderedDict, defaultdict

import torch
from torch import nn
import transformers


def hook_applyfn(hook, model, forward=False, backward=False):
    """

    [description]

    Arguments:
        hook {[type]} -- [description]
        model {[type]} -- [description]

    Keyword Arguments:
        forward {bool} -- [description] (default: {False})
        backward {bool} -- [description] (default: {False})

    Returns:
        [type] -- [description]
    """
    assert forward ^ backward, \
        "Either forward or backward must be True"
    hooks = []

    def register_hook(module):
        if (
            not isinstance(module, nn.Sequential)
            and
            not isinstance(module, nn.ModuleList)
            and
            not isinstance(module, nn.ModuleDict)
            and
            not (module == model)
        ):
            if forward:
                hooks.append(module.register_forward_hook(hook))
            if backward:
                hooks.append(module.register_backward_hook(hook))

    return register_hook, hooks


def get_params(model, recurse=False):
    """Returns dictionary of paramters

    Arguments:
        model {torch.nn.Module} -- Network to extract the parameters from

    Keyword Arguments:
        recurse {bool} -- Whether to recurse through children modules

    Returns:
        Dict(str:numpy.ndarray) -- Dictionary of named parameters their
                                   associated parameter arrays
    """
    params = {k: v.detach().cpu().numpy().copy()
              for k, v in model.named_parameters(recurse=recurse)}
    return params


def get_activations(model, input):

    activations = OrderedDict()

    def store_activations(module, input, output):
        
        if module in activations:
            return
        assert module not in activations, \
            f"{module} already in activations"
        if type(output) == type(()):
            assert len(output)==1
            output = output[0]
        if hasattr(output, 'last_hidden_state'):
            output = output.last_hidden_state

        activations[module] = (output.detach().cpu().numpy().copy())

    fn, hooks = hook_applyfn(store_activations, model, forward=True)
    model.apply(fn)
    model = model.to('cuda:0')
    with torch.no_grad():
        model(input)

    for h in hooks:
        h.remove()

    return activations


def get_gradients(model, inputs, outputs):
    # TODO implement using model.register_backward_hook()
    # So it is harder than it seems, the grad_input contains also the gradients
    # with respect to the weights and so far order seems to be (bias, input, weight)
    # which is confusing
    # Moreover, a lot of the time the output activation we are looking for is the
    # one after the ReLU and F.ReLU (or any functional call) will not be called by
    # the forward or backward hook
    # Discussion here
    # https://discuss.pytorch.org/t/how-to-register-hook-function-for-functional-form/25775
    # Best way seems to be monkey patching F.ReLU & other functional ops
    # That'll also help figuring out how to compute a module graph
    pass


def get_param_gradients(model, inputs, outputs, loss_func=None, by_module=True, include_smoothing = False):

    gradients = OrderedDict()

    def smoothing_grads(module, input, output):

        # if type(output) == type(()):
        #     assert len(output)==1
        #     output = output[0]
        # if hasattr(output, 'last_hidden_state'):
        #     output = output.last_hidden_state
        
        if isinstance(module,torch.nn.Embedding) or isinstance(module,torch.nn.LayerNorm):
            return None
        
        if type(input) == type(()) and isinstance(module,torch.nn.Linear):
            assert len(input)==1
            input = input[0]
        
        # print(module)
        # print(input.shape)

        # if include_smoothing and hasattr(module, 'weight') and hasattr(module, 'bias'):
        #     output = output + torch.matmul((torch.randn_like(module.weight)*1e-3), input) + torch.randn_like(module.bias)*1e-3
        # elif include_smoothing and hasattr(module, 'bias'):
        #     output = output + torch.randn_like(module.bias)*1e-3
        # elif include_smoothing and hasattr(module, 'weight'):
        #     output = output + torch.matmul((torch.randn_like(module.weight)*1e-3), input)
        # elif include_smoothing:
        #     print('module without weights and biases found: ', module)

        if include_smoothing and (hasattr(module, 'weight') or hasattr(module, 'bias')):
            output = output + torch.randn_like(output)*1e-3
        
        return output
    
    if include_smoothing:
        fn, hooks = hook_applyfn(smoothing_grads, model, forward=True)
        model.apply(fn)

    if loss_func is None:
        loss_func = nn.CrossEntropyLoss()
    

    training = model.training
    model.train()
    model = model.to('cuda:0')
    outputs = outputs.to('cuda:0')
    smoothing_cycles = 10
    if include_smoothing:
        for i in range(smoothing_cycles):
            pred = model(inputs)
            loss = loss_func(pred, outputs)
            loss.backward()
    else:
        pred = model(inputs)
        loss = loss_func(pred, outputs)
        loss.backward()
    model = model.to('cpu')

    if by_module:
        gradients = defaultdict(OrderedDict)
        for module in model.modules():
            assert module not in gradients
            for name, param in module.named_parameters(recurse=False):
                if param.requires_grad and param.grad is not None:
                    if include_smoothing:
                        gradients[module][name] = param.grad.detach().cpu().numpy().copy()/smoothing_cycles
                    else:
                        gradients[module][name] = param.grad.detach().cpu().numpy().copy()

    else:
        gradients = OrderedDict()
        for name, param in model.named_parameters():
            assert name not in gradients
            if param.requires_grad and param.grad is not None:
                if include_smoothing:
                    gradients[name] = param.grad.detach().cpu().numpy().copy()/smoothing_cycles
                else:
                    gradients[name] = param.grad.detach().cpu().numpy().copy()


    model.zero_grad()
    if include_smoothing:
        for h in hooks:
            h.remove()  
    model.train(training)

    return gradients


def fraction_to_keep(compression, model, prunable_modules):
    """ Return fraction of params to keep to achieve desired compression ratio

    Compression = total / ( fraction * prunable + (total-prunable))
    Using algrebra fraction is equal to
    fraction = total/prunable * (1/compression - 1) + 1

    Arguments:
        compression {float} -- Desired overall compression
        model {torch.nn.Module} -- Full model for which to compute the fraction
        prunable_modules {List(torch.nn.Module)} -- Modules that can be pruned in the model.

    Returns:
        {float} -- Fraction of prunable parameters to keep to achieve desired compression
    """
    from ..metrics import model_size
    total_size, _ = model_size(model)
    prunable_size = sum([model_size(m)[0] for m in prunable_modules])
    nonprunable_size = total_size - prunable_size
    fraction = 1 / prunable_size * (total_size/compression - nonprunable_size)
    assert 0 < fraction <= 1, \
        f"Cannot compress to {1/compression} model with {nonprunable_size/total_size}" + \
        "fraction of unprunable parameters"
    return fraction
