import numpy as np
import theano
#For a more efficient optimization of the error function, we use momentum
def momentum_function(cost, params, learning_rate, momentum=0.9):
    # Create a list of gradients for all model parameters
    all_grads = theano.grad(cost, params)
    updates = []
    extra_params = []

    for param_i, grad_i in zip(params, all_grads):
        mparam_i = theano.shared(np.zeros(param_i.get_value().shape,
                                          dtype=theano.config.floatX),
                                 broadcastable=param_i.broadcastable)
        v = momentum * mparam_i - learning_rate * grad_i
        updates.append((mparam_i, v))
        updates.append((param_i, param_i + v))

    return updates
