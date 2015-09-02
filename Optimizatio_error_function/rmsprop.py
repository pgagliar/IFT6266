__author__ = 'pierregagliardi'
import theano
from theano import tensor as T
import numpy as np
from itertools import izip

def rmsprop(cost,params,learning_rate,rho=0.9, eps=1e-6,momentum=0.9):
	# create variables to store intermediate updates
    accumulation_r = [theano.shared(np.zeros(p.get_value().shape,
                                          dtype=theano.config.floatX),
                                          broadcastable=p.broadcastable) for p in params]

    velocity=[theano.shared(np.zeros(p.get_value().shape,
                                          dtype=theano.config.floatX),
                                          broadcastable=p.broadcastable) for p in params]

    #compute the gradient on the new spot
    params_jumps=[p+momentum*v for p,v in izip(params,velocity)]
    gradients =theano.grad(cost,params_jumps, disconnected_inputs='warn' )

    #compute the adaptative learning rate
    accumulation_r_new = [rho*r+(1-rho)*g**2 for r,g in izip(accumulation_r,gradients)]

    velocity_new = [momentum * v - (learning_rate/(T.sqrt(r)+eps))* g for v,r,g in izip(velocity,accumulation_r_new,gradients)]

	# Prepare lists of updates
    accumulation_r_updates = zip(accumulation_r,accumulation_r_new)
    velocity_updates = zip(velocity,velocity_new)
    parameters_updates = [ (p,p + v) for p,v in izip(params,velocity_new)]

    return accumulation_r_updates + velocity_updates+ parameters_updates
