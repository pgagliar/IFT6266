import time
import os
import sys
import theano
import numpy
from Optimization_error_function.rmsprop import rmsprop_function
from Optimization_error_function.momentum import momentum_function
from theano import tensor as T
from Layers.LeNet_conv_poollayer import LeNetConvPoolLayer
from Layers.hidden_layer import HiddenLayer
from Layers.logistic_regression_class import LogisticRegression
from Datasets.variable_image_dataset import DogsVsCats, RandomCrop
from Figures.plot import plot_training_validation_error


#Rectified linear unit function for activation function
def relu(x):
    return theano.tensor.switch(x<0, 0, x)

if __name__=="__main__":

    rng=numpy.random.RandomState(23455)

    # Hyperparameters
    learning_rate = 0.01
    n_epochs=200
    #number of feature maps, begins at layer 1
    nkerns=numpy.array([8,16,16,16,32])
    batch_size=50
    momentum=0.9

    #Symbolic theano variables
    index=T.iscalar('index')
    x = T.tensor4(name='x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size, input channels, image_size_0*image_size_0)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    #The tuple represents (batchsize, channels, rows, columns)
    layer0_input = x.reshape((batch_size, 3, 100, 100))

    # For each convolutional pooling layer:
    # First, the convolution of the image with a filter is computed.
    # Second, relu function is applied on the results of convolution to obtain hidden units
    # Third, maxpooling is applied on the hidden units in the same neighborhood
    # layer_n.output is thus of shape (batch_size, nkerns[n], image_size_n+1, image_size_n+1)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 100, 100),
        filter_shape=(nkerns[0],3, 4, 4),
        poolsize=(2, 2)
    )


    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 52,52),
        filter_shape=(nkerns[1], nkerns[0], 4, 4),
        poolsize=(2, 2)
    )


    layer2 = LeNetConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size, nkerns[1], 28, 28),
        filter_shape=(nkerns[2], nkerns[1], 4, 4),
        poolsize=(2, 2)
    )

    layer3 = LeNetConvPoolLayer(
        rng,
        input=layer2.output,
        image_shape=(batch_size, nkerns[2], 16, 16),
        filter_shape=(nkerns[3], nkerns[2], 4, 4),
        poolsize=(2, 2)
    )


    layer4 = LeNetConvPoolLayer(
        rng,
        input=layer3.output,
        image_shape=(batch_size, nkerns[3], 10, 10),
        filter_shape=(nkerns[4], nkerns[3], 4, 4),
        poolsize=(2, 2)
    )

    # The HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape:
    # (batch_size, nkerns[last_CNN_layer] * image_size_last_CNN_layer * image_size_last_CNN_layer)
    layer5_input = layer4.output.flatten(2)

    # construct a fully-connected relu layer
    layer5 = HiddenLayer(
        rng,
        input=layer5_input,
        n_in=nkerns[4] * 7 * 7,
        n_out=256,
        activation=relu
    )

    # Construct a fully-connected relu layer
    layer6 = HiddenLayer(
        rng,
        input=layer5.output,
        n_in=256,
        n_out=256,
        activation=relu
    )

    # Classify the values of the fully-connected softmax layer
    layer7 = LogisticRegression(input=layer6.output, n_in=256, n_out=2)

    # The cost we minimize during training is the NLL of the model
    cost = layer7.negative_log_likelihood(y)

    # Create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [x,y],
        layer7.errors(y),
    )

    validate_model = theano.function(
        [x,y],
        layer7.errors(y),
    )

    # Create a list of all model parameters to be fit by gradient descent
    params = layer7.params+layer6.params + layer5.params+ layer4.params+ layer3.params + layer2.params + layer1.params + layer0.params


    updates = momentum_function(cost, params, learning_rate)

    train_model = theano.function(
        [x,y],
        cost, #Compute the error function: forward propagation
        updates=updates #All the parameters are updated according to the direction of the steepest descent
     )
    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    start_time = time.clock()
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score=0
    start_time_total = time.clock()

    epoch =0
    done_looping = False

    # We split the dataset between training set, validation set and testing set
    # We randomly crop a 100x100 window before training,
    # this technique is called data augmentation and improves generalization
    train_set = DogsVsCats(transformer=RandomCrop(256, 100),start=0, stop=20000)
    valid_set=DogsVsCats(transformer=RandomCrop(256, 100),start=20000, stop=22500)
    test_set=DogsVsCats(transformer=RandomCrop(256, 100),start=22500, stop=25000)

     # Compute number of minibatches for training, validation and testing
    n_train_batches = train_set.get_num_examples()/batch_size
    n_valid_batches = valid_set.get_num_examples()/ batch_size
    n_test_batches = test_set.get_num_examples()/ batch_size

    # Early-stopping parameters#
    ############################
    # Look as this many examples regardless
    patience = 100000
    # wait this much longer when a new best is found
    patience_increase = 2
    # A relative improvement of this much is considered significant
    improvement_threshold = 0.995
    # Go through this many
    # minibatches before checking the network
    # on the validation set; in this case we
    # check every epoch
    validation_frequency = min(n_train_batches, patience / 2)


    train_error=numpy.zeros(n_train_batches)
    train_graph=numpy.zeros(n_epochs)
    validation_graph=numpy.zeros(n_epochs)

    while (epoch < n_epochs) and (not done_looping):
        start_time_epoch = time.clock()
        epoch = epoch + 1
        it_train = train_set.iterator(mode='random_slice', batch_size=batch_size, num_batches= n_train_batches)
        it_valid = valid_set.iterator(mode='random_slice', batch_size=batch_size, num_batches= n_valid_batches)
        it_test = test_set.iterator(mode='random_slice', batch_size=batch_size, num_batches= n_test_batches)
        minibatch_index=0

        for X_train, y_train in it_train:
            y_train_reshape_cast=y_train.reshape(batch_size).astype('int32')
            minibatch_avg_cost = train_model(X_train,y_train_reshape_cast )
            train_error[minibatch_index]=minibatch_avg_cost

            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(X_valid,y_valid.reshape(batch_size).astype('int32'))
                                     for X_valid, y_valid in it_valid]
                this_validation_loss = numpy.mean(validation_losses)

                validation_graph[epoch-1]=this_validation_loss
                train_graph[epoch-1]=numpy.mean(train_error)
                end_time_epoch = time.clock()

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.

                    )
                )
                print(('     epoch %i, minibatch %i/%i, train error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           train_graph[epoch-1] * 100.)
		)
                print >> sys.stderr,('The epoch %i ran for %.2fm' %
                                       (
                                            epoch,
                                            (end_time_epoch- start_time_epoch) / 60.
                                        )
                                    )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

            if patience <= iter:
                done_looping = True
                break
            minibatch_index=minibatch_index+1

    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +os.path.split(__file__)[1] +' ran for %.2fm' % ((end_time - start_time) / 60.))

    #Save training errors and validations errors in npy file
    path='/Users/pierregagliardi/DossierTravail/Programmation/PythonPath/ProjetCatsVsDogs/Figures/'
    numpy.save(path+'training_data',train_graph)
    numpy.save(path+'validation_data',validation_graph)

    #Plot training and validation error
    plot_training_validation_error(path)
