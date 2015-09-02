__author__ = 'pierregagliardi'
import matplotlib.pyplot as plt
import numpy

#Plot the training error and the validation error according to the number of epochs
if __name__=="__main__":
    path='/Users/pierregagliardi/DossierTravail/Programmation/PythonPath/ProjetCatsVsDogs/Figures'

    train_graph=numpy.load(path+'/training_data.npy')
    validation_graph=numpy.load(path+'/validation_data.npy',)

    n_epochs=train_graph.shape[0]

    plt.plot(range(n_epochs),train_graph,color='r',label='training',marker='o')
    plt.plot(range(n_epochs),validation_graph,color='b',label='validation', marker='o')

    #ylabel
    plt.ylabel('Error rates')
    #xlabel
    plt.xlabel('Number of epochs')

    legend = plt.legend(loc='upper center')
    plt.grid(True)
    fig = plt.gcf()
    fig.savefig(path+'/train_validation_error.png')
