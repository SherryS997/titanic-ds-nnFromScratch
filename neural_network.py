import numpy as np
import copy


def sigmoid(z, prime = False):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.
    prime -- True to return the value of the derivative of sigmoid function

    Return:
    if prime:
        s -- sigmoid_prime(z)
    else:
        s -- sigmoid(z)
    """
    if prime:
        s = np.exp(z)/(1+np.exp(z))**2

        return s

    else:
        s = 1/(1+np.exp(-z))

        return s


def initialize_with_random(dim, dim2=1):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias) of type float
    """

    w = np.random.randn(dim,dim2)
    b = np.zeros((dim,1))

    return w, b


def forward_propogation(W, B, X, Layers):
    """
    This function causes the forward propogation for the neural network.

    Argument:
    W -- List of weights corresponding to the appropriate nodes and layers
    B -- Biases corresponding to the appropriate nodes and layers
    X -- data of size (features, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)
    Layers -- Number of the layers in the neural network

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    """

    Y_prediction = copy.deepcopy(X)

    for iter in range(Layers):
        Y_prediction = sigmoid(W[iter].T@Y_prediction + B[iter])

    return Y_prediction


def back_propogation(W, B, X, Y, Layers):
    """
    This function causes the back propogation for the neural network.

    Arguments:
    
    """
