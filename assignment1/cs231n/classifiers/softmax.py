from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        #unnormalized_probabilities
        up_list = X[i].dot(W)
        correct_class_up = up_list[y[i]] #unnom prob calculated for correct class
        loss_i= 0
        dW[:,y[i]]+= -X[i,:]
        for j in range(num_classes): 
            if j == y[i]:
                continue
            loss_i += np.exp(up_list[j])
        for j in range(num_classes): 
            if j == y[i]:
                continue
            dW[:,j]+= X[i,:]*np.exp(up_list[j])/loss_i 
        loss+= -correct_class_up + np.log(loss_i)
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW+=2*reg*W


    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]
    up_list = X.dot(W)
    correct_class_up = up_list[np.arange(num_train),y]
    loss= np.sum(-correct_class_up + np.log(np.sum(np.exp(up_list),axis=1)))/num_train
    loss += reg * np.sum(W * W)
    
    mask = np.ones((num_train,num_classes), dtype=bool)
    mask[range(num_train), y] = False
    incorrect_class_up = up_list[mask].reshape(num_train, num_classes-1)
    summation_j_exp = np.sum(np.exp(incorrect_class_up),axis=1)
    comp = np.divide(np.exp(up_list),summation_j_exp[:,np.newaxis])
    np.put_along_axis(comp, np.expand_dims(y, axis=1), -1, axis=1)
    dW = np.dot(comp.T,X)
    dW = dW.T
    dW /= num_train
    dW+=2*reg*W
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
