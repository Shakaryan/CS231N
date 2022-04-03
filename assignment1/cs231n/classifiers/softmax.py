import numpy as np
#from random import shuffle

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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  num_train = X.shape[0]
  num_class = W.shape[1]
  #we use for loop alnog the training set.
  for i in range(num_train):
      Scores = np.dot(X[i], W)

      Scores -= np.max(Scores)
      sum_exponential=np.sum(np.exp(Scores))
#implmenting the loss function(summation of the exponential and it divided by max score)
      loss = loss + np.log(sum_exponential) - Scores[y[i]]
      dW[:, y[i]] -= X[i]
      Total_scores_exp = np.exp(Scores).sum()
#calculate the gradient
      for j in range(num_class):
          dW[:, j] += np.exp(Scores[j]) / Total_scores_exp * X[i]
  loss = (loss / num_train) + 0.5 * reg * np.sum(W**2)
#divide it among the num_training and add the regularized term to dw
  dW /= num_train
  dW += reg * W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  pass
  num_train = X.shape[0]
  num_class=W.shape[1]
#we vectorized it by multipying X to W.
  Scores = np.dot(X, W)
#for making the score function stablize, we to subtract matrix of score 
#with XW(due to using the vectorization implementation i allocate each step in one variable) 
  Scores -= Scores.max(axis = 1,keepdims=True).reshape(num_train, 1)
  Total_scores_exp = np.exp(Scores).sum(axis = 1)
  loss = np.log(Total_scores_exp).sum() - Scores[range(num_train), y].sum()

  counts = np.exp(Scores) / Total_scores_exp.reshape(num_train, 1)
  counts[range(num_train), y] -= 1
  dW = np.dot(X.T, counts)
#find the loss and divide it along the num_training add reg term at the end.
  loss = loss / num_train + 0.5 * reg * np.sum(W **2)
  dW = dW / num_train + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW