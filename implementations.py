import numpy as np

def standardization (tx):
    """Removes columns that have always the same value
        and standardize the matrix tx containing the data.
    
    Args:
        tx: shape=(N,D)
        
    Returns:
        standard_tx: shape=(N-E,D) matrix with standardized columns data, removing the E column with always the same data
    
    """
    dev_std=np.std(tx,0)
    null_indexes = np.where(dev_std == 0)[0]
    tx = np.delete(tx, null_indexes, axis=1)
    standard_tx=(tx-np.mean(tx,0))/np.std(tx,0)
    return standard_tx


def nan_to_mean (tx):
    """Converts the nan in the data with the average of the that parameter
    
    Args:
        tx: shape=(N,D)
    
    Returns:
        adjusted_tx: shape=(N,D) matrix where nan are substituted with averages
    
    """
    mean_columns= np.nanmean(tx, axis=0)
    nan_indexes = np.where(np.isnan(tx))
    tx[nan_indexes] = mean_columns[nan_indexes[:][1]]
    adjusted_tx=tx
    return adjusted_tx


def removing_nan_columns(tx,percentage):
    """Removes the whole columns where there is a proportion of nan higher than "percentage"
    
    Args: 
        tx: shape=(N,D) containing data
        percentage: scalar indicating which is the maximum percentage of nan accepted (in each column)
        
   Returns:
       reduced_tx: shape=(N-R,D) containing data, where R columns were removed due to excess of nan 
    
    
    """
    num_rows=len(tx)
    nan_per_column=np.sum(np.isnan(tx),axis=0)
    percentage_nan=nan_per_column/num_rows
    reduced_tx = np.delete(tx, np.where(percentage_nan>percentage), axis=1)
    return reduced_tx


def compute_loss(y, tx, w):
    """Calculate the loss using MSE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        The value of the loss (a scalar), corresponding to the input parameters w.
        try
    """

    e = y - tx @ w
    loss = (e.T @ e) / (2 * y.shape[0])
    return np.squeeze(loss)  # to ensure it returns a scalar


def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """

    e = y - tx @ w
    return -(tx.T @ e) / y.shape[0]


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """

    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Perform gradient descent using MSE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial vector of model parameters.
        max_iters: The number of iterations to perform.
        gamma: The step size.

    Returns:
        w: shape=(2, ). The computed vector of model parameters.
        loss: The final loss value.
    """

    w = initial_w

    if max_iters == 0:
        return w, compute_loss(y, tx, w)

    for _ in range(max_iters):
        # compute gradient
        grad = compute_gradient(y, tx, w)
        # update w by gradient
        w = w - gamma * grad
        # compute loss
        loss = compute_loss(y, tx, w)

    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Perform stochastic gradient descent using MSE.
    The batch size is 1.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial vector of model parameters.
        max_iters: The number of iterations to perform.
        gamma: The step size.

    Returns:
        w: shape=(2, ). The computed vector of model parameters.
        loss: The final loss value.
    """

    w = initial_w

    if max_iters == 0:
        return w, compute_loss(y, tx, w)

    for _ in range(max_iters):
        for batch_y, batch_tx in batch_iter(y, tx, batch_size=1, shuffle=True):
            # compute gradient
            grad = compute_gradient(batch_y, batch_tx, w)
            # update w by gradient
            w -= gamma * grad
            # compute loss
            loss = compute_loss(y, tx, w)

    return w, loss


def least_squares(y, tx):
    """Calculate the least squares solution.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)

    Returns:
        w: shape=(2, ). The computed vector of model parameters.
        loss: The final loss value.
    """

    lhs = tx.T.dot(tx)
    rhs = tx.T.dot(y)
    w = np.linalg.solve(lhs, rhs)
    loss = compute_loss(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Calculate the least squares solution with L2 regularization.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        lambda_: The regularization parameter.

    Returns:
        w: shape=(2, ). The computed vector of model parameters.
        loss: The final loss value.
    """

    lhs = tx.T.dot(tx) + 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    rhs = tx.T.dot(y)
    w = np.linalg.solve(lhs, rhs)
    loss = compute_loss(y, tx, w)
    return w, loss





def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """_summary_

    Args:
        y (_type_): _description_
        tx (_type_): _description_
        initial_w (_type_): _description_
        max_iters (_type_): _description_
        gamma (_type_): _description_
    """
    pass


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    pass
