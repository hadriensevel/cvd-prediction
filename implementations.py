import numpy as np


def standardization(tx):
    """Removes columns that have always the same value
        and standardize the matrix tx containing the data.

    Args:
        tx: shape=(N,D)

    Returns:
        standard_tx: shape=(N-E,D) matrix with standardized columns data, removing the E column with always the same data
    """

    dev_std = np.std(tx, axis=0)
    null_indexes = np.where(dev_std == 0)[0]
    tx = np.delete(tx, null_indexes, axis=1)
    standard_tx = (tx - np.mean(tx, axis=0)) / np.std(tx, axis=0)
    return standard_tx, null_indexes


def nan_to_mean(tx):
    """Converts the nan in the data with the average of the parameter

    Args:
        tx: shape=(N,D)

    Returns:
        adjusted_tx: shape=(N,D) matrix where nan are substituted with averages
    """

    mean_columns = np.nanmean(tx, axis=0)
    nan_indexes = np.where(np.isnan(tx))
    adjusted_tx = tx
    adjusted_tx[nan_indexes] = mean_columns[nan_indexes[:][1]]
    return adjusted_tx


def removing_nan_columns(tx, percentage):
    """Removes the columns where there is a proportion of nan higher than given percentage

     Args:
         tx: shape=(N,D) containing data
         percentage: scalar indicating which is the maximum percentage of nan accepted (in each column)

    Returns:
        reduced_tx: shape=(N-R,D) containing data, where R columns were removed due to excess of nan
    """

    num_rows = len(tx)
    nan_per_column = np.sum(np.isnan(tx), axis=0)
    percentage_nan = nan_per_column / num_rows
    index_removed_columns = np.where(
        percentage_nan > percentage
    )  # to confront with test data
    reduced_tx = np.delete(tx, np.where(percentage_nan > percentage), axis=1)
    return reduced_tx, index_removed_columns


# ----------------------------- Linear Regression -----------------------------


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

    Example:

     Number of batches = 9

     Batch size = 7                              Remainder = 3
     v     v                                         v v
    |-------|-------|-------|-------|-------|-------|---|
        0       7       14      21      28      35   max batches = 6

    If shuffle is False, the returned batches are the ones started from the indexes:
    0, 7, 14, 21, 28, 35, 0, 7, 14

    If shuffle is True, the returned batches start in:
    7, 28, 14, 35, 14, 0, 21, 28, 7

    To prevent the remainder datapoints from ever being taken into account, each of the shuffled indexes is added a random amount
    8, 28, 16, 38, 14, 0, 22, 28, 9

    This way batches might overlap, but the returned batches are slightly more representative.

    Disclaimer: To keep this function simple, individual datapoints are not shuffled. For a more random result consider using a batch_size of 1.

    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """

    data_size = len(y)  # Number of data points.
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(
        data_size / batch_size
    )  # The maximum amount of non-overlapping batches that can be extracted from the data.
    remainder = (
        data_size - max_batches * batch_size
    )  # Points that would be excluded if no overlap is allowed.

    if shuffle:
        # Generate an array of indexes indicating the start of each batch
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Add an random offset to the start of each batch to eventually consider the remainder points
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # If no shuffle is done, the array of indexes is circular.
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start  # The first data point of the batch
        end_index = (
            start_index + batch_size
        )  # The first data point of the following batch
        yield y[start_index:end_index], tx[start_index:end_index]


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


# ----------------------------- Logistic Regression -----------------------------


def sigmoid(t):
    """Apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array
    """

    t = np.clip(t, -500, 500)
    return 1 / (1 + np.exp(-t))


def compute_loss_neg_log(y, tx, w):
    """Compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss (scalar)
    """

    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    epsilon = 1e-8
    pred = np.clip(sigmoid(tx.dot(w)), epsilon, 1 - epsilon)

    return (
        -np.sum(y * np.log(pred) + (1 - y) * np.log(1 - pred))
        / y.shape[0]
    )


def compute_gradient_neg_log(y, tx, w):
    """Compute the gradient of loss by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)
    """

    return tx.T @ (sigmoid(tx @ w) - y) / y.shape[0]


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Perform logistic regression using gradient descent.

    Args:
        y: shape=(N, 1)
        tx: shape=(N, D)
        initial_w: shape=(D, 1). The initial vector of model parameters.
        max_iters: The number of iterations to perform.
        gamma: The step size.

    Returns:
        w: shape=(D, 1). The computed vector of model parameters.
        loss: The final loss value.
    """

    w = initial_w

    if max_iters == 0:
        return w, compute_loss_neg_log(y, tx, w)

    for _ in range(max_iters):
        # compute gradient
        grad = compute_gradient_neg_log(y, tx, w)
        # update w by gradient
        w = w - gamma * grad
        # compute loss
        loss = compute_loss_neg_log(y, tx, w)

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, threshold=1e-8):
    """Perform regularized logistic regression using gradient descent.

    Args:
        y: shape=(N, 1)
        tx: shape=(N, D)
        lambda_: The regularization parameter.
        initial_w: shape=(D, 1). The initial vector of model parameters.
        max_iters: The number of iterations to perform.
        gamma: The step size.
        threshold: The stopping criterion threshold.

    Returns:
        w: shape=(D, 1). The computed vector of model parameters.
        loss: The final loss value.
    """

    w = initial_w
    loss = compute_loss_neg_log(y, tx, w)

    if max_iters == 0:
        return w, loss

    for _ in range(max_iters):
        # Store previous loss
        prev_loss = loss

        # compute gradient
        grad = compute_gradient_neg_log(y, tx, w) + 2 * lambda_ * w

        # update w by gradient
        w = w - gamma * grad

        # compute loss
        loss = compute_loss_neg_log(y, tx, w)

        # Check stopping criterion
        if abs(prev_loss - loss) < threshold:
            break

    return w, loss

