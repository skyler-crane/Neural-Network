import numpy as np
import random
import math
import argparse
from sys import stderr
import warnings
warnings.filterwarnings("ignore")

''' LOSS FUNCTIONS '''

sqerror_loss = lambda hx, y : (hx-y)**2

''' ACTIVATION FUNCTIONS '''

identity = lambda x:x
sigmoid = lambda x: 1/(1+np.exp(-x))
tanh = lambda x: np.tanh(x)
ReLU = lambda x: np.clip(x, 0, None)
def softmax(x):
    exp = np.exp(x)
    return exp / np.linalg.norm(exp)

''' DERIVATIVES OF ACTIVATION FUNCTIONS '''

dIdentity = lambda x: np.ones(x.shape)
def dSigmoid(x):
    a = sigmoid(x)
    return a * (1 - a)
def dTanh(x):
    a = tanh(x)
    return 1 - (a**2)
dReLU = lambda x: np.ceil(np.clip(x, 0, 1))

''' METHODS '''

'''
Returns the lines of a file as a list of strings.

Input:
    path: A string representing the path to a file.

Output:
    A list of strings representing the lines of the file.
'''
def get_lines(path):
    f = open(path, 'r')
    lines = f.readlines()
    f.close()
    return lines

'''
Loads the features from a file into an NxD array. Also loads the targets from a file into an
NxC array. 

Input:
    feats_path, targets_path: a path to a file containing N lines representing distinct datapoints
        and D space-delimited floating point or integer values per line.

Output:
    An NxD numpy array containing the features.
    An NxC numpy array containing the targets.
'''
def load_data(feats_path, targets_path):
    feats_lines = get_lines(feats_path)
    targets_lines = get_lines(targets_path)

    N = len(feats_lines)
    
    feats_split = feats_lines[0].split(' ')
    targets_split = targets_lines[0].split(' ')

    # Don't count a newline character when finding dimensions
    if feats_split[-1] == '\n':
        fD = len(feats_split) - 1
    else:
        fD = len(feats_split)
    if targets_split[-1] == '\n':
        tD = len(targets_split) - 1
    else:
        tD = len(targets_split)

    feats_data = np.empty((N,fD))
    targets_data = np.empty((N,tD))

    for n, line in enumerate(feats_lines):
        for d, val in enumerate(line.split(' ')):
            if d < fD:
                feats_data[n][d] = float(val)
    for n, line in enumerate(targets_lines):
        for d, val in enumerate(line.split(' ')):
            if d < tD:
                targets_data[n][d] = float(val)

    return feats_data, targets_data

'''
Randomly initializes weight and bias for a single layer.

Input:
    D: The input dimension
    L: The output dimension
    init_range: The value determining the range from which the weights will be initialized,
        specifically [-init_range, init_range)

Output:
    w: A DxL matrix randomly initialized within the given range.
    b: An L-dimensional vector randomly initialized within the given range.
'''
def initialize_layer(D, L, init_range):
    w = np.random.uniform(-init_range, init_range, (D,L))
    b = np.random.uniform(-init_range, init_range, (1, L))
    return w,b

'''
Initializes the mode to the given specifications.

Inputs:
    args: the output of parse_args()
    input_dim: The dimension of the input to the model.
    hidden_activation: The function to be used for the hidden layer activations.
    output_activation: The function to be used for the output layer activation.
'''
def initialize_model(args, input_dim, hidden_activation, output_activation,):
    model = []

    if args.nlayers == 0:
        if args.num_classes == 1 or args.num_classes == 2:
            w,b = initialize_layer(input_dim, 1, args.init_range)
        else:
            w,b = initialize_layer(input_dim, args.num_classes, args.init_range)
        model.append([[w,b], output_activation])
    else:
        w0, b0 = initialize_layer(input_dim, args.nunits, args.init_range)
        model.append([[w0, b0], hidden_activation])
        for l in range(args.nlayers - 1):
            w,b = initialize_layer(args.nunits, args.nunits, args.init_range)
            model.append([[w,b], hidden_activation])
        # 1D/2D Regression or binary classification
        if args.num_classes == 1 or args.num_classes == 2:
            wf,bf = initialize_layer(args.nunits, 1, args.init_range)
        else:
            wf,bf = initialize_layer(args.nunits, args.num_classes, args.init_range)
        model.append([[wf, bf], output_activation])

    return model


'''
Represents the forward pass of an input through a linear layer.

Input:
    x: A numpy array representing the input to the layer.
    wb: A list of length two. The first element is a numpy array
        representing the weights of the layer, and the second 
        element is a numpy array representing the bias of the layer.

Output:
    The numpy array resulting from xw+b

Preconditions: 
    x.shape[1] == wb[0].shape[0]
    wb[0].shape[1] == wb[1].shape[1] 
'''
def forward_linear_layer(x, wb):
    return np.matmul(x, wb[0]) + wb[1]

'''
Represents the forward pass of an input through an activation layer.

Input:
    x: A numpy array representing the input to the layer.
    act_func: An activation function.

Output:
    The result of act_func(x).
'''
def forward_activation_layer(x, act_func):
    return act_func(x)

'''
Represents the forward propogation of an input through a neural network.

Input:
    x: A numpy array representing the input to the model.
    model: A list of layers. Each layer is a list of two elements. The first element
        is a list of two elements representing the weights and biases of that layer.
        The second element in the layer is the activation function for that layer.

Output:
    A list containing lists containing the outputs from the forward pass through
    the linear layer and the outputs from the forward pass through the activation
    layer. The first element in the list is [0,x] representing the input to the model.
'''
def forward_prop(x, model):
    a = x.copy()
    preds = []
    for layer in model:
        z = forward_linear_layer(a, layer[0])
        a = forward_activation_layer(z, layer[1])
        preds.append([z,a])
    
    preds.insert(0, [0, x])
    return preds


'''
Represents the backpropagation through the model to compute the gradients
for each layer.

Inputs:
    model: A list of layers. Each layer is a list of two elements. The first element
        is a list of two elements representing the weights and biases of that layer.
        The second element in the layer is the activation function for that layer.
    preds: The output from forward_prop().
    targets: A numpy array representing what the model should have outputed in the
        second element of the final element of preds.

Output:
    A list of two elements. The first is a list of numpy arrays containing the 
    gradients for the weights of each layer. The second is a list of numpy arrays
    containing the gradients for the biases of each layer.
'''
def backprop(model, preds, targets):
    k = len(model)
    deltas = list(range(k)) 
    dLdWs = list(range(k))
    dLdBs = list(range(k))
    deltas[k-1] = (preds[-1][-1] - targets).T

    for l in range(k-1, -1, -1):
        dLdWs[l] = (preds[l][1].T @ deltas[l].T)
        dLdBs[l] = (np.ones((deltas[l].shape[1], 1)).T @ deltas[l].T)
        if l > 0:
            if model[l-1][1] == identity:
                deltas[l-1] = np.multiply(dIdentity(preds[l][0]).T, model[l][0][0] @ deltas[l])
            elif model[l-1][1] == sigmoid:
                deltas[l-1] = np.multiply(dSigmoid(preds[l][0]).T, model[l][0][0] @ deltas[l])
            elif model[l-1][1] == tanh:
                deltas[l-1] = np.multiply(dTanh(preds[l][0]).T, model[l][0][0] @ deltas[l])
            elif model[l-1][1] == ReLU:
                deltas[l-1] = np.multiply(dReLU(preds[l][0]).T, model[l][0][0] @ deltas[l])

    return [dLdWs, dLdBs]

'''
Shuffles the training data and creates batches of the desired size to be passed
into the model for training.

Input:
    train_data: A list. The first element is a numpy array representing the
        features, the second element is a numpy array representing the targets.
    batch_size: An integer representing how many training points to put in each batch.

Output:
    A list of two elements. The first is a numpy array containing features in batches of
    batch_size, the second being a numpy array containing the corresponding targets.
'''
def get_batches(train_data, batch_size):
    feats, targets = train_data
    N = feats.shape[0]
    if batch_size == 0:
        batch_size = N

    split_size = N // batch_size
    batch_length = split_size * batch_size

    zipped = list(zip(feats, targets))
    random.shuffle(zipped)
    feats, targets = zip(*zipped)

    feats_split = np.array_split(feats[:batch_length], split_size)
    targets_split = np.array_split(targets[:batch_length], split_size)

    zipped = list(zip(feats_split, targets_split))

    return zipped

'''
Performs gradient descent.

Input:
    model: A list of layers. Each layer is a list of two elements. The first element
        is a list of two elements representing the weights and biases of that layer.
        The second element in the layer is the activation function for that layer.
    gradients: The output from backprop().
    lr: A float representing the learning rate of the model.
    mb: The size of the minibatches.
    threshold: The maximum magnitude of a gradient step to allow.

This method does not return anything, rather it modifies the model in-place.
    
Preconditions:
    mb > 0
    threshold > 0
'''
def update_weights(gradients, model, lr, mb, threshold=10):
    for i, layer in enumerate(model):
        weight_step = (1/mb)*lr*gradients[0][i]
        bias_step = (1/mb)*lr*gradients[1][i]

        weight_norm = np.linalg.norm(weight_step)
        bias_norm = np.linalg.norm(bias_step)
        
        if weight_norm > threshold:
            weight_step /= weight_norm
        if bias_norm > threshold:
            bias_step /= bias_norm

        layer[0][0] -= weight_step
        layer[0][1] -= bias_step

'''
Computes the classification accuracy of the model's outputs.

Inputs:
    hx: A numpy array representing the outputs of the final activation
        layer of the model.
    y: A numpy array representing the corresponding correct target output.

Output:
    A float representing the model's accuracy on this prediction.
'''
def compute_accuracy(hx, y):
    copy = hx.copy()
    # if binary classification
    if y.shape[1] == 1:
        copy = copy.round()
        acc = copy[copy==y].shape[0]/copy.shape[0]
    # else, multiclass classification
    else:
        hx_index = np.argmax(copy, axis=1)
        y_index = np.argmax(y, axis=1)
        acc = hx_index[hx_index==y_index].shape[0] / hx_index.shape[0]

    return acc 

'''
Converts a (batch_size x 1) numpy array into a one-hot representation.

Inputs:
    targets: A (batch_size x 1) numpy array where each element is the
        class value for that target.
    D: The desired dimension for the one-hot representation.

Output:
    A (batch_size x D) numpy array of one-hot vectors representing the
    index of the target class.
'''
def targets_to_one_hot(targets, D):
    out = np.zeros((targets.shape[0], D))
    for i in range(targets.shape[0]):
        out[i, int(targets[i])] = 1.0

    return out

'''
Evaluates the predictions on the targets depending on the problem type.

Inputs:
    preds: The output from forward_prop().
    targets: A numpy array representing what the model should have outputed in the
        second element of the final element of preds.
    mode: 'C' or 'R' representing classification or regression.

Output:
    If mode=='C', the model accuracy on the input predictions.
    If mode=='R', the average loss on the input predictions.
'''
def evaluate(preds, targets, mode):
    if mode=='C':
        if preds.shape[-1] > 1:
            targets = targets_to_one_hot(targets, preds.shape[-1])
        metric = compute_accuracy(preds, targets)
    else:
        metric = loss_func(preds, targets).mean()
    return metric


'''
Represents the training loop of the model.

Inputs:
    model: A list of layers. Each layer is a list of two elements. The first element
        is a list of two elements representing the weights and biases of that layer.
        The second element in the layer is the activation function for that layer.
    loss_func: The function to use to compute the loss if doing regression, otherwise
        should be None.
    hyperparams: A dictionary containing the hyperparameters for the model. Must contain
        entries for "epochs", "batch_size", and "lr".
    data: A list of two elements. The first is a list of two elements, where the first 
        element is a numpy array representing the training features and the second element
        is a numpy array representing the training targets. The second element (of data)
        is a list containing two elements, where the first is a numpy array representing 
        the dev features, and the second is a numpy array representing the dev targets.
    mode: 'C' or 'R' representing classification or regression.
    verbose: If False, will only evaluate on dev on each epoch and print the train and dev
        results then. If True, will evaluate on dev after each model update, and print the 
        train and dev results then.

Output:
    This method does not return anything, rather it modifies the model in-place.
'''
def fit(model, loss_func, hyperparams, data, mode, verbose=False):
    update = 0
    for epoch in range(1, hyperparams["epochs"]+1):
        train_batches = get_batches(data[0], hyperparams['batch_size'])
        dev_batches = get_batches(data[1], 0)
        outputs = []
        train_metrics = []
        for feats, targets in train_batches: # train_batches: [[features, targets], [features, targets], ... ]
            preds = forward_prop(feats, model)
            truths = targets.copy()
            output_preds = preds[-1][-1]
            train_metrics.append(evaluate(output_preds, truths, mode))
            grads = backprop(model, preds, truths)
            update_weights(grads, model, hyperparams["lr"], feats.shape[0])

            if verbose:
                update += 1
                feats, targets = dev_batches[0]
                preds = forward_prop(feats, model)
                output_preds = preds[-1][-1]
                truths = targets.copy()
                dev_metric = evaluate(output_preds, truths, mode)
                print("Update {:0>6}: train={:5.3f} dev={:5.3f}".format(update, train_metrics[-1], dev_metric))

        if not verbose:
            feats, targets = dev_batches[0]
            preds = forward_prop(feats, model)
            output_preds = preds[-1][-1]
            truths = targets.copy()
            dev_metric = evaluate(output_preds, truths, mode)
            print("Epoch {:0>3}: train={:5.3f} dev={:5.3f}".format(epoch, sum(train_metrics)/len(train_metrics), dev_metric))


'''
Returns the function corresponding to the given activation function.

Input:
    hidden_act: A string representing the name of the desired function.

Output:
    The activation function.
'''
def get_hidden_activation(hidden_act):
    if hidden_act == "sig":
        activation = sigmoid
    elif hidden_act == "tanh":
        activation = tanh
    else:
        activation = ReLU
    return activation

'''
Returns the function corresponding to the necessary output activation function.

Inputs:
    mode: 'C' or 'R' representing classification or regression.
    num_classes: If mode == 'C', represents the number of classes in the classification problem.

Output:
    If mode == 'R', returns the identity function.
    If mode == 'C' and num_classes == 2, returns the sigmoid function.
    If mode == 'C' and num_classes > 2, returns the softmax function.
'''
def get_output_activation(mode, num_classes):
    if mode == 'R':
        out_activation = identity
    else:
        if num_classes == 2:
            out_activation = sigmoid
        else:
            out_activation = softmax

    return out_activation

'''
Returns the squared error loss function if the problem type is regression.
Else, returns None.

Input:
    mode: 'C' or 'R' representing classification or regression.

Output:
    If mode == 'R', the squared error loss function.
    Else, None.
'''
def get_loss_func(mode):
    if mode == 'R':
        loss_func = sqerror_loss
    else:
        loss_func = None
    return loss_func

'''
Ensures that the user entered a valid batch size.

Input:
    batch_size: Number of training points per batch.
    num_training_points: Total number of available training points.

This method does not return anything, but if the batch size was invalid
the program will exit.
'''
def batch_check(batch_size, num_training_points):
    if batch_size > num_training_points:
        print("Error: Batch size must be less or equal to total number of training points.", file=stderr)
        exit()
'''
Sets up the argparse.ArgumentParser() object and adds each of the 
necessary arguments for the program to function.
'''
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-train_feat", required=True)
    parser.add_argument("-train_target", required=True)
    parser.add_argument("-dev_feat", required=True)
    parser.add_argument("-dev_target", required=True)
    parser.add_argument("-epochs", required=True, type=int)
    parser.add_argument("-learnrate", required=True, type=float)
    parser.add_argument("-nunits", required=True, type=int)
    parser.add_argument("-type", required=True, choices=['C', 'R'])
    parser.add_argument("-hidden_act", required=True, choices=['sig', 'tanh', 'relu'])
    parser.add_argument("-init_range", required=True, type=float)

    parser.add_argument("-v", action='store_true', default=False)
    parser.add_argument("-num_classes", type=int, default=1)
    parser.add_argument("-mb", type=int, default=0)
    parser.add_argument("-nlayers", type=int, default=1)
    args = parser.parse_args()
    return args

'''
Ensures that command line arguments are valid.

Input:
    args: Output from parse_args()

This method does not return anything, but if it fails a check then 
the program will exit.
'''
def input_validation(args):
    if args.epochs < 1:
        print("Error: Invalid number of epochs.", file=stderr)
        exit()
    if args.learnrate <= 0.0:
        print("Error: Invalid learning rate.", file=stderr)
        exit()
    if args.nunits < 0 or (args.nunits == 0 and args.nlayers != 0):
        print("Error: Invalid number of hidden units.", file=stderr)
        exit()
    if args.init_range == 0.0:
        print("Error: init_range cannot be 0.", file=stderr)
        exit()
    if args.mb < 0:
        print("Error: Invalid minibatch size.", file=stderr)
    if args.num_classes <= 0:
        print("Error: Invalid number of classes.", file=stderr)
        exit()
    if args.nlayers < 0:
        print("Error: Invalid number of layers.", file=stderr)

if __name__=="__main__":

    args = parse_args()

    input_validation(args)

    train_feats, train_targets = load_data(args.train_feat, args.train_target)
    dev_feats, dev_targets = load_data(args.dev_feat, args.dev_target)

    batch_check(args.mb, len(train_feats))

    hyperparams = {
            "epochs":args.epochs, "batch_size":args.mb, "lr":args.learnrate
        }

    hidden_activation = get_hidden_activation(args.hidden_act)
    output_activation = get_output_activation(args.type, args.num_classes)
    loss_func = get_loss_func(args.type)

    model = initialize_model(args, train_feats.shape[1], hidden_activation, output_activation)

    fit(model, loss_func, hyperparams, [[train_feats, train_targets],[dev_feats, dev_targets]], args.type, args.v)
