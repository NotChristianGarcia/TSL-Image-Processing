import numpy as np
import get_data

def grab_folded_data(k_folds):
    return get_data.kfolds(k_folds)

def batch_up_data(image_folds, label_folds, k_folds, batch_size):
    rand_fold = np.random.randint(k_folds)
    rand_start = np.random.randint(len(image_folds[0])-batch_size)
    image_batch = image_folds[rand_fold][rand_start:rand_start + batch_size]
    label_batch = label_folds[rand_fold][rand_start:rand_start + batch_size]
    return image_batch, label_batch

def L1_reg_loss(weights):
    lambda_val = 1
    regularization = np.sum(weights)
    reg_loss = lambda_val * regularization
    return reg_loss

def L2_reg_loss(weights):
    lambda_val = .0001
    regularization = np.sqrt(np.sum(np.square(weights)))
    #regularization = np.sum(np.square(weights))
    reg_loss = lambda_val * regularization
    #reg_loss = 0
    return reg_loss

def svm(image, label, weights):
    delta = 1
    scores = np.dot(weights, image)
    margins = np.maximum(0, scores - scores[int(label)] + delta) # Either 0 or scores-scores[y]+del
    margins[int(label)] = 0

    point_data_loss = np.sum(margins) # Loss before regularization
    return point_data_loss

def eval_grad(image_batch, label_batch, weights):
    data_loss = 0
    for i in range(len(image_batch)):
        data_loss += svm(image_batch[i], label_batch[i], weights)

    data_loss = data_loss/len(image_batch)

    reg_loss = L2_reg_loss(weights)

    return data_loss, reg_loss

def training(image_folds, label_folds, step_size=.001, max_iter=1000):
    batch_size = 500
    weights = np.random.rand(100, 3072) * .001 # W init
    curr_iter = 0
    while curr_iter < max_iter:
        image_batch, label_batch = batch_up_data(image_folds, label_folds, 5, batch_size)
        data_loss, reg_loss = eval_grad(image_batch, label_batch, weights)
        weights_grad = data_loss + reg_loss
        print("Iteration: {} | DLoss: {} | RLoss: {}".format(curr_iter, data_loss, reg_loss))
        weights = weights - (step_size * weights_grad)
        curr_iter += 1
    print(weights)
    return weights

#def testing(weights)

def main():
    k_folds = 5
    image_folds, label_folds = grab_folded_data(k_folds)
    made_weights = training(image_folds, label_folds)
 #   testing(made_weights)

if __name__ == "__main__":
    main()
