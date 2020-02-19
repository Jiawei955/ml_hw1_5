import numpy as np

def check(prediction,sen_attr,val,y_train,y_test):
    """

    :param prediction: the prediction array given by the classifier
    :param sen_attr:
    :param val: val of sensitive attribute
    :param y_train:
    :param y_test:
    :return:
    """
    indexs = np.argwhere(y_train[:,sen_attr]==val)
    pred = prediction[:,indexs]
    true_label = y_test[:,indexs]
    error = np.sum(np.absolute(pred-true_label))/float(len(pred))
    accuracy = 1 - error
    return accuracy
