import numpy as np


def checkEO(prediction,val,x_test,y_test):
    """

    :param prediction: the prediction array given by the classifier
    :param val: val of sensitive attribute
    :param y_train:
    :param y_test:
    :return:
    """
    indexs = np.argwhere(x_test[:,2]==val)
    indexs = np.squeeze(indexs)
    pred = prediction[indexs]
    true_label = y_test[indexs]

    # true positive
    real_positive_index = np.argwhere(true_label[:]==1)
    num_of_positive = len(real_positive_index)
    pred_positive = pred[real_positive_index]
    real_positive = true_label[real_positive_index]
    tpr = 1-np.sum(np.absolute(real_positive-pred_positive))/float(num_of_positive)

    # false positive
    real_negative_index = np.argwhere(true_label[:]==0)
    num_of_negative = len(real_negative_index)
    pred_negative = pred[real_negative_index]
    real_negative = true_label[real_negative_index]
    fpr =  np.sum(np.absolute(pred_negative - real_negative)) / float(num_of_negative)
    return tpr,fpr


def checkPP(prediction,val,x_test,y_test):
    """

    :param prediction: the prediction array given by the classifier
    :param val: val of sensitive attribute
    :param y_train:
    :param y_test:
    :return:
    """
    # print("predicetion shape is " , prediction.shape)
    indexs = np.argwhere(x_test[:, 2] == val)
    indexs = np.squeeze(indexs)
    # print("index shape is ", indexs.shape)
    pred = prediction[indexs]
    true_label = y_test[indexs]
    # print("pred shape is", pred.shape)



    # positive predictive value
    pred_positive_index = np.argwhere(pred[:] == 1)
    # print("pred_index shape is", pred_positive_index.shape)
    num_of_positive = len(pred_positive_index)
    real_case = true_label[pred_positive_index]
    # print("real_case shape is", real_case.shape)
    # print(real_case.shape)

    num_true_positive = np.count_nonzero(real_case)
    # print(num_true_positive,num_of_positive)
    ppv = num_true_positive / float(num_of_positive)

    # negative predictive value
    pred_negative_index = np.argwhere(pred[:] == 0)
    num_of_negative = len(pred_negative_index)
    real_case = true_label[pred_negative_index]
    num_true_negative = len(real_case) - np.count_nonzero(real_case)
    npv = num_true_negative / float(num_of_negative)
    return ppv,npv


def checkDP(prediction,val,x_test):
    """

    :param prediction: the prediction array given by the classifier
    :param val: val of sensitive attribute
    :param y_train:
    :param y_test:
    :return:
    """
    indexs = np.argwhere(x_test[:, 2] == val)
    indexs = np.squeeze(indexs)
    pred = prediction[indexs]
    dp = np.count_nonzero(pred)/float(len(pred))
    return dp


def diffofEO(prediction,x_test,y_test):
    tpr0,fpr0 = checkEO(prediction, 0, x_test, y_test)
    tpr1,fpr1 = checkEO(prediction, 1, x_test, y_test)
    return abs(tpr0-tpr1)+abs(fpr0-fpr1)

def diffofPP(prediction,x_test,y_test):
    ppv0, npv0 = checkPP(prediction, 0, x_test, y_test)
    ppv1, npv1 = checkPP(prediction, 1, x_test, y_test)
    # print(ppv0,ppv1,npv0,npv1)
    return abs(ppv0 - ppv1) + abs(npv0 - npv1)

def diffofDP(prediction,x_test):
    dp0 = checkDP(prediction,0,x_test)
    dp1 = checkDP(prediction,1,x_test)
    return abs(dp0-dp1)
