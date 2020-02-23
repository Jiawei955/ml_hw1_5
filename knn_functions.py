import numpy as np
import queue

def getDistance(x,test1,norm):
    """

    :param x: one train data
    :param test1: one test data
    :norm: the norm we use
    :return: the distance between the train data and test data
    """

    if norm == "l1":
        diff = np.absolute(x-test1)
        distance = np.sum(diff)

    elif norm == "l2":
        diff = (x-test1)**2
        distance = np.sqrt(np.sum(diff))

    else:
        diff = np.absolute(x-test1)
        distance = np.amax(diff)

    return distance


def knn(k,norm, x_train, x_test, y_train):
    """

    :param k: number of nearest neighbors
    :param norm: norm being used
    :param x_train:
    :param x_test:
    :param y_train:
    :return: the prediction of the test points in ndarray
    """
    res_array = []
    for i in range(len(x_test)):
        q = queue.PriorityQueue(k)
        ans = 0
        test_point = x_test[i]
        for j in range(len(x_train)):
            train_point = x_train[j]
            if q.full():
                largest = q.get()
                if largest[0] < (-1)*(getDistance(test_point,train_point,norm)):
                    q.put((-1 * getDistance(test_point, train_point, norm), y_train[j]))
                else:
                    q.put(largest)
            else:
                q.put((-1 * getDistance(test_point, train_point, norm), y_train[j]))

        for k_nearest_neighbors in range(k):
            l = q.get()
            ans += l[1]
        if ans<k/2.0:
            res_array.append(0)
        else:
            res_array.append(1)

        if i%200 == 0:
            print("%d test point is done" % i)

    return np.array(res_array)


def get_accuracy(pred,y_test):
    accuracy = 1 - np.sum(np.absolute(pred - y_test)) / float(len(y_test))
    return accuracy