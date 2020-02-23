import numpy as np
import pandas as pd
import knn_functions
import matplotlib.pyplot as plt
import check_fairness


dftrain = pd.read_csv('compas_dataset/propublicaTrain.csv', sep=',')
dftest = pd.read_csv('compas_dataset/propublicaTest.csv',sep=',')

y_train = dftrain['two_year_recid'].values
x_train = dftrain.drop(columns=["two_year_recid"]).values
y_test = dftest["two_year_recid"].values
x_test = dftest.drop(columns=["two_year_recid"]).values


# knn_accuracy = []
# training_size = [2000,2250,500,2750,3000,3250,3500,3750,4168]
# for i in training_size:
#     prediction = knn_functions.knn(k,norm, x_train, x_test, y_train)
#     x = knn_functions.get_accuracy(prediction,y_test)
#     knn_accuracy.append(x)

# check fairness
# pred = knn_functions.knn(30, "l2", x_train, x_test, y_train)
# dp = check_fairness.diffofDP(pred,x_test)
# eo = check_fairness.diffofEO(pred,x_test,y_test)
# pp = check_fairness.diffofPP(pred,x_test,y_test)
# print(dp,eo,pp)



l1 = []
l2 = []
linfty = []

choices_of_k = [1,3,5,10,20,30,40,50]
# choices_of_k = [1]
for i in choices_of_k:
    pred = knn_functions.knn(i, "l1", x_train, x_test, y_train)
    accuracy = knn_functions.get_accuracy(pred,y_test)
    l1.append(accuracy)

    pred = knn_functions.knn(i, "l2", x_train, x_test, y_train)
    accuracy = knn_functions.get_accuracy(pred, y_test)
    l2.append(accuracy)

    pred = knn_functions.knn(i, "linfty", x_train, x_test, y_train)
    accuracy = knn_functions.get_accuracy(pred,y_test)
    linfty.append(accuracy)

    print("%d is done" % i)
    # print(accuracy)






# figure1 plot different kinds of KNN
plt.figure(1)
divisions = ["k=1","k=3","k=5","k=10","k=20","k=30","k=40","k=50"]
index = np.arange(len(divisions))
width = 0.2
plt.ylim(0.55,0.7)
plt.bar(index,l1,width,color='green',label='l1')
plt.bar(index+width,l2,width,color='red',label='l2')
plt.bar(index+2*width,linfty,width,color='blue',label='linfty')
plt.title("knn accuracy with different norms and number of neighbors")
plt.ylabel("accuracy")
plt.xlabel("KNNs")
plt.xticks(index+width, divisions)


plt.legend(loc='best')




# figure2 plot the comparison of MLE, naives bayes, and the KNN
# plt.figure(2)
# num_training = [2000,2250,2500,2750,3000,3250,3500,3750,4168]
# mleaccuracy = [0.6255, 0.6275, 0.629, 0.6315, 0.6345000000000001, 0.6355, 0.6345000000000001, 0.6355, 0.635]
# naive_accuracy = [0.5920000000000001, 0.5880000000000001, 0.587, 0.596, 0.6074999999999999, 0.604, 0.606, 0.61, 0.604]
# knn_accuracy =
# plt.plot(num_training,mleaccuracy,'g',label = 'MLE')
# plt.plot(num_training,naive_accuracy,'r',label = 'naive_bayes')
# plt.plot(num_training,knn_accuracy,'y',label = 'KNN')
# plt.ylabel("accuracy")
# plt.xlabel("number of training examples")
# plt.title("comparison between MLE,naive,and KNN")
# plt.legend(loc='best')
plt.grid(True,color='k')
plt.savefig('foo2.png')
plt.show()

