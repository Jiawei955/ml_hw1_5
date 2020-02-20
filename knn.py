import numpy as np
import pandas as pd
import functions
import matplotlib.pyplot as plt

dftrain=pd.read_csv('compas_dataset/propublicaTrain.csv', sep=',')
dftest = pd.read_csv('compas_dataset/propublicaTest.csv',sep=',')

y_train = dftrain['two_year_recid'].values
x_train = dftrain.drop(columns=["two_year_recid"]).values

y_test = dftest["two_year_recid"].values
x_test = dftest.drop(columns=["two_year_recid"]).values

l1 = []
l2 = []
linfty = []
for i in range(1,21):
    if i==1 or i ==3 or i%5==0:
        pred = functions.knn(i, "l1", x_train, x_test, y_train)
        accuracy = 1 - np.sum(np.absolute(pred-y_test))/float(len(y_test))
        l1.append(accuracy)

        pred = functions.knn(i, "l2", x_train, x_test, y_train)
        accuracy = 1 - np.sum(np.absolute(pred - y_test)) / float(len(y_test))
        l2.append(accuracy)

        pred = functions.knn(i, "linfty", x_train, x_test, y_train)
        accuracy = 1 - np.sum(np.absolute(pred - y_test)) / float(len(y_test))
        linfty.append(accuracy)

        print("%d is done" % i)
# print(accuracy)


divisions = ["k=1","k=3","k=5","k=10","k=15","k=20"]

index = np.arange(len(divisions))
width = 0.2

plt.bar(index,l1,width,color='green',label='l1')
plt.bar(index+width,l2,width,color='yellow',label='l2')
plt.bar(index+2*width,linfty,width,color='blue',label='linfty')
plt.title("knn accuracy with different norms and number of neighbors")
plt.ylabel("accuracy")
plt.xlabel("KNNs")
plt.xticks(index+width, divisions)
plt.legend(loc='best')
plt.show()



# sex knn mle naive
# p0 = functions.check(pred,1,0,y_train,y_test)

# race knn mle naive
# p0 = functions.check(pred,1,0,y_train,y_test)
