import numpy as np
import pandas as pd
import functions

dftrain=pd.read_csv('compas_dataset/propublicaTrain.csv', sep=',')
dftest = pd.read_csv('compas_dataset/propublicaTest.csv',sep=',')

y_train = dftrain['two_year_recid'].values
x_train = dftrain.drop(columns=["two_year_recid"]).values

y_test = dftest["two_year_recid"].values
x_test = dftest.drop(columns=["two_year_recid"]).values

pred = functions.knn(15,"l1", x_train,x_test,y_train)
diff = np.absolute(pred-y_test)
error = np.sum(diff)/float(len(y_test))
accuracy = 1 - error
print(accuracy)










# sex knn mle naive
# p0 = functions.check(pred,1,0,y_train,y_test)

# race knn mle naive
# p0 = functions.check(pred,1,0,y_train,y_test)
