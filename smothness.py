'''
Author: Zheng Li, Xiaoocng Du
Related publication: https://openaccess.thecvf.com/content_CVPRW_2020/html/w15/Du_Noise-Based_Selection_of_Robust_Inherited_Model_for_Accurate_Continual_Learning_CVPRW_2020_paper.html
Date: Spring 2020

'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def load_cvs(csv_file):
    csv_f = open(csv_file, 'r')
    data = np.loadtxt(csv_f, delimiter=',', skiprows=1)
    csv_f.close()
    
    return data

def delete_top(data, percent):
    # print(data[0:10])
    # print(type(data))
    data = np.array(sorted(data, key = lambda x: x[3]))
    # print(data[0:100])
    # print(type(data))
    n_col = int(data.shape[0] *percent)
    # print(n_col)
    return data[n_col:,:]




def fit(data, args=None):
    model = LinearRegression()
    x, y, z = data[:,1], data[:,2], data[:,3]
    x2, y2 = x*x, y*y

    X = np.stack([x2, y2, x, y], axis=1)
    print(X.shape)
    model.fit(X, z)
    z_pred = model.predict(X)
    # print(model)
    mse = mean_squared_error(z, z_pred)
    print(round(mse, 5))

files = [
        'des121.csv',
        #  'res56.csv',
        #  'res20.csv',
        #  'vgg16.csv',
        # 'res20_noshort.csv',
        # 'res56_noshort.csv',

]

if __name__ == "__main__":
    # data = load_cvs('res56_*.csv')
    # fit(data)

    # data = load_cvs('res56_noshort.csv')
    # fit(data)
    # for percentage in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for f in files:
            # print(f)
            fit(delete_top(load_cvs(f), 0.0))
        print('--------')


