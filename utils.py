import  numpy as np
import pandas as pd

def get_dataframe_wbdata(data, country, start_year, end_year):
    country_index = 0
    countries = data["Country Name"]
    index_values = list()
    y = list()
    for c in countries:
        if c == country:
            break
        country_index += 1
    for year in range(start_year, end_year + 1):
        index_values.append(str(year))
        y.append([data[str(year)][country_index]])
    y = np.array(y)
    df = pd.DataFrame(data = y, index = index_values)[0]
    return df

def get_dataframe_covid(data):
    return data["C4"]
    
def get_axes_values(data):
    t = list()
    y = list()
    for year in data.keys():
        if(np.isnan(data[year]) == False):
            t.append(int(year))
            y.append(data[year])
    return t, y

def filter_by_even_pos(a):
    even = list()
    for i in range(len(a)):
        if(i % 2 == 0):
            even.append(a[i])
    return even

def filter_by_odd_pos(a):
    odd = list()
    for i in range(len(a)):
        if(i % 2 != 0):
            odd.append(a[i])
    return odd

def filter_by_indices(a, indices):
    ans = list()
    for i in indices:
        ans.append(a[i])
    return ans

def train_test_splitting(t, y, p):
    n = len(t)
    t_train, y_train, t_test, y_test = list(), list(), list(), list()
    n_train = int(p*n)
    n_test = n - n_train
    train_indices = list(map(int, np.linspace(0, n - 1, num = n_train)))
    test_indices = list()
    for i in range(n):
        if(i not in train_indices):
            test_indices.append(i)
    t_train, y_train = filter_by_indices(t, train_indices), filter_by_indices(y, train_indices)
    t_test, y_test = filter_by_indices(t, test_indices), filter_by_indices(y, test_indices)
    return t_train, y_train, t_test, y_test

def transform(a, d):
    for i in range(len(a)):
        a[i] = a[i] + d

def abs_error(a, b):
    n = len(a)
    c = [None for _ in range(n)]
    for i in range(n):
        c[i] = abs(a[i] - b[i])
    return c