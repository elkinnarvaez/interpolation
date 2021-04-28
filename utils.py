import  numpy as np
import pandas as pd

def get_dataframe(data, country, start_year, end_year):
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

def train_test_splitting(t, y):
    t_train, y_train, t_test, y_test = list(), list(), list(), list()
    t_train, y_train = filter_by_even_pos(t), filter_by_even_pos(y)
    t_test, y_test = filter_by_odd_pos(t), filter_by_odd_pos(y)
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