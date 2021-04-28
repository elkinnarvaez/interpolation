from sys import stdin
import wbdata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import get_axes_values, filter_by_even_pos, filter_by_odd_pos, transform, get_dataframe, abs_error, train_test_splitting
from interpolation import interpolation_gauss, interpolation_lagrange, interpolation_newton
from interpolation_evaluation import polynomial_evaluation, lagrange_evaluation, newton_evaluation
import time

def main():
    # Data retrieving from wbdata API
    # countries = ["COL", "USA"]
    # indicators = {'SP.POP.TOTL': 'population', 'SP.DYN.CDRT.IN': 'death_rate'}
    # population_united_states = wbdata.get_dataframe(indicators, country = countries, convert_date=False)['population']['United States'].sort_index()
    # t1, y1 = get_axes_values(population_united_states)

    # Data retrieving from local file
    dataset_name = "death_rate"
    country = "Colombia"
    data = pd.read_csv(f"datasets/{dataset_name}.csv")
    df = get_dataframe(data, country, 1960, 2018)
    t, y = get_axes_values(df)

    transform(t, -1960)

    # Train test splitting
    t_train, y_train, t_test, y_test = train_test_splitting(t, y)

    # Interpolation computation
    valid = False
    interpolation_method = False
    print("1. Polynomial interpolation")
    print("2. Lagrange interpolation")
    print("3. Newton interpolation")
    while(not valid):
        interpolation_method = int(input("Please choose the method you want to work with: "))
        if(interpolation_method < 1 or interpolation_method > 3):
            print("Invalid option")
        else:
            valid = True
    start = time.time()
    x = None
    if(interpolation_method == 1):
        x = interpolation_gauss(t_train, y_train)
    elif(interpolation_method == 2):
        x = interpolation_lagrange(t_train, y_train)
    else:
        x = interpolation_newton(t_train, y_train)
    end = time.time()
    elapsed = end - start

    t_continuous = None
    y_continuous = None
    y_estimate = None
    if(interpolation_method == 1):
        # Polynomial evaluation -----> Uncomment the 3 lines below if the interpolation method that was used is gauss method. Please comment these lines otherwise.
        t_continuous = list(np.linspace(t[0], t[len(t) - 1], num = 10000))
        y_continuous = list(polynomial_evaluation(t_continuous, x).T[0])
        y_estimate = list(polynomial_evaluation(t_test, x).T[0])
    elif(interpolation_method == 2):
        # Lagrange evaluation
        t_continuous = list(np.linspace(t[0], t[len(t) - 1], num = 10000))
        y_continuous = lagrange_evaluation(t_continuous, t_train, x)
        y_estimate = lagrange_evaluation(t_test, t_train, x)
    else:
        # Newton evaluation
        t_continuous = list(np.linspace(t[0], t[len(t) - 1], num = 10000))
        y_continuous = newton_evaluation(t_continuous, t_train, x)
        y_estimate = newton_evaluation(t_test, t_train, x)

    # Training evaluation
    error = abs_error(y_estimate, y_test)
    mean_error = np.mean(error)
    error_std = np.std(error)
    print(f"Time: {elapsed} sec")

    # Plotting
    transform(t_train, 1960)
    transform(t_test, 1960)
    transform(t_continuous, 1960)
    transform(t, 1960)
    plt.figure(figsize=(11, 6))
    plt.scatter(t_train, y_train, label = "Training values") # Training values
    plt.scatter(t_test, y_test, label = "Testing values") # Testing values
    # plt.scatter(t_test, y_estimate, label = "Estimate values")
    plt.plot(t_continuous, y_continuous, label = "Continuous polynomial", c='g') # Continuous polynomial
    plt.xticks(t, rotation=90)
    plt.legend()
    plt.xlabel(f'Year')
    plt.ylabel(f'Death rate (per 1000 people)')
    # x0, xmax = plt.xlim()
    y0, ymax = plt.ylim()
    plt.text(2002, y0 + (ymax - y0)/5, "Mean error: {:e} \nStandard deviation: {:e}".format(mean_error, error_std), bbox=dict(boxstyle="round",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   ))
    plt.show()

if __name__ == '__main__':
    # t, y = [-2, 0, 1], [-27, -1, 0]
    # x = interpolation_newton(t, y)
    # print(x)
    # print(newton_evaluation(t, t, x))
    main()