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
    # Data retrieving from API
    # Set up the countries
    # countries = ["COL", "USA"]

    # Set up the indicator
    # indicators = {'SP.POP.TOTL': 'population', 'SP.DYN.CDRT.IN': 'death_rate'}

    # population_united_states = wbdata.get_dataframe(indicators, country = countries, convert_date=False)['population']['United States'].sort_index()
    # t1, y1 = get_axes_values(population_united_states)

    # Data retrieving from local file
    dataset_name = "death_rate"
    country = "Colombia"
    data = pd.read_csv(f"datasets/{dataset_name}.csv")
    df = get_dataframe(data, country, 1960, 2020)
    t1, y1 = get_axes_values(df)
    t1, y1 = get_axes_values(df)

    transform(t1, -1960)

    # Train test splitting
    t1_train, y1_train = filter_by_odd_pos(t1), filter_by_odd_pos(y1)
    t1_test, y1_test = filter_by_even_pos(t1), filter_by_even_pos(y1)

    x_axis = list()
    y_axis_mean_error = list()
    y_axis_error_std = list()

    for n in range(2, 30):
        # Linear least squares computation (training stage)
        start = time.time()
        x = householder(t1_train, y1_train, n)
        end = time.time()
        elapsed = end - start

        # Polynomial evaluation
        y1_estimate = list(polynomial_evaluation(t1_test, x, n).T[0])

        # Training evaluation
        error = abs_error(y1_estimate, y1_test)
        mean_error = np.mean(error)
        error_std = np.std(error)
        x_axis.append(n)
        y_axis_mean_error.append(mean_error)
        y_axis_error_std.append(error_std)

    # Plotting
    fig1, ax1 = plt.subplots()
    ax1.plot(x_axis, y_axis_mean_error)
    ax1.set_xlabel(f'n')
    ax1.set_ylabel(f'Mean error')

    fig2, ax2 = plt.subplots()
    ax2.plot(x_axis, y_axis_error_std)
    ax2.set_xlabel(f'n')
    ax2.set_ylabel(f'Standard deviation')

    plt.show()

if __name__ == '__main__':
    main()