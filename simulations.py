from sys import stdin
import wbdata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import get_axes_values, transform, get_dataframe_wbdata, get_dataframe_covid, abs_error, train_test_splitting
from interpolation import monomial_interpolation, lagrange_interpolation, newton_interpolation, piecewise_interpolation
from interpolation_evaluation import monomial_evaluation, lagrange_evaluation, newton_evaluation, piecewise_evaluation
import time

def main():
    # Data retrieving from wbdata API
    # countries = ["COL", "USA"]
    # indicators = {'SP.POP.TOTL': 'population', 'SP.DYN.CDRT.IN': 'death_rate'}
    # population_united_states = wbdata.get_dataframe(indicators, country = countries, convert_date=False)['population']['United States'].sort_index()
    # t1, y1 = get_axes_values(population_united_states)

    labels = []
    x_axis = []
    p_training = 0.05
    while(p_training <= 0.8):
        x_axis.append(p_training)
        p_training = round(p_training + 0.05, 2)
    y_axis_mean_errors = list()
    y_axis_error_stds = list()
    running_times = list()
    more_simulations = True
    dataset_name = None
    valid = False
    print("1. death_rate.csv")
    print("2. GDP.csv")
    print("3. population.csv")
    print("4. covid.csv")
    while(not valid):
        option = int(input("Please choose the dataset: "))
        if(option == 1):
            dataset_name = "death_rate"
            valid = True
        elif(option == 2):
            dataset_name = "GDP"
            valid = True
        elif(option == 3):
            dataset_name = "population"
            valid = True
        elif(option == 4):
            dataset_name = "covid"
            valid = True
        else:
            print("Invalid option. Please try again.")
    while(more_simulations):
        interpolation_method = False
        print("1. Monomial interpolation")
        print("2. Lagrange interpolation")
        print("3. Newton interpolation")
        print("4. Piecewise interpolation")
        interpolation_method = int(input("Please choose the method you want to work with: "))
        if(interpolation_method < 1 or interpolation_method > 4):
            more_simulations = False
            continue
        else:
            if(interpolation_method == 1):
                labels.append("Monomial interpolation")
            elif(interpolation_method == 2):
                labels.append("Lagrange interpolation")
            elif(interpolation_method == 3):
                labels.append("Newton interpolation")
            else:
                labels.append("Piecewise interpolation")

        # Data retrieving from local file
        data = pd.read_csv(f"datasets/{dataset_name}.csv")
        df = None
        if(dataset_name != "covid"):
            country = None
            if(dataset_name == "death_rate"):
                country = "Russian Federation" # Russian Federation, South Africa
            elif(dataset_name == "GDP"):
                country = "Colombia" # Colombia, Germany
            else:
                country = "Romania" # Romania, Curacao
            df = get_dataframe_wbdata(data, country, 1960, 2018)
        else:
            df = get_dataframe_covid(data)
        t, y = get_axes_values(df)

        if(dataset_name != "covid"):
            transform(t, -1960)

        # Train test splitting
        p_training = 0.05
        y_axis_mean_error = list()
        y_axis_error_std = list()
        running_time = list()
        while(p_training <= 0.8):
            t_train, y_train, t_test, y_test = train_test_splitting(t, y, p_training)

            # Interpolation computation
            start = time.time()
            x = None # This variable will store the result that returns each method. In the case of the piecewise interpolation, x will be a matrix of parameters
            if(interpolation_method == 1):
                x = monomial_interpolation(t_train, y_train)
            elif(interpolation_method == 2):
                x = lagrange_interpolation(t_train, y_train)
            elif(interpolation_method == 3):
                x = newton_interpolation(t_train, y_train)
            else:
                x = piecewise_interpolation(t_train, y_train)
            end = time.time()
            elapsed = end - start

            y_estimate = None
            if(interpolation_method == 1):
                # Monomial evaluation
                y_estimate = list(monomial_evaluation(t_test, x).T[0])
            elif(interpolation_method == 2):
                # Lagrange evaluation
                start = time.time()
                lagrange_evaluation(t_train, t_train, x)
                end = time.time()
                elapsed = end - start
                y_estimate = list(lagrange_evaluation(t_test, t_train, x).T[0])
            elif(interpolation_method == 3):
                # Newton evaluation
                y_estimate = list(newton_evaluation(t_test, t_train, x).T[0])
            else:
                y_estimate = piecewise_evaluation(t_test, t_train, x)

            # Training evaluation
            error = abs_error(y_estimate, y_test)
            mean_error = np.mean(error)
            error_std = np.std(error)
            y_axis_mean_error.append(mean_error)
            y_axis_error_std.append(error_std)
            running_time.append(elapsed)
            p_training = round(p_training + 0.05, 2)
        y_axis_mean_errors.append(y_axis_mean_error)
        y_axis_error_stds.append(y_axis_error_std)
        running_times.append(running_time)
    
    # for i in range(len(y_axis_mean_errors[0])):
    #     print(f"{int(x_axis[i]*100)}\% & {round(y_axis_mean_errors[0][i], 4)} & {round(y_axis_mean_errors[1][i], 4)} & {round(y_axis_mean_errors[2][i], 4)} & {round(y_axis_mean_errors[3][i], 4)}")
    #     print("\hline")

    i = 0
    fig1, ax1 = plt.subplots()
    for y_axis_mean_error in y_axis_mean_errors:
        ax1.plot(x_axis, y_axis_mean_error, label = labels[i])
        i += 1
    ax1.set_xlabel(f'Training data percentage')
    ax1.set_ylabel(f'Mean error')
    ax1.legend()

    i = 0
    fig2, ax2 = plt.subplots()
    for y_axis_error_std in y_axis_error_stds:
        ax2.plot(x_axis, y_axis_error_std, label = labels[i])
        i += 1
    ax2.set_xlabel(f'Training data percentage')
    ax2.set_ylabel(f'Standard deviation')
    ax2.legend()

    i = 0
    fig3, ax3 = plt.subplots()
    for running_time in running_times:
        ax3.plot(x_axis, running_time, label = labels[i])
        i += 1
    ax3.set_xlabel(f'Training data percentage')
    ax3.set_ylabel(f'Time (seconds)')
    ax3.legend()

    plt.show()

if __name__ == '__main__':
    main()