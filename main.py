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

    valid = False
    interpolation_method = False
    print("1. Monomial interpolation")
    print("2. Lagrange interpolation")
    print("3. Newton interpolation")
    print("4. Piecewise interpolation")
    while(not valid):
        interpolation_method = int(input("Please choose the method you want to work with: "))
        if(interpolation_method < 1 or interpolation_method > 4):
            print("Invalid option. Please try again.")
        else:
            valid = True

    dataset_name = None
    x_axis_label_name = None
    y_axis_label_name = None
    if(interpolation_method != 4):
        valid = False
        print("1. death_rate.csv")
        print("2. GDP.csv")
        print("3. population.csv")
        while(not valid):
            option = int(input("Please choose the dataset: "))
            if(option == 1):
                dataset_name = "death_rate"
                valid = True
                y_axis_label_name = "Death rate (per 1000 people)"
                x_axis_label_name = "Year"
            elif(option == 2):
                dataset_name = "GDP"
                valid = True
                y_axis_label_name = "GDP (US$)"
                x_axis_label_name = "Year"
            elif(option == 3):
                dataset_name = "population"
                valid = True
                y_axis_label_name = "Population"
                x_axis_label_name = "Year"
            else:
                print("Invalid option. Please try again.")
    else:
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
                y_axis_label_name = "Death rate (per 1000 people)"
                x_axis_label_name = "Year"
            elif(option == 2):
                dataset_name = "GDP"
                valid = True
                y_axis_label_name = "GDP (US$)"
                x_axis_label_name = "Year"
            elif(option == 3):
                dataset_name = "population"
                valid = True
                y_axis_label_name = "Population"
                x_axis_label_name = "Year"
            elif(option == 4):
                dataset_name = "covid"
                valid = True
                y_axis_label_name = "Ocupación UCI Covid-19"
                x_axis_label_name = "Día"
            else:
                print("Invalid option. Please try again.")

    # Data retrieving from local file
    data = pd.read_csv(f"datasets/{dataset_name}.csv")
    df = None
    scalar_offset = 1.15
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
    valid = False
    while(not valid):
        p_training = float(input("Please indicate the percentage of the data you want to use during the training stage (a value between 0 and 1): "))
        if(p_training <= 0 or p_training >= 1):
            print("Invalid option. Please try again.")
        else:
            valid = True
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

    t_continuous = None
    y_continuous = None
    y_estimate = None
    if(interpolation_method == 1):
        # Monomial evaluation
        t_continuous = list(np.linspace(t[0], t[len(t) - 1], num = 1000))
        y_continuous = list(monomial_evaluation(t_continuous, x).T[0])
        y_estimate = list(monomial_evaluation(t_test, x).T[0])
    elif(interpolation_method == 2):
        # Lagrange evaluation
        t_continuous = list(np.linspace(t[0], t[len(t) - 1], num = 1000))
        y_continuous = list(lagrange_evaluation(t_continuous, t_train, x).T[0])
        y_estimate = list(lagrange_evaluation(t_test, t_train, x).T[0])
    elif(interpolation_method == 3):
        # Newton evaluation
        t_continuous = list(np.linspace(t[0], t[len(t) - 1], num = 1000))
        y_continuous = list(newton_evaluation(t_continuous, t_train, x).T[0])
        y_estimate = list(newton_evaluation(t_test, t_train, x).T[0])
    else:
        y_estimate = piecewise_evaluation(t_test, t_train, x)

    # Training evaluation
    error = abs_error(y_estimate, y_test)
    mean_error = np.mean(error)
    error_std = np.std(error)
    print(f"Time: {elapsed} sec")

    # Plotting
    if(interpolation_method != 4):
        transform(t_train, 1960)
        transform(t_test, 1960)
        transform(t_continuous, 1960)
        transform(t, 1960)
        plt.figure(figsize=(11, 6))
        plt.scatter(t_train, y_train, label = "Training values", s = 20) # Training values
        plt.scatter(t_test, y_test, label = "Testing values", s = 20) # Testing values
        # plt.scatter(t_test, y_estimate, label = "Estimate values")
        plt.plot(t_continuous, y_continuous, label = "Continuous polynomial", c='g') # Continuous polynomial
    else:
        # Need to graph every piece continuously
        plt.figure(figsize=(11, 6))
        for i in range(len(t_train) - 1):
            t_continuous = list(np.linspace(t_train[i], t_train[i + 1], num = 100))
            y_continuous = piecewise_evaluation(t_continuous, t_train, x)
            if(dataset_name != "covid"):
                transform(t_continuous, 1960)
            plt.plot(t_continuous, y_continuous, c='g') # Continuous polynomial
        if(dataset_name != "covid"):
            transform(t_train, 1960)
            transform(t_test, 1960)
            transform(t, 1960)
        if(dataset_name == "covid"):
            plt.scatter(t_train, y_train, label = "Training values", s = 15) # Training values
            plt.scatter(t_test, y_test, label = "Testing values", s = 15) # Testing values
        else:
            plt.scatter(t_train, y_train, label = "Training values", s = 20) # Training values
            plt.scatter(t_test, y_test, label = "Testing values", s = 20) # Testing values
        # plt.scatter(t_test, y_estimate, label = "Estimate values")
    # if(dataset_name != "covid"):
    #     plt.xticks(t, rotation=90)
    plt.legend()
    plt.xlabel(f'{x_axis_label_name}')
    plt.ylabel(f'{y_axis_label_name}')
    x0, xmax = plt.xlim()
    y0, ymax = plt.ylim()
    text_x_pos = 145 if dataset_name == "covid" else 1980
    text_y_pos = y0 + (ymax - y0)/scalar_offset
    plt.text(text_x_pos, text_y_pos, "Mean error: {:e} \nStandard deviation: {:e} \nRunning time: {:e} sec".format(mean_error, error_std, elapsed), bbox=dict(boxstyle="round",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   ))
    # plt.gcf().canvas.draw()
    # leg_pos = leg.get_window_extent()
    # plt.annotate("Mean error: {:e} \nStandard deviation: {:e} \nRunning time: {:e} sec".format(mean_error, error_std, elapsed), (leg_pos.p1[0] + 7, leg_pos.p0[1] + 11), 
    #         xycoords='figure pixels', bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.9, 0.9)), zorder=9).set_alpha(1)
    plt.show()

if __name__ == '__main__':
    # t, y = [0, 1, 2, 3, 4], [0, 1, 4, 3, 6]
    # x = piecewise_interpolation(t, y)
    # print(x)
    # print(piecewise_evaluation([1.5, 2.5], t, x))

    # t, y = [-2, 0, 1], [-27, -1, 0]
    # x = newton_interpolation(t, y)
    # print(x)
    main()