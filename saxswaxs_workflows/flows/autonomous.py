import os
import time

import numpy as np
import pandas as pd
from file_handling import write_gp_mean
from gpcam.gp_optimizer import GPOptimizer

from prefect import flow, task


@task(name="query_instrument")
def instrument(x_data, file_path, scan_id_range=(0, 1)):
    print("Suggested by gpCAM: ", x_data)
    y_data = np.empty(len(x_data))
    i = 0
    for id in scan_id_range:
        file_path_with_id = file_path.replace("_00000", "_" + str(id).zfill(5))
        while not os.path.exists(file_path_with_id):
            time.sleep(1)
        df = pd.read_csv(file_path_with_id)
        x_data[i] = np.array([[df["ELLI_Y"][0], df["ELLI_Z"][0]]])
        y_data[i] = df["y"][0]
        i += 1
    print("Received from Apparatus: ", y_data.reshape(-1, 1), " @ ", x_data)
    # Return values need to be a 2d array, because it could be several
    return x_data, y_data.reshape(-1, 1)


# Setting up a user-defined acquisition function,
# but we can also use a standard one provided by gpCAM
def acq_func(x, obj):
    a = 3.0  # 3.0 for 95 percent confidence interval
    mean = obj.posterior_mean(x)["f(x)"]
    cov = obj.posterior_covariance(x)["v(x)"]
    return mean + a * np.sqrt(cov)


@flow(name="autonomous_experiment")
def gp_optimizer(first_file_name):
    # bound of the input space (parameters that are controlled)
    bounds = np.array([[-5, 4.975], [-5, 4.975]])

    # Number of random starting positions
    init_N = 10
    x_init = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=(init_N, 2))

    print("Initial random positions", x_init)

    # Set up hyperparameters for kernel
    hps_bounds = np.array([[0.0001, 10.0], [0.01, 10.0], [0.01, 10]])

    # x_data may be overwritten
    x_data, y_data = instrument(
        x_init, scan_id_range=(0, init_N), file_path=first_file_name
    )

    # initialize the GPOptimizer
    my_gpo = GPOptimizer(2, bounds)

    # tell() it some data
    my_gpo.tell(x_data, y_data)

    # initialize a GP ...
    my_gpo.init_gp(np.ones(3))

    # and train it
    my_gpo.train_gp(hps_bounds)
    print("hyperparameters after 1st training: ", my_gpo.hyperparameters)

    training_list = [25, 50]  # when do you want to train?
    count = init_N

    # for prediction
    x_pred = np.zeros((134 * 134, 2))
    x = np.linspace(bounds[0, 0], bounds[0, 1], 134)
    y = np.linspace(bounds[1, 0], bounds[1, 1], 134)
    x, y = np.meshgrid(x, y)
    counter = 0
    for i in range(100):
        for j in range(100):
            x_pred[counter] = np.array([x[i, j], y[i, j]])
            counter += 1

    # control your break
    while count < 100:
        print(count)
        new_x = my_gpo.ask(
            n=1,
            acquisition_function=acq_func,
            bounds=None,  # New bounds, here: no new bounds
            pop_size=20,  # parameter for global optimizer
            max_iter=20,
            tol=1e-3,
            vectorized=False,
        )
        print("new suggestion: ", new_x)
        new_x, new_y = instrument(
            new_x["x"], scan_id_range=(count, count + 1), file_path=first_file_name
        )
        x_data = np.row_stack([x_data, new_x])
        y_data = np.row_stack([y_data, new_y])
        # Needs to be all the data
        my_gpo.tell(x_data, y_data, variances=np.ones((y_data.shape)) * 0.01)
        if count in training_list:
            my_gpo.train_gp(hps_bounds)
            print("new hyperparameters: ", my_gpo.hyperparameters)
        count += 1

        res = my_gpo.posterior_mean(x_pred)
        f = res["f(x)"]
        f = f.reshape(134, 134)

        write_gp_mean(first_file_name, counter, x, y, f, my_gpo.x_data, my_gpo.y_data)
