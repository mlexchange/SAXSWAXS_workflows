import os

import BCSz_noasyncio
import numpy as np
import zmq
from dotenv import load_dotenv
from file_handling import write_gp_mean_tiled
from gpcam.gp_optimizer import GPOptimizer
from tiled.client import from_uri

from prefect import flow, get_run_logger, task

load_dotenv()

# Initialize the Tiled server
TILED_API_KEY = os.getenv("TILED_API_KEY")

# Communication of motor positions
BL_SERVER = os.getenv("BL_SERVER")
BL_PORT = os.getenv("BL_PORT")
bl = BCSz_noasyncio.BCSServer()
connection_status = bl.connect(addr=BL_SERVER, port=BL_PORT)
print(connection_status)

host = "127.0.0.1"
port = "5001"

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.bind("tcp://{}:{}".format(host, port))
socket.setsockopt_string(zmq.SUBSCRIBE, "")


def extract_info_from_reduction(reduction_uri, feature_name):
    reduction_metadata = from_uri(reduction_uri, api_key=TILED_API_KEY).metadata
    # Get feature from the meta data
    feature = reduction_metadata[feature_name]
    # Get motor positions from the input_uri
    input_metadata = from_uri(reduction_metadata["input_uri"]).metadata
    x = input_metadata["Sample X Stage"]
    y = input_metadata["Sample Y Stage"]
    return x, y, feature


@task(name="query_instrument")
def instrument(x_data):
    logger = get_run_logger()
    y_data = np.empty(len(x_data))
    for id in range(len(x_data)):
        # Ask the instrument to move to the new position
        m1_pos = x_data[id][0]
        m2_pos = x_data[id][1]
        logger.info(f"Position suggested by gpCAM: {m1_pos}, {m2_pos}")
        # Send the next motor position to measure, will wait until returning,
        # Either due to time out or having received a success signal
        bl.move_all_trigger733(
            ["Sample X Stage", "Sample Y Stage"],
            [m1_pos, m2_pos],
            timeout_s=30,
            wait_for_data_saved=True,
        )
        # Wait for the reduction to be finished
        message = socket.recv_json()
        logger.info(f"Received info about new reduction {message}")
        reduction_uri = message["reduction_uri"]
        x, y, feature = extract_info_from_reduction(reduction_uri, "max_intensity")
        y_data[id] = feature
        x_data[id][0] = x
        x_data[id][1] = y

    logger.info(f"Received from Apparatus: {y_data.reshape(-1, 1)} @ {x_data}")
    # Return values need to be a 2d array, because it could be several
    return x_data, y_data.reshape(-1, 1)


# Setting up a user-defined acquisition function,
# but we can also use a standard one provided by gpCAM
def acq_func(x, obj):
    # a = 3.0  # 3.0 for 95 percent confidence interval
    # mean = obj.posterior_mean(x)["f(x)"]
    cov = obj.posterior_covariance(x)["v(x)"]
    return np.sqrt(cov)  # mean + a * np.sqrt(cov)


# Setting prior mean
def mean_func(x, hyperparameter, obj):
    return np.full((len(x)), 100)


@task(name="write_gp_mean")
def save_posterior_mean(
    experiment_uri, x, y, counter, x_pred, num_pred_x, num_pred_y, obj
):
    res = obj.posterior_mean(x_pred)
    f = res["f(x)"]
    f = f.reshape(num_pred_x, num_pred_y)
    write_gp_mean_tiled(
        experiment_uri, counter, x, y, f, obj.x_data, obj.y_data, obj.hyperparameters
    )


@flow(name="autonomous_experiment")
def gp_optimizer(init_N, iterations, experiment_uri):
    # bound of the input space (parameters that are controlled)
    logger = get_run_logger()

    # bounds = np.array([[-54, -70], [12, 27]])
    # bounds = np.array([[8, 22], [12, 26]])
    # Colloids sample 14
    # bounds = np.array([[-26, -31], [43, 46]])
    # Colloids sample 13
    bounds = np.array([[-41.34, -45.35], [44.23, 46.23]])

    # Number of random starting positions
    x_init = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=(init_N, 2))

    logger.info(f"Initial random positions {x_init}")

    # Set up hyperparameters for kernel
    # first: signal variance, length scale in each parameter
    hps_bounds = np.array([[0.0001, 10000.0], [0.01, 100.0], [0.01, 100]])

    # x_data may be overwritten
    x_data, y_data = instrument(x_data=x_init)

    # initialize the GPOptimizer
    my_gpo = GPOptimizer(2, bounds)

    # tell() it some data
    my_gpo.tell(x_data, y_data)

    # initialize a GP ...
    # my_gpo.init_gp(np.ones(3), gp_mean_function=mean_func)
    my_gpo.init_gp(np.ones(3))

    # and train it
    my_gpo.train_gp(hps_bounds)
    logger.info(f"Hyperparameters after 1st training: {my_gpo.hyperparameters}")

    training_list = [25, 50, 75]  # when do you want to train?
    count = init_N + 1

    # for prediction
    num_pred_x = 100
    num_pred_y = 100
    x_pred = np.zeros((num_pred_x * num_pred_y, 2))
    x = np.linspace(bounds[0, 0], bounds[0, 1], num_pred_x)
    y = np.linspace(bounds[1, 0], bounds[1, 1], num_pred_y)
    x, y = np.meshgrid(x, y)
    pred_counter = 0
    for i in range(num_pred_x):
        for j in range(num_pred_y):
            x_pred[pred_counter] = np.array([x[i, j], y[i, j]])
            pred_counter += 1

    # control your break
    while count < iterations:
        print(count)
        new_x = my_gpo.ask(
            n=1,
            acquisition_function="variance",  # acq_func,
            bounds=None,  # New bounds, here: no new bounds
            pop_size=20,  # parameter for global optimizer
            max_iter=20,
            tol=1e-3,
        )
        logger.info(f"New suggestion: {new_x}")
        new_x, new_y = instrument(new_x["x"])
        x_data = np.row_stack([x_data, new_x])
        y_data = np.row_stack([y_data, new_y])
        # Needs to be all the data
        my_gpo.tell(x_data, y_data, variances=np.ones((y_data.shape)) * 0.01)
        if count in training_list:
            my_gpo.train_gp(hps_bounds)
            logger.info(f"new hyperparameters: {my_gpo.hyperparameters}")
        save_posterior_mean(
            experiment_uri, x, y, count, x_pred, num_pred_x, num_pred_y, my_gpo
        )

        count += 1


if __name__ == "__main__":
    experiment_uri = (
        "http://127.0.0.1:8888/api/v1/metadata/processed/Autonomous/S13_21nm_24nm_1_1"
    )
    init_N = 25
    iterations = 500
    gp_optimizer(init_N, iterations, experiment_uri)
