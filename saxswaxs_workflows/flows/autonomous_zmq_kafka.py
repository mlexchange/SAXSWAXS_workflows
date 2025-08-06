import datetime
import json
import os
import sys

import h5py
import numpy as np
import typer
import zmq
from gpcam.gp_optimizer import GPOptimizer
from kafka import KafkaConsumer
from tiled.client import from_uri

# Initilize ZMQ socket for sending and Kafka for receiving

# Kafka Receiver
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "new_scan_entry")
KAFKA_SERVER = os.getenv("KAFKA_SERVER", "localhost")
KAFKA_PORT = os.getenv("KAFKA_PORT", "9092")
consumer = KafkaConsumer(
    KAFKA_TOPIC,
    bootstrap_servers=f"{KAFKA_SERVER}:{KAFKA_PORT}",
    # Convert JSON bytes back to dict
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
)

# ZMQ Sender
ZMQ_PORT = os.getenv("ZMQ_PORT", "5001")
ZMQ_HOST = os.getenv("ZMQ_HOST", "localhost")
# Creates a socket instance
context = zmq.Context()
# Create sockets for sending messages and receiving messages
socket_sender = context.socket(zmq.PUB)
# Bind the sender socket to its port
socket_sender.bind(f"tcp://{ZMQ_HOST}:{ZMQ_PORT}")
print("Created publisher socket on port %s" % ZMQ_PORT)

# Tiled client for accessing data
TILED_API_KEY = os.getenv("TILED_API_KEY")


# Extraction of information from a message for a very specific experiment
def extract_info_from_result_message(message):
    scan_uri = message.get("scan_uri")
    scan_client = from_uri(scan_uri, api_key=TILED_API_KEY)
    metadata = scan_client.metadata
    motor1_position = metadata["Sample X Stage"]
    motor2_position = metadata["Sample Y Stage"]

    reduced_uri = scan_uri.replace("raw", "processed")
    parts = reduced_uri.split("/")
    reduced_uri = f"{reduced_uri}/{parts[-1]}_integration-azimuthal"
    reduced_client = from_uri(reduced_uri, api_key=TILED_API_KEY)

    q = reduced_client["q"].read()
    intensity = reduced_client["intensity"].read()
    feature = q[np.argmax(intensity)]

    return motor1_position, motor2_position, feature


def instrument(x_data, motor1_name="x", motor2_name="y"):
    y_data = np.empty(len(x_data))
    for id in range(len(x_data)):
        # Ask the instrument to move to the new position
        m1_pos = x_data[id][0]
        m2_pos = x_data[id][1]
        print(f"Position suggested by gpCAM: {m1_pos}, {m2_pos}")
        # Send the next motor position to measure, will wait until returning,
        new_position = {
            "command": "PositionUpdate",
            "motors": [motor1_name, motor2_name],
            "goals": [m1_pos, m2_pos],
            "timeout_s": 0,
            "wait_for_data_saved": False,
            "time": f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
        }

        print("Sending position update to simulator: %s" % new_position)
        # Send message encoded since this is what BCSz code does
        socket_sender.send(json.dumps(new_position).encode())

        # Process incoming messages from Kafka
        for message in consumer:
            data = message.value
            # Check if message has a scan_uri key
            if "scan_uri" in data:
                print(f"Received scan URI: {data['scan_uri']}")
                break
            else:
                print(f"Received unexpected message: {data}")
        x, y, feature = extract_info_from_result_message(data)
        y_data[id] = feature
        x_data[id][0] = x
        x_data[id][1] = y

    print(f"Received from Apparatus: {y_data.reshape(-1, 1)} @ {x_data}")
    return x_data, y_data


# Setting up a user-defined acquisition function,
# but we can also use a standard one provided by gpCAM
def acq_func(x, obj):
    # a = 3.0  # 3.0 for 95 percent confidence interval
    # mean = obj.posterior_mean(x)["f(x)"]
    cov = obj.posterior_covariance(x)["v(x)"]
    return np.sqrt(cov)  # mean + a * np.sqrt(cov)


def acq_func_fwhm(x, obj):
    a = 3.0  # 3.0 for 95 percent confidence interval
    mean = obj.posterior_mean(x)["f(x)"]
    cov = obj.posterior_covariance(x)["v(x)"]
    return -mean / (a * np.sqrt(cov))


# Setting prior mean
def mean_func(x, hyperparameter, obj):
    return np.full((len(x)), 100)


def save_gpCAM_result(
    experiment_folder, x, y, counter, x_pred, num_pred_x, num_pred_y, gp
):
    mean_calulation = gp.posterior_mean(x_pred)
    mean = mean_calulation["f(x)"]
    mean = mean.reshape(num_pred_x, num_pred_y)

    variance_calculation = gp.posterior_covariance(x_pred)
    covariance = variance_calculation["v(x)"]
    covariance = covariance.reshape(num_pred_x, num_pred_y)

    gp_x = gp.x_data
    gp_y = gp.y_data
    hyperparameters = gp.hyperparameters

    current_folder = experiment_folder
    current_folder = os.path.join(current_folder, "gp")
    os.makedirs(current_folder, exist_ok=True)

    output_file_path = os.path.join(current_folder, f"gp_{counter}.h5")
    print(output_file_path)
    output_file = h5py.File(output_file_path, "w")

    output_file.create_dataset("x", data=x)
    output_file.create_dataset("y", data=y)
    output_file.create_dataset("gp_mean", data=mean)
    output_file.create_dataset("gp_covariance", data=covariance)
    output_file.create_dataset("gp_x", data=gp_x)
    output_file.create_dataset("gp_y", data=gp_y)
    output_file.create_dataset("gp_hyperparameters", data=hyperparameters)


def gp_optimizer(
    init_N, iterations, experiment_folder, acquisition_function="variance"
):
    bounds = np.array([[8, 22], [12, 26]])
    print(f"Number of points: {iterations}")

    # Number of random starting positions
    x_init = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=(init_N, 2))

    print(f"Initial random positions {x_init}")

    # Set up hyperparameters for kernel
    # first: signal variance, length scale in each parameter
    # These we might need to update, based on the data
    hps_bounds = np.array([[0.0001, 10000.0], [0.01, 100.0], [0.01, 100]])

    # x_data may be overwritten
    x_data, y_data = instrument(x_data=x_init)

    # initialize the GPOptimizer
    my_gpo = GPOptimizer(
        x_data, y_data, init_hyperparameters=np.array([50000.0, 0.5, 0.5])
    )

    # and train it
    my_gpo.train(hyperparameter_bounds=hps_bounds)
    print(f"Hyperparameters after 1st training: {my_gpo.hyperparameters}")

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

    save_gpCAM_result(
        experiment_folder, x, y, count - 1, x_pred, num_pred_x, num_pred_y, my_gpo
    )

    # control your break
    while count <= iterations:
        print("Starting new iteration")
        new_x = my_gpo.ask(bounds, n=1, acquisition_function=acquisition_function)
        print(f"New suggestion: {new_x}")
        new_x, new_y = instrument(new_x["x"])
        x_data = np.row_stack([x_data, new_x])
        y_data = np.concatenate([y_data, new_y])
        # Needs to be all the data
        my_gpo.tell(x_data, y_data)
        if count % 50 == 0:
            my_gpo.train(hyperparameter_bounds=hps_bounds)
            print(f"new hyperparameters: {my_gpo.hyperparameters}")
        save_gpCAM_result(
            experiment_folder, x, y, count, x_pred, num_pred_x, num_pred_y, my_gpo
        )

        count += 1


app = typer.Typer()


@app.command()
def main(
    experiment_folder: str = typer.Argument(
        ..., help="Directory to store gpCAM results."
    ),
    init_n: int = typer.Option(5, help="Number of initial random points."),
    acquisition_function: str = typer.Option(
        "variance", help="Acquisition function to use for optimization."
    ),
    iterations: int = typer.Option(
        100, help="Total number of iterations for optimization."
    ),
):
    if os.path.exists(experiment_folder):
        print(
            f"Error: Folder '{experiment_folder}' already exists. Choose a different name or remove it first."
        )
        sys.exit(1)

    print(
        f"Starting optimization with {init_n} points, saving results to {experiment_folder}, ",
        f"and using {acquisition_function} acquisition function.",
    )
    gp_optimizer(init_n, iterations, experiment_folder, acquisition_function)


if __name__ == "__main__":
    app()
