import datetime
import json
import os
import sys

import h5py
import numpy as np
import typer
import zmq
from gpcam.gp_optimizer import GPOptimizer

# Initilize ZMQ sockets

# Sender + Receiver on maxwell
host_maxwell = "127.0.0.1"
port_sender_maxwell = "5002"
port_receiver_maxwell = "5001"

# Creates a socket instance
context_maxwell = zmq.Context()
# Create sockets for sending messages and receiving messages
socket_sender_maxwell = context_maxwell.socket(zmq.PUB)
socket_receiver_maxwell = context_maxwell.socket(zmq.SUB)

# Bind the sender socket to its port and connect the receiver socket to its port
socket_sender_maxwell.bind(f"tcp://{host_maxwell}:{port_sender_maxwell}")
print("Created publisher socket (maxwell) on port %s" % port_sender_maxwell)
socket_receiver_maxwell.connect(f"tcp://{host_maxwell}:{port_receiver_maxwell}")
print("Connecting to receiver socket (maxwell)at port: %s" % port_receiver_maxwell)

# Subscribe to all messages (empty filter means receive everything)
socket_receiver_maxwell.setsockopt_string(zmq.SUBSCRIBE, "")


def extract_info_from_result_message(message, feature_name):
    motor1_position = message["motor_positions"][0]
    motor2_position = message["motor_positions"][1]
    feature = message[feature_name]
    return motor1_position, motor2_position, feature


def instrument(x_data, motor1_name="motor1", motor2_name="motor2"):
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
            "motor_positions": [m1_pos, m2_pos],
        }

        print("Sending position update to haspp03: %s" % new_position)
        # Send message encoded since this is what BCSz code does
        socket_sender_maxwell.send(json.dumps(new_position).encode())

        # Wait until a new result is available
        data = socket_receiver_maxwell.recv_json()
        print(f"Received info about new reduction {data}")
        x, y, feature = extract_info_from_result_message(data, "feature")
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


def gp_optimizer(init_N, experiment_folder):
    # bound of the input space (parameters that are controlled)

    # Poller for receive timeout
    poller = zmq.Poller()
    poller.register(socket_receiver_maxwell, zmq.POLLIN)

    # === Wait for Initialization ===
    timeout_ms = 1000 * 60 * 5  # Wait for 5 min
    socks = dict(poller.poll(timeout_ms))

    if socket_receiver_maxwell in socks:
        print("maxwell is waiting for init message...")
        message = socket_receiver_maxwell.recv()
        data = json.loads(message.decode())
        print(f"Received init message: {data}")

        if data.get("command") != "gpCAMInitialization":
            raise RuntimeError(f"Unexpected command: {data.get('command')}")

    # Send acknowledgment
    ack_message = {
        "command": "gpCAMInitializationSuccess",
        "time": f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
    }
    socket_sender_maxwell.send(json.dumps(ack_message).encode())
    print("Sent gpCAMInitializationSuccess acknowledgment")

    range1 = data["motor_ranges"][0]
    range2 = data["motor_ranges"][1]
    motor1_name = data["motors"][0]
    motor2_name = data["motors"][1]
    iterations = data["number_of_points"]

    bounds = np.array([[range1[0], range1[1]], [range2[0], range2[1]]])
    print(f"Bounds for the experiment: {bounds}")
    print(f"Motors: {motor1_name}, {motor2_name}")
    print(f"Number of points: {iterations}")

    # Number of random starting positions
    x_init = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=(init_N, 2))

    print(f"Initial random positions {x_init}")

    # Set up hyperparameters for kernel
    # first: signal variance, length scale in each parameter
    # These we might need to update, based on the data
    hps_bounds = np.array(
        [
            # signal variance based on a data range of 96,000 when hitting the sample, and 148,000 when measuring air -> 52000
            # [1e3, 5e7],
            # signal variance based on a data range of 14,000 when hitting the sample, and 22,000 when measuring air -> 8000
            # [1e4, 5e7],
            [
                1e4,
                1e9,
            ],  # signal variance based on a data range of 0 when hitting the sample, and 52,000 when measuring air -> 55000
            [0.05, 25.0],  # length scale motor 1
            [0.05, 9.0],  # length scale motor 2
        ]
    )

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
        new_x = my_gpo.ask(
            bounds,
            n=1,
            acquisition_function="variance",  # acq_func,
            # acquisition_function="gradient",
        )
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
):
    if os.path.exists(experiment_folder):
        print(
            f"Error: Folder '{experiment_folder}' already exists. Choose a different name or remove it first."
        )
        sys.exit(1)

    print(
        f"Starting optimization with {init_n} points, saving results to {experiment_folder}"
    )
    gp_optimizer(init_n, experiment_folder)


if __name__ == "__main__":
    app()
