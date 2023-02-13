# (C) Copyright IBM Corp. 2019, 2020, 2021, 2022.

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#           http://www.apache.org/licenses/LICENSE-2.0

#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

import os

import matplotlib.pyplot as plt
import numpy as np

# In order to execute this script, it is necessary to
# set the environment variable engine as "pytorch" before initializing
# simulai
os.environ["engine"] = "pytorch"

from examples.utils.lorenz_solver import lorenz_solver
from simulai.metrics import L2Norm
from simulai.optimization import Optimizer
from simulai.regression import RBFNetwork


# This is a very basic script for exploring the PDE solving via
# feedforward fully-connected neural networks
class TestLorenzTorch:
    def __init__(self):
        pass

    def test_lorenz_torch(self):
        dt = 0.005
        T_max = 50
        rho = 28
        beta = 8 / 3
        beta_str = "8/3"
        sigma = 10

        initial_state = np.array([1, 2, 3])[None, :]
        lorenz_data, derivative_lorenz_data = lorenz_solver(
            rho=rho,
            dt=dt,
            T=T_max,
            sigma=sigma,
            initial_state=initial_state,
            beta=beta,
            beta_str=beta_str,
            data_path="on_memory",
        )

        # The fraction of data used for training the model.
        train_fraction = 0.8

        # Dumping the CSV data containing the simulation data
        # to a Pandas dataframe.
        input_data = lorenz_data
        output_data = derivative_lorenz_data

        # Choosing the number of training and testing samples
        n_samples = input_data.shape[0]
        train_samples = int(train_fraction * n_samples)
        test_samples = n_samples - train_samples

        input_labels = ["x", "y", "z"]
        output_labels = ["x_dot", "y_dot", "z_dot"]

        time = np.arange(0, n_samples, 1) * dt

        # Training dataset
        train_input_data = input_data[:train_samples]
        train_output_data = output_data[:train_samples]
        time_train = time[:train_samples]

        # Testing dataset
        test_input_data = input_data[train_samples:]
        test_output_data = output_data[train_samples:]
        time_test = time[train_samples:]

        n_inputs = len(input_labels)
        n_outputs = len(output_labels)

        lambda_1 = 1e-5  # Penalty for the L¹ regularization (Lasso)
        lambda_2 = 1e-5  # Penalty factor for the L² regularization
        n_epochs = 20_000  # Maximum number of iterations for ADAM
        lr = 1e-3  # Initial learning rate for the ADAM algorithm

        # Configuration for the fully-connected network
        config = {
            "name": "lorenz_net",
            "xmax": T_max,
            "xmin": 0,
            "Nk": 20,
            "output_size": n_outputs,
        }

        optimizer_config = {"lr": lr}
        maximum_values = (1 / np.linalg.norm(train_input_data, 2, axis=0)).tolist()
        params = {"lambda_1": lambda_1, "lambda_2": lambda_2, "weights": maximum_values}

        # Instantiating and training the surrogate model
        lorenz_net = RBFNetwork(**config)

        lorenz_net.forward(input_data=time_train[:, None])

        lorenz_net.fit(input_data=time_train[:, None], target_data=train_input_data)

        optimizer = Optimizer("adam", params=optimizer_config)

        optimizer.fit(
            op=lorenz_net,
            input_data=time_train[:, None],
            target_data=train_input_data,
            n_epochs=n_epochs,
            loss="wrmse",
            params=params,
        )

        approximated_data = lorenz_net.eval(input_data=time_train[:, None])

        l2_norm = L2Norm()

        error = 100 * l2_norm(
            data=approximated_data, reference_data=train_input_data, relative_norm=True
        )

        for ii in range(n_inputs):
            plt.plot(approximated_data[:, ii], label="Approximated")
            plt.plot(test_output_data[:, ii], label="Exact")
            plt.legend()
            plt.show()

        print(f"Approximation error for the derivatives: {error} %")
