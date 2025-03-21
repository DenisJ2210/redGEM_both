# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Filesystem Utilities for RedGEM.

This module provides utility functions for setting up directories and saving
workspace files during the RedGEM workflow.

Functions:
    - setup_directories: Create necessary directories for the workflow.
    - save_workspace: Save the state of the workspace at specific workflow steps.
"""

import os
from datetime import datetime
import logging


def setup_directories(config):
    """
    Create and organize output directories for the RedGEM workflow.

    This function ensures that the required directory structure is created based
    on the user's configuration. If directories already exist, they are preserved.

    Args:
        config (dict): A dictionary containing user-defined parameters, including:
            - output_path (str): Base path for output directories.
            - organism (str): Organism name (e.g., "ecoli").
            - gem_name (str): GEM model name (e.g., "iJO1366").

    Returns:
        dict: A dictionary containing the paths of the created directories:
            - temp_workspace: Temporary workspace directory.
            - user_output_runtimes: Directory for storing runtime logs.
            - connecting_paths_folder: Directory for storing connecting path data.
            - user_output_models: Directory for storing reduced GEM models.

    Raises:
        ValueError: If `output_path`, `organism`, or `gem_name` are missing from the configuration.
    """
    # Validate required keys
    required_keys = ["output_path", "organism", "gem_name"]
    for key in required_keys:
        if key not in config or not config[key]:
            raise ValueError(f"Missing required configuration parameter: {key}")

    # Define base paths
    base_path = config["output_path"]
    temp_workspace = os.path.join(
        base_path, "TEMP", "WorkSpaces", config["organism"], config["gem_name"]
    )
    user_output_runtimes = os.path.join(base_path, "UserOutputs", "RunTimes")
    connecting_paths_folder = os.path.join(base_path, "UserOutputs", "ConnectingPaths")
    user_output_models = os.path.join(
        base_path, "UserOutputs", "Models", config["organism"]
    )

    # Create directories
    paths = {
        "temp_workspace": temp_workspace,
        "user_output_runtimes": user_output_runtimes,
        "connecting_paths_folder": connecting_paths_folder,
        "user_output_models": user_output_models,
    }

    for path_name, path in paths.items():
        os.makedirs(path, exist_ok=True)
        logging.info(f"Created or verified directory: {path_name} -> {path}")

    return paths


def save_workspace(config, step_name):
    """
    Save the current workspace state to a timestamped file.

    This function generates a text file containing metadata about the current
    state of the workflow. It is intended for debugging or workflow recovery.

    Args:
        config (dict): A dictionary containing user-defined parameters, including:
            - output_path (str): Base path for output directories.
            - organism (str): Organism name (e.g., "ecoli").
            - gem_name (str): GEM model name (e.g., "iJO1366").
        step_name (str): A descriptive name for the current workflow step
            (e.g., "pre_lumping").

    Returns:
        str: The full path to the saved workspace file.

    Raises:
        ValueError: If `output_path`, `organism`, or `gem_name` are missing from the configuration.
    """
    # Validate required keys
    required_keys = ["output_path", "organism", "gem_name"]
    for key in required_keys:
        if key not in config or not config[key]:
            raise ValueError(f"Missing required configuration parameter: {key}")

    # Generate the timestamped filename
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")
    file_name = f"{date_str}_{time_str}_{step_name}.txt"
    save_path = os.path.join(
        config["output_path"],
        "TEMP",
        "WorkSpaces",
        config["organism"],
        config["gem_name"],
        file_name,
    )

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Write workspace details to the file
    with open(save_path, "w") as file:
        file.write("Workspace saved for debugging or reloading.\n")
        file.write(f"Date: {date_str}, Time: {time_str}\n")
        file.write(f"Organism: {config['organism']}\n")
        file.write(f"GEM Name: {config['gem_name']}\n")
        file.write(f"Step: {step_name}\n")
        logging.info(f"Workspace details written to: {save_path}")

    return save_path
