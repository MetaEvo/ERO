from utils_fn import initialize
initialize(enable_ray=True)

import json
import os
import shutil
from recorder import Recorder
from ero import run_ero
import argparse

def main():
    # parse
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", "-s", type=int, default=1000, help="Size of the population")
    parser.add_argument("--epochs", "-e", type=int, default=12, help="Number of evolutions")
    parser.add_argument("--scale", type=float, default=0.15, help="To scale std")
    args = parser.parse_args()

    # configs
    num_models = args.size # Size of the population
    num_epochs = args.epochs # Number of evolutions
    std_scale = args.scale # To scale std
    ori_model_path = "models/base_model" # Path to local model
    mean_model_path = "models/mean_models" # Folder used to storage the mean model when running es
    best_model_path = "models/best_models" # Folder used to save the best model when running es
    data_path = "data" # Folder used to storage tasks
    output_path = "results" # Folder used to save results
    std_path = "results/std.json" # Path to std json
    clamp_range_path = "results/clamp_range.json" # Path to clamp range json
    es_list = [
        ["Test", ["9a4bb226"]], # ["NAME FOR THIS RUN", ["LIST OF ARC TASKS"]]
    ]
    with open(std_path, 'r') as f:
        std = json.load(f)
        for name in std.keys():
            std[name] = std[name] * std_scale
    with open(clamp_range_path, 'r') as f:
        clamp_range = json.load(f)
    recorder = Recorder(output_path, data_path)

    # run ero
    for es_name, es_tasks in es_list:
        # reset mean model
        for file in os.listdir(ori_model_path):
            full_file = os.path.join(ori_model_path, file)
            if os.path.isfile(full_file):
                shutil.copy(full_file, mean_model_path)

        # run
        run_ero(es_name, ori_model_path, mean_model_path, num_models, std, clamp_range, es_tasks, num_epochs, recorder, output_path, best_model_path)
        print()


if __name__ == "__main__":
    main()
