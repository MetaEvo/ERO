import torch
from recorder import Recorder
from models_fn import load_qwen2_5_vl, evaluate_models
import argparse

def main():
    # parse
    parser = argparse.ArgumentParser()
    parser.add_argument("--state_dict", "-s", type=str, default="models/best_models/9a4bb226_state_dict.pth", help="Path to state dict")
    parser.add_argument("--task", "-e", type=str, default="9a4bb226", help="Path to your task")
    args = parser.parse_args()

    # configs
    model_path = "models/base_model"
    state_dict_path = args.state_dict
    processor, model = load_qwen2_5_vl(model_path)
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)
    tasks = [args.task]
    recorder = Recorder("results", "data")

    # eval
    final_scores, final_results = evaluate_models([model], [processor], tasks, recorder)

    # 打印结果
    print(f"Scores: {final_scores}")
    print(f"Results: {final_results}")

if __name__ == "__main__":
    main()