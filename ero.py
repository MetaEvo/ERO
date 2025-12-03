import json
import ray
import models_fn
import torch
from utils_fn import current_time, show_gpu_status
from concurrent.futures import ThreadPoolExecutor


def run_ero(es_name, ori_model_path, model_path, num_models, std, clamp_range, tasks, num_epochs, recorder, output_path, best_model_path, seed=114514):
    print(f"{es_name}-ES starts at {current_time()}!\n")
    num_gpus = torch.cuda.device_count()
    final_scores = []
    final_max_scores = []
    final_min_scores = []
    final_best_scores = []
    degen_num = 0
    for epoch in range(num_epochs):
        print(f"ES Epoch {epoch+1} starts at {current_time()}:")
        show_gpu_status()

        # es
        workers = [models_fn.ModelWorker.remote(model_path, i+1, seed+i) for i in range(num_gpus)]
        futures = []
        for i in range(num_gpus):
            n = list(range(i, num_models, num_gpus))
            futures.append(workers[i].run.remote(n, std, clamp_range, tasks, recorder))
        ray_results = ray.get(futures)

        # process data
        epoch_best_scores, epoch_scores, epoch_results = [list(res) for res in zip(*ray_results)]
        epoch_best_scores = [list(item) for item in zip(*epoch_best_scores)]
        epoch_scores = [list(item) for item in zip(*epoch_scores)]
        epoch_results = [list(item) for item in zip(*epoch_results)]
        for idx in range(len(tasks)):
            epoch_best_scores[idx] = sum(epoch_best_scores[idx], [])
            epoch_scores[idx] = sum(epoch_scores[idx], [])
            epoch_results[idx] = sum(epoch_results[idx], [])

        # save data
        data_to_save = dict()
        for i, task in enumerate(tasks):
            data_to_save[task] = dict()
            for j, score in enumerate(epoch_scores[i]):
                data_to_save[task][str(j)] = (score, epoch_results[i][j])
        with open(f'{output_path}/{es_name}_data_{current_time()}.json', 'w') as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)

        # clear
        ray_results.clear()
        futures.clear()
        data_to_save.clear()
        for i in reversed(range(len(workers))):
            ray.kill(workers[i])
            del workers[i]
        models_fn.clear_gpu_memory()
        print(f"Data saved at {current_time()}")

        # load elites
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = [
                executor.submit(torch.load, f"{model_path}/elite_{i+1}_state_dict.pth", f"cuda:{i}") for i in range(num_gpus)
            ]
        state_dicts = [f.result() for f in futures]
        models_fn.sort_models(state_dicts, epoch_best_scores)
        print(f"State dicts loaded at {current_time()}.")

        # save best
        epoch_best_scores = [list(item) for item in zip(*epoch_best_scores)]
        if sum(final_best_scores) <= sum(epoch_best_scores[0]):
            torch.save(state_dicts[0], f"{best_model_path}/{es_name}_state_dict.pth")
            final_best_scores = epoch_best_scores[0]
            print(f"Best state dict(scores: {str(final_best_scores)}) saved at {current_time()}.")
            degen_num = 0
        else:
            print(f"Best state dict(scores: {str(final_best_scores)}) NOT updated at {current_time()}.")
            degen_num += 1

        # average
        if degen_num >= 3:  # Back to original model
            degen_num = 0
            seed += 1919810
            tmp_processor, tmp_model = models_fn.load_qwen2_5_vl(ori_model_path)
            models_fn.save_model(tmp_model, model_path)
            del tmp_processor
            del tmp_model
            print(f"Reset mean model at {current_time()}.")
        else:
            models_fn.average_state_dicts_inplace(state_dicts)
            tmp_processor, tmp_model = models_fn.load_qwen2_5_vl(model_path)
            tmp_model.load_state_dict(state_dicts[0])
            models_fn.save_model(tmp_model, model_path)
            del tmp_processor
            del tmp_model
            print(f"Mean model saved at {current_time()}.")

        # clear
        futures.clear()
        for i in reversed(range(len(state_dicts))):
            del state_dicts[i]
        models_fn.clear_gpu_memory()

        # score
        # print(f"Scores: {epoch_scores}")
        tmp_scores = [sum(scores) / len(scores) for scores in zip(*epoch_scores)]
        final_scores.append(epoch_scores)
        final_min_scores.append(min(tmp_scores))
        final_max_scores.append(max(tmp_scores))

        # summary
        with open(f"{output_path}/{es_name}_scores_total.json", 'w') as f:
            json.dump(final_scores, f, indent=2, ensure_ascii=False)
        with open(f"{output_path}/{es_name}_scores_min.json", 'w') as f:
            json.dump(final_min_scores, f, indent=2, ensure_ascii=False)
        with open(f"{output_path}/{es_name}_scores_max.json", 'w') as f:
            json.dump(final_max_scores, f, indent=2, ensure_ascii=False)
        with open(f"{output_path}/{es_name}_scores_best.json", 'w') as f:
            json.dump(final_best_scores, f, indent=2, ensure_ascii=False)

        # empty line
        print()

        # early stop
        if len(final_best_scores) > 0 and (sum(final_best_scores) * 1.0 / len(final_best_scores)) >= 1.0:
            break
    return final_scores, final_min_scores, final_max_scores, final_best_scores
