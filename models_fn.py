import ray
import torch
import copy
import gc
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from utils_fn import filter_text, fill_matrix, calculate_similarity_between, initialize


def load_qwen2_5_vl(model_path: str, device_map="auto"):
    """Load a Qwen2.5 VL model and a processor from local storage."""
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No available GPUs。")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype="auto",
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()

    torch.cuda.synchronize()

    return processor, model


def clear_gpu_memory():
    """Clear GPU memory by releasing unused cache and synchronizing CUDA operations."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize()


def save_model(model, path):
    """Save model."""
    import os
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(
        path,
        safe_serialization=True,
        max_shard_size="4GB"
    )


def sample_model_from(mean_model, std, clamp_range=None, device=None):
    """Sample a new model from distribution N(mean_model, std^2)."""
    clear_gpu_memory()
    new_model = copy.deepcopy(mean_model) if device is None else copy.deepcopy(mean_model).to(device)
    with torch.no_grad():
        for name, params in new_model.named_parameters():
            params.add_(torch.randn_like(params) * torch.tensor(std[name]))

            if clamp_range is not None:
                params.clamp_(min=torch.tensor(clamp_range[name][0], device=params.device),
                              max=torch.tensor(clamp_range[name][1], device=params.device))
    return new_model


def average_state_dicts_inplace(state_dicts):
    """
    Calculate the average of state dicts on the first state dict.
    """
    n = len(state_dicts)
    assert n > 0, "Empty state dict list"
    if n == 1:
        return

    merge_device = torch.device("cuda:0")

    with torch.no_grad():
        for key in state_dicts[0].keys():
            avg = None
            for sd in state_dicts:
                t = sd[key].to(merge_device, non_blocking=True)
                if avg is None:
                    avg = t / torch.tensor(n)
                else:
                    avg += t / torch.tensor(n)

            state_dicts[0][key].copy_(avg)


def sort_models(models, scores):
    """
    Sort the models and scores in descending order based on the scores.
    """
    scores_t = [list(item) for item in zip(*scores)]
    total_scores = [sum(self_scores) for self_scores in scores_t]
    sorted_data = sorted(zip(total_scores, scores_t, models), key=lambda x: x[0], reverse=True)
    for i, (total_score, score_t, model) in enumerate(sorted_data):
        total_scores[i] = total_score
        scores_t[i] = score_t
        models[i] = model
    for i, item in enumerate(zip(*scores_t)):
        scores[i] = list(item)


def evaluate_models(models, processors, tasks, recorder):
    """
    Eval models.
    """
    assert len(models) == len(processors), "The number of models is inconsistent with the number of processors."

    def infer_single(model, processor, messages, max_new_tokens):
        model.eval()
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output_text

    prompt = lambda task: [
        {
            "role": "system",
            "content": [
                {
                    "type": "input_text",
                    "text": "You are an expert in visual abstract reasoning, specializing in solving ARC (Abstraction and Reasoning Corpus) tasks. \nYour goal is to infer the underlying transformation rule that maps each input grid to its corresponding output grid. \nThese rules may involve spatial patterns, symmetry, color changes, counting, grouping, object movements, or logical operations.\n\nGuidelines:\n1. Analyze the given input-output pairs carefully.\n2. Identify and describe the abstract transformation rule that applies across all examples.\n3. Apply the inferred rule to the test input grid to predict the correct output.\n4. Focus on high-level relational and structural patterns rather than pixel-level matching.\n5. Avoid guessing randomly. Always explain your reasoning clearly.\n6. Follow a structured reasoning process.\nOutput format (always):\n- Only output the final **Predicted Output** as a 2D array of integers.\n- Do not include any reasoning, explanations, or extra text in your response.\n- Internally, you should still perform all reasoning steps before producing the output."
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": f"Below is an ARC reasoning task.\n\nTraining examples:\nExample 1:\nInput grid:\n{str(recorder.get(task, 'input1'))}\nOutput grid:\n{str(recorder.get(task, 'output1'))}\n\nExample 2:\nInput grid:\n{str(recorder.get(task, 'input2'))}\nOutput grid:\n{str(recorder.get(task, 'output2'))}\n\nThe numerical values in the grid serve solely as identifiers for different colors. Their mathematical properties—including but not limited to arithmetic operations and comparisons—are not employed. Now, apply the same rule to the following test input grid and predict the output:\nTest input grid:\n\n{str(recorder.get(task, 'question'))}\n\nOnly output the final **Predicted Output** as a 2D array of integers.\nDo not include any reasoning, explanations, or extra text in your response.\nInternally, you should still perform all reasoning steps before producing the output."
                }
            ],
        },
    ]

    final_results = []
    final_scores = []
    for task in tasks:
        answer = filter_text(str(recorder.get(task, 'answer')))
        scores = []

        results = []
        for i, (model, processor) in enumerate(zip(models, processors)):
            response = infer_single(model, processor, prompt(task), 2 * len(answer))
            results.append(filter_text(fill_matrix(filter_text(response))))

        for result in results:
            score = calculate_similarity_between(result, answer)
            scores.append(score)

        final_scores.append(scores)
        final_results.append(results)

    return final_scores, final_results


@ray.remote(num_gpus=1)
class ModelWorker:
    def __init__(self, model_path, no, seed):
        initialize(seed)
        self.model_path = model_path
        self.no = no
        self.processor, self.base_model = load_qwen2_5_vl(model_path, "cuda:0")

    def run(self, model_no_list, std, clamp_range, tasks, recorder):
        best_model_state_dict = None
        best_model_scores = []
        final_scores = [[] for _ in range(len(tasks))]
        final_results = [[] for _ in range(len(tasks))]
        for _ in model_no_list:
            sample = sample_model_from(self.base_model, std, clamp_range)
            scores, results = evaluate_models([sample], [self.processor], tasks, recorder)
            for j in range(len(tasks)):
                final_scores[j].extend(scores[j])
                final_results[j].extend(results[j])
            if sum([sum(s) for s in best_model_scores]) <= sum([sum(s) for s in scores]):
                best_model_state_dict = {k: v.cpu() for k, v in sample.state_dict().items()}
                best_model_scores = scores
            del sample
            clear_gpu_memory()
        del self.base_model
        del self.processor
        if best_model_state_dict is not None:
            torch.save(best_model_state_dict, f"{self.model_path}/elite_{str(self.no)}_state_dict.pth")
            del best_model_state_dict
        clear_gpu_memory()
        return best_model_scores, final_scores, final_results
