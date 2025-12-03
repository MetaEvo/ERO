from datetime import datetime, timedelta, timezone
import torch
import ast
import re
import random
import ray


def initialize(seed=114514, dtype=torch.bfloat16, enable_ray=False):
    """Set seed, dtype and init ray."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.set_default_dtype(dtype)

    if enable_ray:
        import os
        os.environ["RAY_memory_monitor_refresh_ms"] = "0"
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
        ray.init(ignore_reinit_error=True)


def current_time():
    """
    Get current time(string). Shanghai Timezone.
    """
    utc_8 = timezone(timedelta(hours=8))
    beijing_time = datetime.now(utc_8)
    time_str = beijing_time.strftime("%Y-%m-%d_%H-%M-%S")
    return time_str


def fill_matrix(text):
    """Fill the matrix(string format) with 0 if able."""
    try:
        arr = ast.literal_eval(text)
        max_len = max(len(row) for row in arr)
        padded = [row + [0] * (max_len - len(row)) for row in arr]
        return str(padded)
    except Exception:
        return text


def filter_text(text):
    """Filter out all content except , [, ] and 0~9."""
    pattern = r'[^0-9\[\],]'
    return re.sub(pattern, '', text)


def calculate_similarity_between(response: str, answer: str):
    """
    Returns a float value between 0 and 1 representing the similarity between the two strings. 0.0 means completely different, and 1.0 means completely the same.
    """
    cnt = 0
    for idx in range(min(len(response), len(answer))):
        cnt += 1 if response[idx] == answer[idx] else 0
    return cnt * 1.0 / max(len(response), len(answer))


def show_gpu_status():
    """Displays the memory usage of each graphics card."""
    for i in range(torch.cuda.device_count()):
        print(
            f"GPU {i}: {torch.cuda.get_device_name(i)} | Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 2:.1f} MB | Reserved:  {torch.cuda.memory_reserved(i) / 1024 ** 2:.1f} MB")
    print()