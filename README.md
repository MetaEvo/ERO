# ERO
 
An python implementation for ERO.

![intro](intro.png)
![scores](scores.png)

### Step 1

**Install dependencies**
```
pip install -r requirements.txt
```

### Step 2

**Check Hardware conditions**

To run ERO, we require the system equipped with:

* 64 GB RAM
* 4 * 64 GB GPUs (every GPU we use is a virtual node of H20)
* 150 GB disk capacity

When additional GPUs are used, the RAM and disk capacity should be increased accordingly.

### Step 3

**Download Model**

Download the model Qwen2.5-vl-7b-instruct from huggingface and place it in the folder "models/base_model".

Download: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct

### Step 4

**Download ARC Task Data (Optional)**

We have provided some of the ARC-1 tasks we used in our experiments. If you want to use other tasks, download and put them in the folder "data". Then, modify the variable "es_list" in "main.py" accordingly.

Download: ARC-1: https://github.com/fchollet/ARC-AGI

### Step 5

**Run**

An example can be:
```
python main.py
```
which means we run ERO with default configs.

A detailed example can be:
```
python main.py --size 1000 --epochs 12 --scale 0.15
```
In this example, we use a population size of 1,000, run the evolution for 20 generations and set the scale factor to 0.15.

In addition, the results will be saved in the folder "results" and the best model will be saved in "models/best_models"

The configs can be found in "main.py". 