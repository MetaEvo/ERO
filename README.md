# ERO
 
An python implementation for ERO.

### Requirements
```
pip install -r requirements.txt
```

### Before Running

* **System Requirements**

To run ERO, we require the system to be equipped with at least 64 GB of main memory, two 64 GB GPUs, and 150 GB of local disk space. When additional GPUs are used, the main memory and disk capacity should be increased accordingly.

* **Download Model**

Download the model Qwen2.5-vl-7b-instruct from huggingface and place it in the folder "models/base_model".

Download: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct

* **Download ARC (Optional)**

We have provided one of the ARC-1 tasks we used in our experiments. If you want to use other tasks, download and put them in the folder "data".

Download: ARC-1: https://github.com/fchollet/ARC-AGI

### Running

An example can be:
```
python main.py
```
which means we run ERO with default configs.

A detailed example can be:
```
python main.py --size 500 --epochs 12 --scale 0.15
```
In this example, we use a population size of 1,000, run the evolution for 20 generations and set the scale factor to 0.15.

In addition, the results will be saved in the folder "results" and the best model will be saved in "models/best_models"

The configs can be found in "main.py". 