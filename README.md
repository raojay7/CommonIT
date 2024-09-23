# CommonIT
Code for the EMNLP 2024 (long talk) paper: "CommonIT: Commonality-Aware Instruction Tuning for Large Language Models via Data Partitions"


### **Overview**

We propose CommonIT to enhance the instruction-following capabilities of LLMs.

<div align="center">
    <img src="llmzoo/figures/method.jpg" width=500></img>
    <p class="image-caption">CommonIT</p>
</div>


### **Installation**

```bash
git clone https://github.com/raojay7/CommonIT.git
cd CommonIT
pip install -r requirments.txt
```

### **Fine-tuning**

(1) **Dataset Preparation**: 

Our approach requires a user-defined data set to be structured into multiple datasets, which typically works best if divided precisely by task type.

(2) **Training New Models**

- [train/train_7B.sh](https://github.com/raojay7/CommonIT/tree/main/scripts/train_7B.sh)
  
The core code can be found in MultiSubsetBatchSampler and DistributedSubsetRandomSampler in train.py.


### **Evaluation**

Using open-instruct for evaluating:
[URL: https://github.com/allenai/open-instruct/tree/main](https://github.com/allenai/open-instruct/tree/main)


## **📝 Citation**<a name="citation"></a>
If you find this repo useful, please cite our paper as:
```
@inproceedings{commonIT,
    title = "CommonIT: Commonality-Aware Instruction Tuning for Large Language Models via Data Partitions",
    author = "Rao, Jun  and
      Xuebo, Liu  and
      Lian, Lian  and
      Cheng, Shengjun  and
      Liao, Yunjie  and
      Zhang, Min",
    booktitle = "EMNLP",
    year = "2024",
}
```
## Acknowledgements
The code is based on [LLMZoo](https://github.com/FreedomIntelligence/LLMZoo) and [ZeroVL](https://github.com/zerovl/ZeroVL).


