### 0.Environment
Before running the code, make sure you have setup the environment and installed the required packages. Make sure you meet the above requirements, and then install the dependent libraries.
```
pip install -r requirements.txt
```

### 1.Describe Learning
DISFA+ dataset is accessible [here](http://mohammadmahoor.com/disfa-plus-request-form/)

if choose to employ GPT-4V to annotate:
```
cd model
python GPTannotate.py {your_api_key}
```

#### Transform original images&labels into natural language QA pairs.
Place DISFA at /data

```
cd ../data
python create_nat_annotation.py
```

#### Learning QA pairs

For quickstart, a checkpoint of Qwen-VL trained with DISFA+ is available [Cloud](https://cloud.tsinghua.edu.cn/f/4c2d59a0f9ea4c85beb2/)
### 2.Assessment Learning
We finetune the model on stress detection samples.
```
cd model
sh ./finetune/finetune_qlora_ds.sh
```
the format of data sample follows ```data/sample.json```
### 3.Evaluation and Explanation
```
sh ./eval_mm/evaluate_highlight.sh
```
Due to the setting of QLora, the code only supports up to 2 GPU devices for parallel operation.

### Visual classification
to be done
