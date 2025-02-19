### Environment
Before running the code, make sure you have setup the environment and installed the required packages. Make sure you meet the above requirements, and then install the dependent libraries.
```
pip install -r requirements.txt
```

### Describe Learning
DISFA+ dataset is accessible [here](http://mohammadmahoor.com/disfa-plus-request-form/)
#### Transform original images&labels into QA pairs.

TBD. (CL)
#### Learning QA pairs

For quickstart, a checkpoint of Qwen-VL trained with DISFA+ is available [Cloud](https://cloud.tsinghua.edu.cn/f/4c2d59a0f9ea4c85beb2/)
### Assessment Learning
We finetune the model on stress detection samples.
```
cd model
sh ./finetune/finetune_qlora_ds.sh
```
The results can be evaluated:
```
python inf.py
```

### Visual classification

