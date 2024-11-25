
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
torch.manual_seed(1234)
import json

import time
tokenizer = AutoTokenizer.from_pretrained("./Chat4bit", trust_remote_code=True)

# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cpu", trust_remote_code=True).eval()
# use cuda device
#model = AutoModelForCausalLM.from_pretrained(
#    "./Chat4bit",
#    device_map="auto",
#    trust_remote_code=True
#).eval()
log_file = json.load(open("./item2loss.json",'r',encoding='utf-8'))
pp = list(log_file.keys())
pplist = {}
for item in pp:
    pplist[item]=[]
counter = {}
for item in pp:
    counter[item]=0
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
    "./output_qwen_2rounds2", # path to the output directory
    device_map="auto",
    trust_remote_code=True
).eval()

import json
a = open("./test_data_bid2rounds.json",'r',encoding='utf-8')
b = json.load(a)
correct = 0
total = 0
itemname2responses=json.load(open("./sample_scoresR.json",'r'))
tcount =0 
file_lists = {}
ttdd = 0
flag = False
y_true = []
y_pred = []

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
 

start = time.time()
for item in b:


    ttdd+=1
    print(ttdd)
    print((time.time()-start)*1.0/ttdd)
    #raw

    text = item["conversations"][0]["value"]
    raw_img = text.split("</img>")[0].split("<img>")[1]
    print(raw_img,"raw_image")
    raw_que = text.split("</img>")[-1]
    raw_desp = item["conversations"][1]["value"]
    label = item["conversations"][1]["value"]
    itemname = text.split("/")[3]

    history = [(text,label.lstrip())]
    tt = label
    text = item["conversations"][2]["value"]
    label = item["conversations"][3]["value"]
    raw_appen = item["conversations"][2]["value"].split("</img>")[-1]
    response, history = model.chat(tokenizer, query=text, history=history)
    
    if response.lower().replace(".","") == "yes":
        raw_answer = 1
        y_pred.append(1)
    else:
        raw_answer = 0
        y_pred.append(0)
    
    label = label.lower()
    if "yes" in label:
        y_true.append(1)
    else:
        y_true.append(0)
    macro_f1 = f1_score(y_true, y_pred,average="binary")
    macro_acc = accuracy_score(y_true, y_pred)
    macro_rec = recall_score(y_true, y_pred,average="binary")
    macro_prec = precision_score(y_true, y_pred,average="binary")
    print("f1:",macro_f1,"acc:",macro_acc,"recall:",macro_rec,"prec:",macro_prec)
