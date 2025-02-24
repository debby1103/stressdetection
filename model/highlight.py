
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
torch.manual_seed(1234)

# Note: The default behavior now has injection attack prevention off.
#tokenizer = AutoTokenizer.from_pretrained("../autodl-tmp/Chat4bit", trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained("./Chat4bit", trust_remote_code=True)

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
explList = []
with open("dumped_explsegments.json",'w') as hl:
    try:
        for item in b:
            text = item["conversations"][0]["value"]
            label = item["conversations"][1]["value"]
            history = [(text,label.lstrip())]
            tt = label
            text = item["conversations"][2]["value"]
            label = item["conversations"][3]["value"]
            
            response, history = model.chat(tokenizer, query=text, history=history)
            ins3 = "Do these highlighted cues really matter to you? Can you provide facial actions that make you think the subject TOKEN stressed more faithfully? List them in descending order of significance."
            
            if response.lower().replace(".","") == "yes":
                explain,_ = model.chat(tokenizer,ins3.replace("TOKEN","is"),history=history)
            else:
                explain,_ = model.chat(tokenizer,ins3.replace("TOKEN","is not"),history=history)

            explList.append({"id":item["id"],"segments":explain.rstrip().split("\n")})
        json.dump(explList,hl)
    except:
        json.dump(explList,hl)

    

