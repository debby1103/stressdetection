
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
torch.manual_seed(1234)
import json


# Note: The default behavior now has injection attack prevention off.
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
# Either a local path or an url between <img></img> tags.
#image_path = 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'
import json
a = open("./regen.json",'r',encoding='utf-8')
b = json.load(a)
correct = 0
total = 0

tcount =0 
file_lists = {}
ttdd = 0
flag = False
for item in b:


    ttdd+=1
    print(ttdd)

    #raw

    text = item["conversations"][0]["value"]
    raw_img = text.split("</img>")[0].split("<img>")[1]
    print(raw_img,"raw_image")
    raw_que = text.split("</img>")[-1]
    raw_desp = item["conversations"][1]["value"]
    label = item["conversations"][1]["value"]
  #  print(label)
    itemname = text.split("/")[3]
 #   if itemname == "23_12_stressed":
 #       flag = True
 #   if not flag:
 #       continue
   # response, history = model.chat(tokenizer, query=text, history=None)#item["conversations"][:2]
    history = [(text,label.lstrip())]
    tt = label
    text = item["conversations"][2]["value"]
    label = item["conversations"][3]["value"]
    raw_appen = item["conversations"][2]["value"].split("</img>")[-1]
    response, history = model.chat(tokenizer, query=text, history=history)#item["conversations"][:2]
    
    
    if response.lower().replace(".","") == "yes":
        raw_answer = 1
    else:
        raw_answer = 0
    

    id_name = item["conversations"][0]["value"].split("/")[3]
    try:
        part_names = log_file[id_name][counter[id_name]]
    except:
        itemname2responses[id_name]=["useless"]
        continue

    counter[id_name]+=1
    new_part_names = {}
    j_ = 0
    flip = len(part_names.keys())+1

    for i_ in range(len(part_names.keys())):
        while not str(j_)  in part_names.keys():
            j_+=1
        new_part_names[str(i_)] = part_names[str(j_)]
        j_+=1
    for i_ in range(len(new_part_names.keys())):

        decaf = new_part_names[str(i_)]
        
        real_parts = decaf.split("-de")[1].split(".jpg")[0].split("+")
        tee = {"eye":False,"nose":False,"lip":False,"jaw":False,"lip corner":False,"eyebrow":False,"mouth":False,"lid":False}
        for j__ in real_parts:
            j_ = j__.split(":")[0]
            if "lip" in j_:
                if "corner" in j_:
                    tee["lip corner"]=True
                else:
                    tee["lip"]=True
            else:
                for k_ in tee.keys():
                    if (not "lip" in k_ and k_ in j_):
                        tee[k_]=True
        #print(tee)
       

        decaf_on = decaf.replace("apex","onset")
        new_img = "Before: <img>./revised/revised/{}</img>\nAfter: <img>./revised/revised/{}</img>\n ".format(decaf_on,decaf)
        new_one = new_img+raw_que
        parts = raw_desp.split("\n")
        new_parts =[]
        for part_ in parts:
            remov = False
            if ":" in part_:
                part = part_.split(":")[0]
            else:
                part = part_.strip()

            if "lip" in part:
                if "corner" in part:
                    if tee["lip corner"]:
                        remov = True
                else:
                    if tee["lip"]:
                        remov = True
            else:
                for k_ in tee.keys():
                    if tee[k_] and k_ in part:
                        remov = True
            if not remov:
                new_parts.append(part_)
        new_parts = "=".join(new_parts)
        #print(new_parts)
        new_two = new_parts
        new_three = new_img+"According to above descriptions, the character has the following facial expressions:  \n"+new_parts+"\n"+"Determine if the character in the video is showing signs of stress based on their facial expressions."
        print(new_one,"one")
        print(new_two,"two")
        print(new_three,"three")
        response, history = model.chat(tokenizer, query=new_three, history=[(new_one,new_two)])#item["conversations"][:2]

        response = response.lower()
       # print(response)
        if "yes" in response:
            answer =1 
        else:
            answer = 0
        if raw_answer!=answer:
            flip = i_+1
            break
    try:
        value = 1.0*flip/(len(part_names.keys()))
    except:
        value=-1
    if not itemname in itemname2responses.keys():
        itemname2responses[itemname]=[(counter[id_name],value)]
    else:
        itemname2responses[itemname].append((counter[id_name],value))

   # print(itemname2responses)
    if ttdd%500==0:
        tout = open("sample_scoresRR.json",'w')
        json.dump(itemname2responses,tout)
        tout.close()
tout = open("sample_scoresRR.json",'w')
json.dump(itemname2responses,tout)
tout.close()
#    print(tt.replace("\n","-"))
#    print("==============================")

    
# 图中是一名女子在沙滩上和狗玩耍，旁边是一只拉布拉多犬，它们处于沙滩上。 

# 2nd dialogue turn
#response, history = model.chat(tokenizer, '框出图中击掌的位置', history=history)
#print(response)

