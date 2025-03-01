hints = [
"eyebrow: inner portions of the eyebrows raising",
"eyebrow: outer parts of the eyebrows raising",
"eyebrow: eyebrows lowering",
"lid: upper lid raising",
"cheek: cheek raising",
"nose: nose wrinkling",
"lip: the corners of the lips pulled outward",
"mouth: the corners of the mouth pulled down",
"chin: chin raising",
"lip: lip stretching",
"lip: lips parted",
"jaw: jaw dropping"
]


import glob
import json
AUS=["1","2","4","5","6","9","12","15","17","20","25","26"]
extent = [" very slightly"," slightly",""," obviously"," very obviously"]
a = glob.glob("./DISFA/Labels/SN*")
output_file = open("./train_expression.json",'w')
new_lines = []
for item in a[:-1]:
    b = glob.glob(item+"/"+"*")
    for jtem in b:
        track = jtem.split("/")[-1]
        c = glob.glob("./DISFA/Images/"+item.split("/")[-1]+"/"+track+"/"+"*.jpg")
        c = sorted(c)

        aus = []
        for _ in range(len(c)):
            aus.append([])
        for kk,ktem in enumerate(AUS):
            filei = open(item+"/"+track+"/"+"AU"+ktem+".txt",'r')
            idx = 0
            for line_ in filei.readlines():
                if "jpg" in line_:
                    idx+=1
                    ext = int(line_.split("     ")[-1])
                    if ext>1:
                        aus[idx-1].append(hints[kk]+extent[ext-1])
            assert idx==len(c)
        for xx,yy in zip(c,aus):
            template = {"eyebrow":[],"lid":[],"cheek":[],"nose":[],"lip":[],"mouth":[],"chin":[],"jaw":[]}
            for pp in yy:
                item1 = pp.split(": ")[0]
                item2 = pp.split(": ")[1]
                template[item1].append(item2)
            tmpp = ""
            for ke in template.keys():
                if template[ke]==[]:
                    continue
                else:
                    tmpp+=ke+": "+"; ".join(template[ke])
                    tmpp+="\n"
            if (len(tmpp)>0 and tmpp[-1]=="\n"):
                tmpp = tmpp[:-1]
                tmpp+="."
            if len(tmpp)==0:
                tmpp = "No facial actions observed."
            new_line = {"img":xx,"exp":tmpp}
            new_lines.append(new_line)
json.dump(new_lines,output_file)




