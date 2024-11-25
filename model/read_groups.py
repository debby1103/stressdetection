import json
a = open("./sample_scoresR.json",'r')
files = json.load(a)
regen = []
for name in files.keys():
    max,maxr,min,minr = None,None,None,None
  #  print(len(files[name]))
    if len(files[name])<2:
        continue
    for kk in files[name]:
        kk[1] = float(kk[1])
        if kk[1]>1:
            if max is None:
                max = kk[1]
                maxr = kk[0]
            else:
                if max<kk[1]:
                    max = kk[1]
                    maxr = kk[0]
        else:
            if min is None:
                min = kk[1]
                minr = kk[0]
            else:
                if min>kk[1]:
                    min = kk[1]
                    minr = kk[0]
    if max is None:
     #   print(files[name])
        continue
    print(max,min,files[name])
    if min is None:
        regen.append({"name":name,"output":maxr})
pout=open("regen.json",'w')
json.dump(regen,pout)
pout.close()
    

