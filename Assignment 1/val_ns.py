import re
import string
import csv
import random 


with open('val.csv','r') as f, open('stopwords.txt','r') as sw:
    f_content=f.read()
    processed = re.split('\n',f_content)
    processed = [x.lower() for x in processed]
    clean_text=[]
    sw_content=sw.read()
    sw_list=set(sw_content.split())
    for i in processed:
        tmp=re.split(' ',i)
        for n in tmp:
            tmp1=re.split(',',n)
            tmp_list=[]
            for m in tmp1:
                if m not in sw_list:
                    tmp_list.append(m)
        clean_text.append(tmp_list)   
with open('val_ns.csv','w')as q:
    writer=csv .writer(q)
    writer.writerows(clean_text)
