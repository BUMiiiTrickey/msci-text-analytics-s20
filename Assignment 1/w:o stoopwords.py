import re
import string
import csv

pos = neg = ""
txt=open("/Users/iris/Desktop/pos.txt", "r")
pos=txt.read()
txt1=open("/Users/iris/Desktop/neg.txt", "r")
neg=txt1.read()
pos += "\n"
pos += neg 
with open ('file3.txt', 'w') as fp: 
    fp.write(pos) 
    
review=open('file3.txt','r')
no_punctuation_documents = []
import string
for i in review:
  no_punctuation_documents.append(i.translate(str.maketrans('','', '!"#$%&()*+/:;<=>@[\\]^`{|}~\t\n')))

reviews = []
for i in no_punctuation_documents:
  reviews.append(i.split(' '))

with open('assignment1.txt','w')as s:
    writer=csv.writer(s,delimiter=' ')
    writer.writerows(reviews)



with open('/Users/iris/Desktop/train.csv','w') as f, open('/Users/iris/Desktop/stopwords.txt','r') as sw:
    f_content=f.read()
    processed = re.split('\n',f_content)
    processed = [x.lower() for x in processed]
    clean_text=[]
    sw_content=sw.read()
    sw_list=set(sw_content.split())
    for i in processed:
        tmp=re.split(' ',i)
        tmp_list=[]
        for m in tmp:
            if m not in sw_list:
                tmp_list.append(m)
        clean_text.append(tmp_list)

with open('out.csv','w')as q:
    writer=csv .writer(q)
    writer.writerows(clean_text)
