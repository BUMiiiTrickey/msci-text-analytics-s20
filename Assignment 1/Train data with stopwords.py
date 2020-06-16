import re
import string
import csv
import random

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

random.shuffle(reviews)
train_data=reviews[:int((len(reviews))*.80)]
test_data=reviews[int(len(reviews)*.80):int((len(reviews))*.90)]
val_data=reviews[int(len(reviews)*.90):]
print(train_data)

with open('trainwsw.csv','w')as q:
writer=csv .writer(q)
writer.writerows(train_data)
