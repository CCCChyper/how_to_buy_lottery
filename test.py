import csv
import pprint
import random

from shuangseqiu_extract import extractor
from shuangseqiu_sequence_generateByDL import DLgenerator
import torch
from shuangseqiu_deep_learning import MultiLabelClassifier

if __name__=='__main__':
    ex=extractor()
    ex.update()
    sq = []
    dlg = DLgenerator(-1)

    for i in range(1000000):
        print(i)
        sq.append(dlg.tell_me_sequence())

    with open('./mynumber.csv','w',encoding='utf-8',newline='')as a:
        cw=csv.writer(a)
        cw.writerows(sq)