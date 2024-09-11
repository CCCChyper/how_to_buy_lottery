import random

import numpy as np
import torch
from sklearn.preprocessing import MultiLabelBinarizer

from shuangseqiu_check_win_prize import checker
from shuangseqiu_extract import extractor
from shuangseqiu_deep_learning import MultiLabelClassifier

class DLgenerator():
    def __init__(self,n):
        self.model=torch.load('./dl_model.pth')
        random.seed(random.randint(1, 100000))
        self.historical_data=[[int(j) for j in i[1:]] for i in extractor().historical_data[::-1]]
        self.n=None
        self.predict_sequence=None
        self.kaijiang_number=None

        self.last_number_appear = None
        self.last_five_adjacent=None
        self.last_three_number_appear=None
        self.reset_n(n)

    def reset_n(self,n):
        self.n=n

        def find_last_number_appear():
            last_number = self.historical_data[self.n + 1][0:6]
            appear=[]
            for i in last_number:
                if i - 1 >= 1:
                   appear.append(i - 1)
                if i + 1 <= 33:
                    appear.append(i + 1)
                appear.append(i)
            return list(set(appear))
        def find_last_three_number_appear():
            last_three_sequence = [i[0:6] for i in self.historical_data[self.n + 1:self.n + 1 + 3]]
            appear=[]
            for l in last_three_sequence:
                for i in l:
                    if i - 1 >= 1:
                        appear.append(i - 1)
                    if i + 1 <= 33:
                        appear.append(i + 1)
                    appear.append(i)
            last_number = self.historical_data[self.n + 1][0:6]
            for i in last_number:
                if i-1>=1:
                    appear.append(i-1)
                if i+1<=33:
                    appear.append(i+1)
                appear.append(i)
            return list(set(appear))

        self.last_number_appear=find_last_number_appear()
        self.last_three_number_appear=find_last_three_number_appear()
        self.last_five_adjacent = self.find_last_five_adjacent()
        if self.n==-1:
            self.predict_sequence=self.model_predict(-1)
            self.kaijiang_number=None
        else:
            self.predict_sequence=self.model_predict(self.n)
            self.kaijiang_number=self.historical_data[self.n]
    def tell_me_sequence(self):
        qualified=False
        sq=None
        while(qualified==False):
            sq = self.generate_random_sequence()
            qualified=self.check_qualified(sq)
        return sq
    def generate_random_sequence(self):
        flag=True
        min=17
        sq=None
        while(min>16):
            sq = random.sample(set(range(1,34)), 7)
            sq=sorted(sq)
            min=sq[0]
        while (flag):
            for i in sq:
                if i<=16:
                    if random.randint(1,2)==1:
                        rear=sq.pop(sq.index(i))
                        sq=sq+[rear]
                        return sq

    def model_predict(self,target_idx):

        mlb = MultiLabelBinarizer()
        mlb.fit([range(1, 34)])
        one_hot_rawdata = mlb.transform(self.historical_data)
        data = np.array(one_hot_rawdata, dtype=np.float32)

        seq_length=10
        threshhold=0.25
        n = target_idx
        tmp = data[n+1]
        for j in range(1, seq_length):
            tmp = tmp + data[n +1 + j]
        input = torch.from_numpy(np.array(tmp))

        with torch.no_grad():
            predicted_labels = (self.model(input) > threshhold).float()

        result = []
        for i in range(0,len(predicted_labels)):
            if predicted_labels[i]==1:
                result.append(i+1)

        return result

    def check_qualified(self,sq):
        def two_three_in_predict(sq):
            in_count=0
            notin_count=0
            for i in sq:
                if i in self.predict_sequence:
                    in_count+=1
                if i not in self.predict_sequence:
                    notin_count+=1
            if in_count>=1 or notin_count>=len(self.predict_sequence)-1:
                return True
            else:
                return False
        def two_adjacent(front):
            last_five_adjacent=self.last_five_adjacent
            count=0
            pair=None
            for i in range(len(front)-1):
                if front[i]+1==front[i+1]:
                    count+=1
                    pair=(front[i],front[i+1])
            if count>2 or count==0:
                return False
            if pair in last_five_adjacent:
                return False
            else:
                return True

        def three_four_repeat(front):
            appear=self.last_number_appear
            count=0
            for i in front:
                if i in appear:
                    count+=1
            if count>=2 and count<=5:
                return True
            else:
                return False
        def atLeast_one_not_appear_in_last_three(front):
            appear=self.last_three_number_appear
            count = 0
            for i in front:
                if i not in appear:
                    count += 1
            if count >= 1 and count <= 3:
                return True
            else:
                return False

        #if two_three_in_predict(sq)!=True:
         #   return False
        rear = sq[-1]
        front = sq[0:6]
        if two_adjacent(front)!=True:
            return False
        if three_four_repeat(front)!=True:
            return False
        if atLeast_one_not_appear_in_last_three(front)!=True:
            return False
        else:
            return True

    def find_last_five_adjacent(self):
        those_adjacent=[]
        for i in range(self.n+1,self.n+1+10):
            tmp=self.historical_data[i][0:6]
            pair = None
            for i in range(len(tmp) - 1):
                if tmp[i] + 1 == tmp[i + 1]:
                    pair = (tmp[i], tmp[i + 1])
                    those_adjacent.append(pair)
        return those_adjacent

if __name__=='__main__':
    dlg=DLgenerator(0)
    sq=dlg.tell_me_sequence()
    print(sq)
    print(dlg.predict_sequence)
    print(dlg.kaijiang_number)

    total = 0
    prize_count = dict()
    prize_list = ['一等奖', '二等奖', '三等奖', '四等奖', '五等奖', '六等奖']
    for p in prize_list:
        prize_count[p] = 0


    c=0
    trys=500000
    ck = checker()
    cost=0
    for i in range(c,c+1):
        dlg.reset_n(i)
        kaijiang_number = dlg.kaijiang_number
        predict_sequence=dlg.predict_sequence
        for i in range(trys):
            sq=dlg.tell_me_sequence()
            award, win = ck.check(kaijiang_number, sq)
            if win != 0:
                #print(f'我的号码是：{sq}，获得{_}')
                if award=='一等奖' or award=='二等奖':
                    print(f'{sq}-------{award}')
                prize_count[award] += 1
            total += win
            cost+=2
        print(f'获奖：{total}元，成本：{cost}元，收益：{total - cost}元')
        print(prize_count)
        print(f'开奖号码是：{kaijiang_number}')
        print(f'预测号码是：{predict_sequence}')
