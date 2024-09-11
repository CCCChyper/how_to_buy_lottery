import random


from shuangseqiu_check_win_prize import checker

from shuangseqiu_extract import extractor
from shuangseqiu_machine_learning import data_transformer, predictor


class MLgenerator:
    def __init__(self,idx):
        random.seed(random.randint(1,100000))
        self.ex=extractor()
        self.ex.update()
        self.totestidx=idx                  #if-1，表示进行全新的预测
        self.sequence=None
        self.last_five=self.load_last_five()
        self.mainNUM=None
        self.secondaryNUM=None
        self.lastNUM=None
        self.rear=None
        self.fix_mainNUM='fix_mainNUM'
        self.last_y_neighbor=self.find_last_y_neighbor(last_y=self.last_five[0])
        self.discard_num = [0, 0, 0, 0, 0, 0, 0]
        self.discard_num =[random.randint(1,5),random.randint(5,11),random.randint(11,16),random.randint(16,21),random.randint(21,27),random.randint(27,33)]+[random.randint(1,16)]

        self.all_lastfive_num=self.count_lastfive_num(self.last_five)
        self.mainfore_candidate,self.model_pred,self.last_model_pred=self.model_pred_mainfore_candidate()
        self.lastfore_candidate=set(range(1,34))-self.mainfore_candidate-self.all_lastfive_num


    def load_last_five(self):
        #ex.update()
        tti=self.totestidx
        return [[int(j) for j in i[1:]] for i in self.ex.historical_data[-(tti+2):-(tti+7):-1]]
    def tell_me_sequence(self,howmany,mode=None):
        se_list=[]
        if mode=='fix_mainNUM':
            self.mainNUM=self.tell_me_mainfore()
            while howmany!=0:
                self.secondaryNUM=self.tell_me_secondaryfore()
                self.lastNUM=self.tell_me_lastfore()
                self.rear=self.tell_me_rear()
                self.sequence=sorted(self.mainNUM+self.secondaryNUM+self.lastNUM,reverse=False)+self.rear
                if self.check_qualified(self.sequence)==True:
                    howmany=howmany-1
                    se_list.append(self.sequence)
        else:
            while howmany != 0:
                self.mainNUM = self.tell_me_mainfore()
                self.secondaryNUM = self.tell_me_secondaryfore()
                self.lastNUM = self.tell_me_lastfore()
                self.rear = self.tell_me_rear()
                self.sequence=sorted(self.mainNUM+self.secondaryNUM+self.lastNUM,reverse=False)+self.rear
                if self.check_qualified(self.sequence) == True:
                    howmany = howmany - 1
                    se_list.append(self.sequence)
        return se_list
    def check_qualified(self,sequence):
        a=self.check_has_twotogether(sequence)
        b=self.check_has_in_y_neighbor(sequence)
        c=self.check_num_in_span(sequence)
        f_list=[a,b,c]
        true_list=[f for f in f_list if f==True]
        qualified=None
        if len(true_list)==len(f_list):
            qualified=True
        else:
            qualified=False
        return qualified
    def check_num_in_span(self,sequence):
        first=sequence[0]
        end=sequence[-1]
        flag1=False
        flag2=False
        if random.randint(1000,1000000)%20!=0:
            if first in range(1,11):
                flag1=True
        else:
            flag1=True
        if random.randint(1000,1000000)%10!=0:
            if end in range(25,34):
                flag2=True
        else:
            flag2=True
        if flag1==True and flag2==True:
            return True
        else:
            return False

    def check_has_twotogether(self,sequence):
        has_twotogether=False
        if random.sample([0,0,0,1],1)[0]!=1:
            wait_to_check=sequence[0:6]
            #print(wait_to_check)
            for w in wait_to_check:
                if w+1 in wait_to_check:
                    has_twotogether=True
                    break
        else:
            has_twotogether=False
        return has_twotogether
    def check_has_in_y_neighbor(self,sequence):
        has_in_neighbor = False
        neighbor_count = 0
        wait_to_check = sequence[0:6]
        for w in wait_to_check:
            if w in self.last_y_neighbor:
                neighbor_count+=1
        if random.sample([4,3,3,2, 2, 2, 1,1,1,1], 1)[0] ==neighbor_count:
            has_in_neighbor=True
        else:
            has_in_neighbor = False
        return has_in_neighbor
    def find_last_y_neighbor(self,last_y):
        last_y_neighbor = []
        for n in last_y:
            front = n - 1
            rear = n + 1
            if not front == 0:
                last_y_neighbor.append(front)
            if not rear == 34:
                last_y_neighbor.append(rear)
        last_y_neighbor = set(last_y_neighbor)
        return last_y_neighbor
    def count_lastfive_num(self,last_five):
        candidate = []
        lf = [l[0:6] for l in last_five]
        for l in lf:
            candidate += l
        return set(candidate)
    def model_pred_mainfore_candidate(self):
        model_name = 'DecisionTreeClassifier'
        tf = data_transformer()
        XX, _ = tf.pretransform(idx=self.totestidx)       ##做预测时没办法统计位置信息，要换一种转换方法,idx应为-1
        XX2,_2=tf.pretransform(idx=self.totestidx+1)      ##输出上一期的预测，看准不准,此时idx为0

        mlb=0
        if mlb==0:
            pr = predictor(f'./{model_name}_model_mlb.m')
            result=[int(k) for k in pr.pred(x=XX.reshape(1,33)).toarray()[0].tolist()]
            ShangYiQi_result=[int(k) for k in pr.pred(x=XX2.reshape(1,33)).toarray()[0].tolist()]
        else:
            pr = predictor(f'./{model_name}_model.m')
            result = pr.pred(x=XX.reshape(1, 33)).tolist()[0]
            ShangYiQi_result = [int(k) for k in pr.pred(x=XX2.reshape(1, 33)).toarray()[0].tolist()]

        model_pred = []
        for i in range(0, 33):
            if result[i] == 1:
                model_pred.append(i + 1)
        last_model_pred=[]
        for i in range(0, 33):
            if ShangYiQi_result[i] == 1:
                last_model_pred.append(i + 1)

        mainfore_candidate = set(model_pred)&set(self.all_lastfive_num|self.last_y_neighbor) - set(self.discard_num[0:6])
        return mainfore_candidate,set(model_pred),set(last_model_pred)
    def tell_me_mainfore(self):
        howmany=random.sample([1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3],1)[0]
        if howmany==0:
            return []
        else:
            if howmany>=len(self.mainfore_candidate):
                return list(self.mainfore_candidate)
            elif howmany<len(self.mainfore_candidate):
                return []
                return random.sample(set(self.mainfore_candidate),howmany)
    def tell_me_secondaryfore(self):
        #howmany=random.randint(4,5)-len(self.mainNUM)
        howmany=random.sample([4,4,4,4,5,5,5,5,5,5,6,6,6,6,6,6,6,6],1)[0]-len(self.mainNUM)

        if howmany!=0:
            candidate=set(self.all_lastfive_num)|self.last_y_neighbor-set(self.mainNUM) - set(self.discard_num[0:6])
            return random.sample(set(candidate),howmany)
        else:
            return []
    def tell_me_lastfore(self):
        howmany=6-len(self.mainNUM)-len(self.secondaryNUM)
        if howmany!=0:
            candidate = set(range(1, 34)) - set(self.mainNUM + self.secondaryNUM)-set(self.all_lastfive_num) - set(self.discard_num[0:6])
            critical_num=candidate&set(self.last_y_neighbor)
            candidate=list(candidate)
            self.lastfore_candidate=candidate
            if len(critical_num)!=0:      ##增加last_y_neighbor的数字的权重
                for c in critical_num:
                    candidate=candidate+[c]+[c]+[c]+[c]+[c]+[c]+[c]

            #candidate=set(range(1,34))-set(self.mainNUM+self.secondaryNUM)-set(self.all_lastfive_num)
            #print(set(candidate))
            f=True
            lastfore=None
            while f==True:
                lastfore=random.sample(candidate,howmany)
                if len(set(lastfore))==howmany:
                    f=False
            return lastfore
        else:
            return []
    def tell_me_rear(self):
        lf_r=[r[-1] for r in self.last_five]
        l_r_extend=[k for k in [lf_r[0],lf_r[0]+1,lf_r[0]-1] if k in range(1,17)]
        #condition=lf_r+l_r_extend
        flag=True
        rear=0
        while flag:
            r_candidate=random.sample(range(1,17),2)
            #if abs(r_candidate[0]-r_candidate[1])<=8:
            if len(set(r_candidate)&set(l_r_extend))!=0:
                rear=random.sample(r_candidate,1)
                if rear!=self.discard_num[-1]:
                    flag=False
        #return [random.randint(1,16)]
        return rear

if __name__=='__main__':
    iii=-1
    mlge=MLgenerator(idx=iii)
    llll=[]
    m=mlge.fix_mainNUM
    trys=1000000

    while len(llll)<trys:
        lll=mlge.tell_me_sequence(howmany=2,mode=m)
        #print(f'----------------------------lastNUM:{ge.lastNUM}')
        llll+=lll

    ck=checker()
    ex=extractor()
    ex.update()
    kaijiang_number=[int(j) for j in ex.historical_data[-(iii+1)][1:]]
    #kaijiang_number=[2,3,10,24,28,30,8]

    total=0
    prize_count=dict()
    prize_list=['一等奖','二等奖','三等奖','四等奖','五等奖','六等奖']
    for p in prize_list:
        prize_count[p]=0
    for l in llll:
        _,win=ck.check(kaijiang_number,l)
        if win!=0:
            print(f'我的号码是：{l}，获得{_}')
            prize_count[_]+=1
        total+=win
    print(f'获奖：{total}元，成本：{2*trys}元，收益：{total-2*trys}元')
    print(prize_count)
    print(f'开奖号码是：{kaijiang_number}')
    print(f'-----------------model_predict:{mlge.model_pred}')
    print(f'-----------------mainfore_candidate:{mlge.mainfore_candidate}')
    print(f'-----------------all_lastfive_num:{mlge.all_lastfive_num}')
    print(f'-----------------last_fore_candidate:{mlge.lastfore_candidate}')
    print(f'-----------------num_tobe_discarded:{mlge.discard_num}')
