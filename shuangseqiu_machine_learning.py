import csv
import os.path

import joblib

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, \
    AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from shuangseqiu_extract import extractor



class data_transformer:  #预测核心号码

    def __init__(self):
        ex = extractor()
        self.rawdata=[[int(j) for j in i[1:8]] for i in ex.historical_data[::-1]]
        self.idx_list=range(0,len(self.rawdata))
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([range(1, 34)])
    def transform(self,idx):
        last_five=self.load_last_five(idx=idx)
        last_fifty=self.load_last_fifty(idx=idx)
        raw_y=self.rawdata[idx]
        #raw_y_core=self.find_core_y(raw_y,last_five)
        y = self.mlb.transform([raw_y])
        last_five_label = self.mlb.transform(last_five)
        x = np.zeros((1, 33))
        for lfl in last_five_label:
            x = x + lfl
        idxInfor=self.idxinformation(last_five,raw_y)
        spweight=self.span3_information(x)
        freweight=self.frequency_information(self.mlb,last_fifty)
        x=(x+idxInfor)*spweight*freweight
        return x[0],y[0]
    def pretransform(self, idx):
        #idx=-1   #因为在数据表里没有，必须设为-1
        last_five = self.load_last_five(idx=idx)
        last_fifty = self.load_last_fifty(idx=idx)
        if idx!=-1:
            now_y=self.rawdata[idx]
            last_y=self.rawdata[idx+1]

        if idx==-1:
            last_y=self.rawdata[0]
            now_y=last_y
        last_y_neighbor  = []
        for n in last_y:
            front = n - 1
            rear = n + 1
            if not front == 0:
                last_y_neighbor.append(front)
            if not rear == 34:
                last_y_neighbor.append(rear)
        last_y_neighbor=set(last_y_neighbor)

        y=self.mlb.transform([now_y])

        last_five_label = self.mlb.transform(last_five)
        x = np.zeros((1, 33))
        for lfl in last_five_label:
            x = x + lfl

        idxInfor = self.idxinformation(last_five, last_y_neighbor)
        spweight = self.span3_information(x)
        freweight = self.frequency_information(self.mlb, last_fifty)
        x = (x + idxInfor) * spweight * freweight
        return x[0], y[0]
    def load_last_fifty(self,idx):
        return self.rawdata[idx+1:idx+51]
    def load_last_five(self,idx):
        return self.rawdata[idx+1:idx+6]
    def frequency_information(self,mlb,last_fifty):
        last_fifty_label=mlb.transform(last_fifty)
        x = np.zeros((1, 33))
        for lfl in last_fifty_label:
            x = x + lfl
        minmax_scaler=MinMaxScaler()
        x=minmax_scaler.fit_transform(x.reshape(-1,1))
        return x.reshape(1,-1)+np.ones((1,33))
    def find_core_y(self,raw_y,last_five):
        core_y=[]
        candidate = []
        lastone = last_five[0][0:6]
        for n in lastone:
            front = n - 1
            rear = n + 1
            candidate.append(n)
            if not front == 0:
                candidate.append(front)
            if not rear == 34:
                candidate.append(rear)
        for l in last_five:
            candidate += l
        candidate = set(candidate)
        for num in raw_y:
            if num in candidate:
                core_y.append(num)
        return core_y

    def idxinformation(self,last_five,y_neighbor):
        y_d=dict()
        for i in range(1,34):
            y_d[str(i)]=0
        for i in y_neighbor:
            y_d=self.left_idxinformation(0,last_five,y_d,i)
            y_d=self.right_idxinformation(0,last_five,y_d,i)

        _,ii =zip(*y_d.items())
        ii=np.array(ii)
        return ii

    def left_idxinformation(self,j,last_five,y_d:dict,i):
        if (j<5) and (i-(j+1) in last_five[j]):
            y_d[str(i)]+=2
            j=j+1
            self.left_idxinformation(j,last_five,y_d,i)
        return y_d
    def right_idxinformation(self, j, last_five, y_d: dict, i):
        if (j < 5) and (i + (j + 1) in last_five[j]):
            y_d[str(i)] += 2
            j = j + 1
            self.right_idxinformation(j, last_five, y_d, i)
        return y_d
    def span3_information(self,x):
        axis=list(x[0])
        sp=np.zeros((1,33))
        for idx in range(0,33):
            i=idx
            j=idx+1-33
            k=idx+2-33
            sum=axis[i]+axis[j]+axis[k]
            sp[0,i]=sp[0,i]+sum
            sp[0,j]=sp[0,j]+sum
            sp[0,k]=sp[0,k]+sum
        return sp/30+np.ones((1,33))
    def write_dataset(self,x,y,xp,yp):
        with open(xp,'a',encoding='utf-8',newline='')as a:
            cw=csv.writer(a)
            cw.writerow(x)
        with open(yp,'a',encoding='utf-8',newline='')as b:
            cw=csv.writer(b)
            cw.writerow(y)


class trainer:
    def __init__(self):
        self.x = None
        self.y = None
    def load_x_y_data(self,xp,yp):
        self.x = self.load_x(xp)
        self.y = self.load_y(yp)
    def load_x(self,xp):
        with open(xp,'r',encoding='utf-8')as a:
            cr=[[float(j) for j in i] for i in list(csv.reader(a))]
            x=np.array(cr)
        return x
    def load_y(self,yp):
        with open(yp,'r',encoding='utf-8')as a:
            cr=[[float(j) for j in i] for i in list(csv.reader(a))]
            y=np.array(cr)
        return y
    def train(self,trainer_model,state):
        model=self.train_OneVsRestClassifier(self.x,self.y,trainer_model=trainer_model,state=state)
        print(model)
        joblib.dump(model,f'./{trainer_model}_model.m')
    def train_OneVsRestClassifier(self,x,y,trainer_model,state):
        if trainer_model=='GaussianNB':
            clf = OneVsRestClassifier(GaussianNB()).fit(x, y)
            return clf
        if trainer_model == 'RandomForestClassifier':
            clf = OneVsRestClassifier(RandomForestClassifier(random_state=state)).fit(x, y)
            clf.decision_function_shape = 'ovr'
            return clf
        if trainer_model == 'SVC':
            clf = OneVsRestClassifier(SVC(probability=True,random_state=state)).fit(x, y)
            return clf
        if trainer_model == 'LinearSVC':
            clf = OneVsRestClassifier(LinearSVC(random_state=state)).fit(x, y)
            return clf

        if trainer_model == 'LogisticRegression':
            clf = OneVsRestClassifier(LogisticRegression(random_state=state)).fit(x, y)

            return clf

        if trainer_model == 'DecisionTreeClassifier':
            clf = OneVsRestClassifier(DecisionTreeClassifier(random_state=state)).fit(x, y)
            clf.decision_function_shape = 'ovr'
            return clf
        if trainer_model == 'GradientBoostingClassifier':
            clf = OneVsRestClassifier(GradientBoostingClassifier(random_state=state)).fit(x, y)
            clf.decision_function_shape = 'ovr'
            return clf
        if trainer_model == 'ExtraTreesClassifier':
            clf = OneVsRestClassifier(ExtraTreesClassifier(random_state=state)).fit(x, y)
            return clf

        if trainer_model == 'AdaBoostClassifier':
            clf = OneVsRestClassifier(AdaBoostClassifier(random_state=state)).fit(x, y)
            clf.decision_function_shape = 'ovr'
            return clf

        if trainer_model == 'BaggingClassifier':
            clf = OneVsRestClassifier(BaggingClassifier(random_state=state)).fit(x, y)
            return clf

class predictor:
    def __init__(self,model_path):
        self.model_path=model_path
        self.model=self.load_model(self.model_path)
    def load_model(self,model_path):
        m=joblib.load(model_path)
        return m
    def pred(self,x):
        return self.model.predict(x)


def transformdata_and_write(tf,span:tuple,xp,yp):
    if os.path.exists(xp):
        os.remove(xp)
    if os.path.exists(yp):
        os.remove(yp)
    start,end=span
    for d,idx in enumerate(tf.idx_list[start:end]):
        x,y=tf.pretransform(idx)
        tf.write_dataset(x,y,xp,yp)

def get_score(estimators):
    # 获取每个标签的预测得分
    for i, estimator in enumerate(estimators):
        scores = estimator.predict_proba(XX.reshape(1, 33))
        print("Label {}: {}".format(i, scores))
    for i, estimator in enumerate(estimators):
        scores = estimator.decision_function(XX.reshape(1, 33))
        print("Label {}: {}".format(i, scores))


if __name__=='__main__':
    train_x_path = './train_x-new.csv'
    train_y_path = './train_y-new.csv'

    tf=data_transformer()
    transformdata_and_write(tf,tuple((7,507)),train_x_path,train_y_path)
    #[GaussianNB,DecisionTreeClassifier,RandomForestClassifier，GradientBoostingClassifier，ExtraTreesClassifier，AdaBoostClassifier,BaggingClassifier]
    model_name='DecisionTreeClassifier'
    tr=trainer()
    tr.load_x_y_data(train_x_path,train_y_path)
    tr.train(trainer_model=model_name,state=1234)

    myidx=5
    pr=predictor(model_path=f'./{model_name}_model.m')
    XX,_=tf.pretransform(idx=myidx)
    y=tf.mlb.transform([tf.rawdata[myidx]]).tolist()[0]
    #print(XX)
    print(y)
    result=pr.pred(x=XX.reshape(1,33)).tolist()[0]
    print(result)


    totalcount=0
    c1 = 0
    countdict = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0,'5':0,'6':0,'7':0}
    for mi in range(1100,1400):
        XX, _ = tf.pretransform(idx=mi)
        y = tf.mlb.transform([tf.rawdata[mi]]).tolist()[0]
        print(y)
        result = pr.pred(x=XX.reshape(1, 33)).tolist()[0]
        print(result)
        count = 0
        for i,j in zip(y,result):
            if i==1 and j==1:
                count+=1
            if j==1:
                c1+=1
        totalcount+=count
        print(f'----------------------答对{count}个')
        countdict[str(count)] += 1
    print(f'共计答对{totalcount}个')
    print(f'一共有{c1}个1')
    print(c1 / totalcount)
    print(countdict)


    # 获取每个标签的分类器
    estimators = pr.model.estimators_
    #get_score(estimators)




