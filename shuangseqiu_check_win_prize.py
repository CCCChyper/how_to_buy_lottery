



class checker:
    def __init__(self,):
        pass
    def check(self,kaijiang_number:list,my_number:list):
        forecount = 0
        rearcount = 0
        kaijiang_fore=kaijiang_number[0:6]
        kaijiang_rear=kaijiang_number[6]
        for i in my_number[0:6]:
            if i in kaijiang_fore:
                forecount+=1
        if my_number[6]==kaijiang_rear:
            rearcount=1
        return self.prize(forecount,rearcount)
    def prize(self,fore,rear):
        if fore==6 and rear==1:
            return ('一等奖',6500000)
        if fore==6 and rear==0:
            return ('二等奖',150000)
        if fore==5 and rear==1:
            return ('三等奖',3000)
        if (fore==5 and rear==0) or (fore==4 and rear==1):
            return ('四等奖',200)
        if (fore==4 and rear==0) or (fore==3 and rear==0):
            return ('五等奖',10)
        if (fore==2 and rear==1) or (fore==1 and rear==1) or (fore==0 and rear==1):
            return ('六等奖',5)
        else:
            return ('Noneprize',0)