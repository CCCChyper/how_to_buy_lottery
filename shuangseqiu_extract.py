import re
import time
import csv
from datetime import datetime,timedelta
import datetime as dt
from selenium import webdriver
from selenium.webdriver.common.by import By

class extractor:
    def __init__(self):
        self.historical_data=None
        self.load_historical_data()

    def load_historical_data(self):
        with open('./shuangseqiu_data.csv', 'r', encoding='utf-8') as a:
            self.historical_data= list(filter(None, list(csv.reader(a))))
    def update_all(self,from_date=None):
        url = 'http://www.cwl.gov.cn/ygkj/wqkjgg/ssq/'
        from selenium.webdriver.chrome.options import Options
        chrome_options = Options()
        chrome_options.add_experimental_option("detach", True)
        chrome_options.headless = True  ###设置隐藏窗口，或chrome_options.add_argument('--headless')
        browser = webdriver.Chrome('./chromedriver.exe', options=chrome_options)
        browser.get(url)

        sequence_list = []
        if from_date == None:
            from_date = '2013-01-01'
        i = 0
        flag = 1
        while flag:
            time.sleep(3)
            i += 1
            print(f'now extract data from page {i}')
            p = re.compile(r'(.*)\(.*\)')
            datelist = [(re.match(p, i.text).group(1)) for i in browser.find_elements(By.XPATH,
                                                                                      value='/html/body/div[2]/div/div/div[2]/div/div[1]/div[2]/table/tbody/tr/td[2]')]

            firstlist = [(i.text) for i in browser.find_elements(By.XPATH,
                                                                 value='/html/body/div[2]/div/div/div[2]/div/div[1]/div[2]/table/tbody/tr/td[3]/div/div[1]')]
            secondlist = [(i.text) for i in browser.find_elements(By.XPATH,
                                                                  value='/html/body/div[2]/div/div/div[2]/div/div[1]/div[2]/table/tbody/tr/td[3]/div/div[2]')]
            thirdlist = [(i.text) for i in browser.find_elements(By.XPATH,
                                                                 value='/html/body/div[2]/div/div/div[2]/div/div[1]/div[2]/table/tbody/tr/td[3]/div/div[3]')]
            forthlist = [(i.text) for i in browser.find_elements(By.XPATH,
                                                                 value='/html/body/div[2]/div/div/div[2]/div/div[1]/div[2]/table/tbody/tr/td[3]/div/div[4]')]
            fithlist = [(i.text) for i in browser.find_elements(By.XPATH,
                                                                value='/html/body/div[2]/div/div/div[2]/div/div[1]/div[2]/table/tbody/tr/td[3]/div/div[5]')]
            sixthlist = [(i.text) for i in browser.find_elements(By.XPATH,
                                                                 value='/html/body/div[2]/div/div/div[2]/div/div[1]/div[2]/table/tbody/tr/td[3]/div/div[6]')]
            seventhlist = [(i.text) for i in browser.find_elements(By.XPATH,
                                                                   value='/html/body/div[2]/div/div/div[2]/div/div[1]/div[2]/table/tbody/tr/td[3]/div/div[7]')]

            if from_date not in datelist:
                pagelist = list(
                    zip(datelist, firstlist, secondlist, thirdlist, forthlist, fithlist, sixthlist, seventhlist))
            else:
                idx = datelist.index(from_date)
                pagelist = list(
                    zip(datelist, firstlist, secondlist, thirdlist, forthlist, fithlist, sixthlist, seventhlist))[:idx+1]
                flag = 'exit'
            sequence_list = sequence_list + pagelist
            next_page = browser.find_elements(By.XPATH, value='/html/body/div[2]/div/div/div[2]/div/div[2]/div/div/a')[
                -1]

            if flag == 'exit':
                break
            if next_page.get_attribute('class') == 'layui-box layui-laypage layui-laypage-molv':
                break
            next_page.click()
        self.write_csv(sequence_list)

    def write_csv(self,sequence_list):
        with open('./shuangseqiu_data.csv', 'a', encoding='utf-8', newline='') as a:
            sequence_list = sorted(sequence_list, key=lambda x: x[0], reverse=False)
            c = csv.writer(a)
            c.writerows(sequence_list)

    def update(self):
        current=datetime.now()
        date = current.date()
        weekday=date.isoweekday()
        lastdate = self.historical_data[-1][0]
        last_prize_day=self.check_last_prize_day(weekday,current)
        if str(last_prize_day)==lastdate:
            print('no need to update')
            pass
        if str(last_prize_day)!=lastdate:
            next_prize_day=self.check_next_prize_day(lastdate)
            self.update_all(str(next_prize_day))
            print('update complete')
            self.load_historical_data()
    def check_next_prize_day(self,last_date:str):
        next_prize_day=None
        d_list=list(map(int,last_date.split('-')))
        date=dt.date(d_list[0],d_list[1],d_list[2])
        weekday=date.isoweekday()
        if weekday in [2, 7]:
            next_prize_day =(date + timedelta(days=2))
        if weekday == 4:
            next_prize_day = (date +timedelta(days=3))
        return next_prize_day
    def check_last_prize_day(self, weekday,current):
        last_prize_day = None
        hour=current.hour
        date=current.date()
        if weekday in [2, 4, 7]:
            isprizeday = True
            if hour < 22:
                if weekday in [2, 4]:
                    last_prize_day = (current - timedelta(days=2)).date()
                if weekday == 7:
                    last_prize_day = (current - timedelta(days=3)).date()
            if hour >= 22:
                last_prize_day = date
        else:
            isprizeday = False
            if weekday in [1,3, 5]:
                last_prize_day = (current - timedelta(days=1)).date()
            if weekday == 6:
                last_prize_day = (current - timedelta(days=2)).date()
        return last_prize_day
        #self.update_all(from_date=lastdate)


if __name__=='__main__':
    ex=extractor()
    ex.update()
    print(ex.historical_data[-1])




