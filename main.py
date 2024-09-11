# This is a sample Python script.
import re
from functools import partial

from PyQt5.QtCore import Qt, QRegExp
from PyQt5.QtGui import QFont, QColor, QRegExpValidator, QIntValidator, QTextCursor
from PyQt5.QtWidgets import QMainWindow, QApplication, QTableWidgetItem, qApp, QMenu, QAction, QAbstractItemView, \
    QMessageBox, QWidget

from shuangseqiu_machine_learning import data_transformer
from shuangseqiu_mainwindow import Ui_MainWindow
from shuangseqiu_sequence_generateByML import MLgenerator


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


class mainwindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui=Ui_MainWindow()
        self.ui.setupUi(self)

        #4个按钮设置
        self.ui.jixuan.clicked.connect(self.jixuan)
        self.ui.sure.clicked.connect(self.sure)
        self.ui.clear_text.clicked.connect(self.clear_text)
        self.ui.deleteitem.clicked.connect(self.delete_item)

        #7个display窗格设置
        self.Numdisplay_matric=[self.ui.a,self.ui.b,self.ui.c,self.ui.d,self.ui.e,self.ui.f,self.ui.g]
        self.set_display_block(self.Numdisplay_matric)
        self.set_display_block_textconstraint(self.Numdisplay_matric)

        #展示我选的号码
        self.ui.mynumberchosen.itemClicked.connect(self.on_item_clicked)
        self.ui.mynumberchosen.setSelectionMode(QAbstractItemView.ExtendedSelection)

        #限制文本输入
        self.validator = QIntValidator(0,99)

        # 添加右键菜单
        self.ui.copyAction.triggered.connect(self.copyItem)
        self.ui.contextMenu.addAction(self.ui.copyAction)
        self.ui.mynumberchosen.setContextMenuPolicy(Qt.CustomContextMenu)
        self.ui.mynumberchosen.customContextMenuRequested.connect(self.showContextMenu)

        #加载号码序列生成器
        self.mlge=MLgenerator(idx=-1)
        #self.mlge.ex.update()
        self.sequence_list=[]
        self.sequence_tobesure=None

        #显示信息
        self.historical_data = self.mlge.ex.historical_data[-1:-101:-1]
        self.historical_data_show()
        self.analysis_show()
    def set_display_block_textconstraint(self,block_list):
        for block in block_list:
            block.textChanged.connect(partial(self.on_text_changed,block))
    def on_text_changed(self,block):
        text = block.toPlainText()
        if text=='':
            pass
        else:
            if self.validator.validate(text, 0)[0] != 2 or len(text)>2:
                # 如果文本长度大于2或者字符不是数字，则过滤文本
                ttt = re.sub(r'[^0-9]*', '', text)
                # block.setAlignment(Qt.AlignCenter)
                block.setText(ttt[0:2])
                block.setAlignment(Qt.AlignCenter)
                cursor = block.textCursor()
                cursor.movePosition(QTextCursor.End)
                block.setTextCursor(cursor)

    def set_display_block(self,block_list):
        font = QFont('Arial', 56, QFont.Bold)
        for block in block_list:
            block.setMinimumHeight(125)
            block.setFont(font)
            block.setAlignment(Qt.AlignCenter)

    def analysis_show(self):
        last_model_pred=self.mlge.last_model_pred
        model_pred=self.mlge.model_pred
        mainfore=self.mlge.mainfore_candidate
        all_last_five=self.mlge.all_lastfive_num
        lastfore=self.mlge.lastfore_candidate
        sss=f'------上一期模型预测号码\n{last_model_pred}\n------模型预测号码\n{model_pred}\n------核心号码\n{mainfore}\n------前五期出现号码\n{all_last_five}\n------前五期未出现号码\n{lastfore}'

        self.ui.analyse_blank.setText(sss)
    def historical_data_show(self):
        font = QFont('Arial', 12, QFont.Bold)
        fontcolor = QColor('red')
        backcolor=QColor('blue')
        self.ui.historicdata.setHorizontalHeaderLabels(['日期']+list(map(str,range(1,34)))+['后区'])
        self.ui.historicdata.setColumnWidth(0,100)
        self.ui.historicdata.setColumnWidth(34,40)
        for i in range(0,100):
            date=self.historical_data[-(i+1)][0]
            datafore = self.historical_data[-(i+1)][1:7]
            rear=self.historical_data[-(i+1)][7]
            for j in range(0,35):

                if j==0:
                    item=QTableWidgetItem(date)
                if j==34:
                    item=QTableWidgetItem(str(rear))
                if j in range(1,34):
                    item=QTableWidgetItem(str(j))
                item.setTextAlignment(Qt.AlignCenter)
                if j in range(1,34):
                    if str(j) in datafore:
                        item.setFont(font)
                        item.setForeground(fontcolor)
                        item.setBackground(backcolor)
                    if str(j)==rear:
                        item.setFont(font)
                        item.setForeground(fontcolor)
                        item.setBackground(QColor('yellow'))
                self.ui.historicdata.setItem(i, j, item)
        self.ui.historicdata.scrollToBottom()

        #self.ui.historiccaData.setText(str(self.historical_data))
    def jixuan(self):
        sequence=self.mlge.tell_me_sequence(howmany=1,mode=None)[0]
        font = QFont('Arial', 56, QFont.Bold)
        for i,block in enumerate(self.Numdisplay_matric):
            block.setText(str(sequence[i]))
            block.setFont(font)
            block.setAlignment(Qt.AlignCenter)
    def sure(self):
        self.sequence_tobesure=[]
        for i,block in enumerate(self.Numdisplay_matric):
            s=self.check_isNum_qualified(block.toPlainText(),i)
            if s!=False:
                self.sequence_tobesure.append(s)
            else:
                QMessageBox.information(QWidget(),'提示', '号码不符合规定，请检查后重新输入')
                self.sequence_tobesure=None
                break

        if self.sequence_tobesure!=None:
            if self.check_sequence_qualified(self.sequence_tobesure)==True:
                self.sequence_tobesure = sorted(self.sequence_tobesure[0:6], reverse=False) + [self.sequence_tobesure[6]]
                if self.sequence_tobesure not in self.sequence_list:
                    self.sequence_list.append(self.sequence_tobesure)
                    self.ui.mynumberchosen.addItem(str(self.sequence_tobesure))
            else:
                self.sequence_tobesure=None
                QMessageBox.information(QWidget(), '提示', '号码不符合规定，请检查后重新输入')
        else:
            pass
    def clear_text(self):
        for i,block in enumerate(self.Numdisplay_matric):
            block.clear()
            block.setAlignment(Qt.AlignCenter)
    def check_sequence_qualified(self,se):
        tocheck=se[0:6]
        if len(set(tocheck))==6:
            return True
        else:
            return False
    def check_isNum_qualified(self,s,idx):
        try:
            i=int(s)
            if i in range(1,34) and idx in range(0,6):
                return i
            elif i in range(1,17) and idx==6:
                return i
            else:
                return False
        except:
            return False

    def on_item_clicked(self,item):
        print(item.text())
    def delete_item(self):
        selected_items = self.ui.mynumberchosen.selectedItems()
        if selected_items:
            for si in selected_items:
                self.ui.mynumberchosen.takeItem(self.ui.mynumberchosen.row(si))
                numbers = re.findall(r'\d+', si.text())
                wait_to_delete = list(map(int, numbers))
                self.sequence_list.remove(wait_to_delete)
                del si

    def showContextMenu(self, pos):
        self.ui.contextMenu.exec_(self.ui.mynumberchosen.mapToGlobal(pos))

    def copyItem(self):
        select_Item = self.ui.mynumberchosen.selectedItems()
        if select_Item is not None:
            text = '\n'.join([item.text() for item in select_Item])
            qApp.clipboard().setText(text)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app=QApplication([])
    mw=mainwindow()
    mw.show()
    app.exec_()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
