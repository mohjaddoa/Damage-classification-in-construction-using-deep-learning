from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
from os import path
from PyQt5.uic import loadUiType
import os
import shutil
import time
from PyQt5 import QtGui
from cv2 import imread
from tensorflow.keras.models import load_model
import numpy as np
import cv2

uisystem,_ = loadUiType(path.join(path.dirname(__file__),"test_ui.ui"))


class Window(QMainWindow,uisystem):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.path_file=" "
        self.handling_buttons()
        self.results_txt=[]
        self.num_files=0
        self.label_numfiles.setText(str(self.num_files))
        self.remove_img_files()
        self.btn_multi_activation()
        self.currentIndex=0
        self.onstart()

    def onstart(self):
        self.loading.setVisible(False)
        self.progressBar.setVisible(False)
        self.next.setVisible(False)
        self.back.setVisible(False)
        self.groupBox.setVisible(False)
        self.gallery.setVisible(False)
        self.setWindowIcon(QtGui.QIcon("support.png"))
        self.display_class_files.setEnabled(False)

    def handling_buttons(self):
        self.selectfiles.clicked.connect(self.open_files)
        self.run_multi_class.clicked.connect(self.mutifiles_classifcation)
        self.next.clicked.connect(self.next_item)
        self.back.clicked.connect(self.back_item)
        self.display_class_files.clicked.connect(self.display_results)

    def btn_multi_activation(self):
        if self.textEdit.text() != '':
            self.run_multi_class.setEnabled(True)
        else:
            self.run_multi_class.setEnabled(False)

    def open_files(self):
        self.remove_img_files()
        self.set_files=[]
        file = QFileDialog().getOpenFileNames(self,'Open File','D:\\project\\OneDrive_1_1-14-2022\\Classification\\Dataset3\\val','JPG FILE(*.jpg);;JPEG FILE (*.jpeg);;PNG FILE (*.png)')
        if len(file[0][0:]) == 0:
            QMessageBox.critical(self, "Error", "You need to select image file!")
        else:
            self.run_multi_class.setEnabled(True)
            self.directory = os.path.dirname(file[0][0])
            self.num_files=len(file[0][0:])
            self.label_numfiles.setText(str(self.num_files))
            self.textEdit.setText(self.directory)
            for item in range(0,len(file[0][0:])):
                self.set_files.append(file[0][item])
            self.copy_file()
                
    def mutifiles_classifcation(self):
        self.loading.setVisible(True)
        self.progressBar.setVisible(True)
        self.animate_progressBar()
        while True:
            self.results_files= os.listdir('results')
            if len(self.set_files) == len(self.results_files):
                self.show_gallery(0)
                self.display_class_files.setEnabled(True)
                break
        
    def remove_img_files(self):
        path= os.listdir('input')
        for file in path:
            os.remove(os.path.join('input',file))
        path= os.listdir('results')
        for file in path:
            os.remove(os.path.join('results',file))    
        

    def copy_file(self):
        for file in self.set_files:
            file_name = os.path.basename(file)
            input_copy = 'input/'+str(file_name)
            shutil.copyfile(file, input_copy)

    def next_item(self):
        if self.currentIndex  < len(self.set_files)-1:
            self.next.setEnabled(True)
            self.back.setEnabled(True)

            self.show_gallery(self.currentIndex + 1)
        else:
            self.currentIndex = len(self.set_files)-1
            self.show_gallery(self.currentIndex)
            self.next.setEnabled(False)
            self.back.setEnabled(True)

      
    def back_item(self):
        if self.currentIndex < 0:
            self.currentIndex=0
            self.back.setEnabled(False)
            self.next.setEnabled(True)
        else:
            self.back.setEnabled(True)
            self.next.setEnabled(True)
            self.show_gallery(self.currentIndex - 1)

    def animate_progressBar(self):
        for i in range(101):
            time.sleep(0.05)
            if i == 1:
                self.loading.setText('loading Model...')
            if i == 50:
                self.loading.setText('Data Processing...')
            if i == 95:
                self.loading.setText('Please wait ...')
            self.progressBar.setValue(i)
        ##### run model
        self.multi_files_class()

    def multi_files_class(self):
        ## loading model
        #model = load_model("ResNet50.h5")
        model = load_model("ResNet18 dim_224.h5")
        path= os.listdir('input')
        for file in path:
            full_input = os.path.join('input',file)
            #print(full_input)
            img = cv2.imread(full_input)
            input_dim=224
            img = cv2.resize(img,(input_dim,input_dim))
            img = np.reshape(img,[1,input_dim,input_dim,3])
            img = img/255.0
            #print(np.argmax(model.predict(img)))
            only_img_name= file.replace('.jpg','')
            result_path = "results\\"+only_img_name+".txt"
            f = open(result_path, "a")
            f.write(str(np.argmax(model.predict(img))))
            f.close()


    def show_gallery(self,imageindex):
        self.loading.setVisible(False)
        self.progressBar.setVisible(False)
        self.next.setVisible(True)
        self.back.setVisible(True)
        self.groupBox.setVisible(True)
        self.gallery.setVisible(True)
        self.currentIndex = imageindex
        name_img = os.path.basename(self.set_files[imageindex])
        only_img_name= name_img.replace('.jpg','')
        self.gallery.setText('')
        ##### here we resize image 
        img = imread(self.set_files[imageindex])
        img = cv2.resize(img,(300,300))
        frame = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        image_re = QImage(frame,frame.shape[1],frame.shape[0],frame.strides[0],QImage.Format_RGB888)
        pimax = QPixmap(image_re)
        self.gallery.setPixmap(pimax)
        self.gallery.setScaledContents(True)
        path_txt = only_img_name+".txt"
        f = open(os.path.join( 'results', path_txt ),'r')
        class_data = str(f.read())
        self.class_info(class_data)
        f.close()
        return self.currentIndex
    
    def display_results(self):
        image_input = os.listdir('input')
        for file in image_input:
            only_img_name= file.replace('.jpg','')
            name_and_txt = only_img_name+".txt"
            f = open(os.path.join( 'results', name_and_txt ),'r')
            class_data=f.read()
            path2 = os.getcwd()+'\\'+'output\\'+class_data
            file_status = os.path.isdir(path2)
            if file_status == False:
                os.mkdir(path2)
            shutil.copyfile("input/"+file, 'output/'+class_data+"/"+file)
            shutil.copyfile("results/"+name_and_txt, 'output/'+class_data+"/"+name_and_txt)

        os.system('explorer.exe "output"')
        
    def class_info(self,class_data):
        if int(class_data) == 1:
            self.class_text.setText('Extensive pitting (SC3)')
            self.level_type.setText('Level 1B')
            self.sentance.setText('There are immediate problems in ​the areas, Treatment 0-1 years')
            self.sentance.setStyleSheet('color:red')

        if int(class_data) == 2:
            self.class_text.setText('Localised pitting (SC2)')
            self.level_type.setText('Level 2')
            self.sentance.setText('There are individual problems in this area, Treatment 2-4 years')
            self.sentance.setStyleSheet('color:orange')

        if int(class_data) == 3:
            self.class_text.setText('Rash rusting, no loss of section (SC1)')
            self.level_type.setText('Level 2')
            self.sentance.setText('There are individual problems in this area, Treatment 2-4 years')
            self.sentance.setStyleSheet('color:orange')

        if int(class_data) == 4:
            self.class_text.setText('Paint system totally ineffective (PC5)')
            self.level_type.setText('Level 2')
            self.sentance.setText('There are individual problems in this area, Treatment 2-4 years')
            self.sentance.setStyleSheet('color:orange')

        if int(class_data) == 5:
            self.class_text.setText('Damage 5-40% top coat breakdown (PC4)')
            self.level_type.setText('Level 2')
            self.sentance.setText('There are individual problems in this area, Treatment 2-4 years')
            self.sentance.setStyleSheet('color:orange')
        
        
                

app = QApplication(sys.argv)
window = Window()
window.show()
sys.exit(app.exec_())

    
    

