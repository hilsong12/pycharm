import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtGui import QPixmap
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

form_window = uic.loadUiType('./cat_and_dog.ui')[0]
# .ui 파일을 읽어서 Python UI 클래스(폼)를 가져온다.

class Exam(QWidget, form_window):
    # QWidget(기본 윈도우 기능), form_window(UI 기능) 둘 다 상속한다.

    def __init__(self):
        super().__init__()   # QWidget 초기화
        self.path = None     # 이미지 경로 저장 변수
        self.setupUi(self)   # UI 구성요소들을 현재 객체에 적용
        self.model = load_model('./cat_and_dog_binary_classification2_0.8676000237464905.h5')
        self. pushButton.clicked.connect(self.button_slot)



    def button_slot(self):
      self.path =  QFileDialog.getOpenFileName(self, 'Opne file', '/home/user12/Downloads', 'Image Files(*.jpg);;All file(*.*)')

      print(self.path)
      pixmap = QPixmap(self.path[0])
      self.label.setPixmap(pixmap)

      try:
          img = Image.open(self.path[0])
          img = img.convert('RGB')  # RGB 3채널로 변환
          img = img.resize((64, 64))  # 모델 입력에 맞게 리사이즈
          data = np.asarray(img)  # numpy 배열 변환
          data = data / 255  # 정규화 (0~255 → 0~1)
          data = data.reshape(1, 64, 64, 3)  # 모델 입력형태 (배치 포함)

          predict_value = self.model.predict(data)
          print(predict_value)
          if predict_value > 0.5:
              self.label_2.setText('개일 확률 ' + str((predict_value[0][0]*100).round()) + '%')
          else:
              self.label_2.setText('고영희 확률 ' + str((100-predict_value[0][0]*100).round()) + '%')

      except:
          print('error')


app = QApplication(sys.argv)
mainWindow = Exam()     # UI 적용된 윈도우 객체 생성
mainWindow.show()        # 화면에 표시
sys.exit(app.exec_())    # 이벤트 루프 실행  #앱이 종료하면 파이썬 종료해라.

#디자인은 ui로 만들고 안에 버튼누르면 어떻게 할지는 코드로 작성한다.
