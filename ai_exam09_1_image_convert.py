
from PIL import Image
import glob
import numpy as np
from sklearn.model_selection import train_test_split

img_dir = '/media/user12/data/pycharm/cat_dog/train/'
categories = ['cat', 'dog']

image_w = 64
image_h = 64

pixel = image_h * image_w * 3   #컬러 이미지여서 3채널

X=[]
Y=[]
files = None

for idx, category in enumerate (categories):   #0 은 cat 1은 dog
    files = glob.glob(img_dir + category + '*.jpg')
    for i, f in enumerate(files):
        try:                   #try ,except 시도를 하다가 에러가나면 except를 시행
            img = Image.open(f)
            img = img.convert('RGB')   #컬러도 여러가지 포맷이 있어서 하나로 맞췄다.
            data = img.resize((image_w,image_h))    #사이즈는 튜플로 준다.
            X.append(data)
            Y.append(idx)   #라벨에 맞춰서 저장
            if i% 300 ==0:   #프로그램이 잘 돌아가는지 확인하기 위해 300번 실행 될때 프린트 ..
                print(category, ':', f)
        except:           #없으면 에러나고 프로그램 멈춤, 에러나도 프로그램 멈추지 않게 해주는
            print(category,i,'error')

X = np.array(X)
Y = np.array(Y)
X = X/255
print(X[0])
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.1)

np.save('./binary_X_train.npy', X_train)
np.save('./binary_X_test.npy', X_test)
np.save('./binary_Y_train.npy', Y_train)
np.save('./binary_Y_test.npy', Y_test)



    # print(files)
# '/media/user12/data/pycharm/cat_dog/train/cat*.jpg'    *은 와일드카드    cat으로 시작하고 .jpg로 끝나는 모든 애들