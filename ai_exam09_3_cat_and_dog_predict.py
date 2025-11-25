from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np
import glob

categories = ['cat', 'dog']

# 저장된 모델(.h5) 불러오기
model = load_model('./cat_and_dog_binary_classification2_0.8676000237464905.h5')
model.summary()  # 모델 구조 출력

# 이미지 폴더 경로
img_dir = '/media/user12/data/pycharm/cat_dog/train/'

# 이미지 크기 설정 (모델 학습 시 사용한 입력 크기와 동일해야 함)
image_w = 64
image_h = 64

# 훈련 폴더에서 랜덤으로 개이미지/고양이 이미지 파일 선택
dog_files = glob.glob(img_dir + 'dog*.jpg')
dog_sample = np.random.randint(len(dog_files))
dog_sample_path = dog_files[dog_sample]

cat_files = glob.glob(img_dir + 'cat*.jpg')
cat_sample = np.random.randint(len(cat_files))
cat_sample_path = cat_files[cat_sample]

print(dog_sample_path)
print(cat_sample_path)

try:
    # 1) 개 이미지 로드 (랜덤 대신 직접 테스트용 이미지 사용)
    # img = Image.open(dog_sample_path)
    img = Image.open('/home/user12/Downloads/Funny_Dog_H.jpg')

    img.show()  # 원본 이미지 화면에 보여주기
    img = img.convert('RGB')  # RGB 3채널로 변환
    img = img.resize((image_w, image_h))  # 모델 입력에 맞게 리사이즈
    data = np.asarray(img)  # numpy 배열 변환
    data = data / 255  # 정규화 (0~255 → 0~1)
    dog_data = data.reshape(1, 64, 64, 3)  # 모델 입력형태 (배치 포함)

    # 2) 고양이 이미지 로드 (랜덤 대신 테스트 파일 사용)
    # img = Image.open(cat_sample_path)
    img = Image.open('/home/user12/Downloads/FELV-cat.jpg')

    img.show()
    img = img.convert('RGB')
    img = img.resize((image_w, image_h))
    data = np.asarray(img)
    data = data / 255
    cat_data = data.reshape(1, 64, 64, 3)

    # 예측 결과 출력 (round()로 0 또는 1로 변환)
    print('dog data (raw prediction):', model.predict(dog_data).round())
    print('cat data (raw prediction):', model.predict(cat_data).round())

except Exception as e:
    print('error:', e)

# 예측 결과를 label 이름으로 변환하여 출력
# 예: categories = ['cat', 'dog']
print('dog data:', categories[int(model.predict(dog_data).round()[0][0])])
print('cat data:', categories[int(model.predict(cat_data).round()[0][0])])

