# HEART DISEASE AI DATATHON 2021

주최: 과학기술정보통신부, 한국지능정보사회진흥원  
주관: 서울대학교병원, 연세대학교 산학협력단  
https://github.com/DatathonInfo/H.D.A.I.2021

## Task: 심장 초음파 영상 좌심실 영역 분할
- 주제1.심초음파 데이터셋을 활용한 좌심실 분할 AI모델 공모(Apical 2 chamber(A2C) & Apical 4 chamber(A4C) view 이미지를 활용해 좌심실 분할하는 딥러닝 모델 개발)
## Team: 🫀리슨투마핫빝
강현구 손종욱 유지수 이선빈 홍정민 from SKKU DMLAB

## Key Points
1. **Pretraining**
* Base model pretrained with both A2C and A4C
* Fine-tuned with target tasks (A2C or A4C)
2. **Architectures**
* Unet (Pretrained on 2D Brain MRI segmentation)
* Caranet (SOTA on Kvasir dataset)
3. **Ensemble**
* Ensemble of best 4 models (Unet 2, Caranet 2)
4. **Test Time Augmentation**
* Augmentated images used for test prediction

## Dependencies ##
* python 3.7.11
* pytorch 1.9.1
* torchvision 0.10.1
* albumentations 1.1.0

## Data Path 설정
1. 데이터 폴더를 생성한다.
```bash
mkdir data
```
2. 데이터를 data 폴더에 넣는다

## Model Path 설정
1. 모델 폴더를 생성한다.
```bash
mkdir models
```
2. 모델을 models 폴더에 넣는다

## Directory
```bash
├── data
│   ├── train
│   │   ├── A2C
│   │   └── A4C
│   ├── validation
│   │   ├── A2C
│   │   └── A4C
│   └── test (expected)
│       └── A2C
│       └── A4C
├── models
│   ├── A2C_caranet_1.pt
│   ├── A2C_caranet_2.pt
│   ├── A2C_unet_1.pt
│   ├── A2C_unet_2.pt
│   ├── A4C_caranet_1.pt
│   ├── A4C_caranet_2.pt
│   ├── A4C_unet_1.pt
│   └── A4C_unet_2.pt
├── models.py
├── test_predict_mask.py
├── final_utils.py
├── final_test.py
├── utils.py
├── preprocess.py
├── main.py
└── train.sh

``` 
# Train

## How To Use
```bash
bash train.sh
```

# Validation Evalutation

## How To Use
- For A2C task
```bash
python final_test.py --data_path "data/validation" --data_type "A2C" 
```

- For A4C task
```bash
python final_test.py --data_path "data/validation" --data_type "A4C"
```


# Test Prediction

## How To Use
- For A2C task
```bash
python test_predict_mask.py --data_path "data/test" --data_type "A2C" 
```

- For A4C task
```bash
python test_predict_mask.py --data_path "data/test" --data_type "A4C"
```
