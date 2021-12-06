# Heart_Disease_Datathon_2021

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
│   ├── validation
│   │   ├── A2C
│   │   └── A4C
│   └── test
│       ├── A2C
│       └── A4C
├── models
│   ├── A2C_caranet_1.pt
│   ├── A2C_caranet_2.pth
│   ├── A2C_unet_1.pth
│   ├── A2C_unet_2.pth
│   ├── A4C_caranet_1.pt
│   ├── A4C_caranet_2.pth
│   ├── A4C_unet_1.pth
│   └── A4C_unet_2.pth
├── test_predict_mask.py
├── final_utils.py
└── Final_test.py
``` 
## How To Use
- For A2C task
```bash
python Final_test.py --data_path "data/test" --data_type "A2C" --pretrained_path "models"
```

- For A4C task
```bash
python Final_test.py --data_path "data/test" --data_type "A4C" --pretrained_path "models"
```
