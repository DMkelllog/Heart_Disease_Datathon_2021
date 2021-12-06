# Heart_Disease_Datathon_2021

## Data Path 설정
1. 데이터 폴더를 생성한다.
```bash
mkdir data
```
2. 데이터를 data 폴더에 넣는다

## Model Path 설정
1. 데이터 폴더를 생성한다.
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
│   ├── A2C_model.pt
│   └── A4C_model.pt
├── test_predict_mask.py
├── final_utils.py
└── Final_test.py
``` 
## How To Use
- For A2C task
```bash
python Final_test.py --data_path "data/test" --data_type "A2C" --pretrained_path "models/A2C_model.pt"
```

- For A4C task
```bash
python Final_test.py --data_path "data/test" --data_type "A4C" --pretrained_path "models/A4C_model.pt"
```
