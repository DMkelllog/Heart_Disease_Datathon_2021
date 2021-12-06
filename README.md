# Heart_Disease_Datathon_2021

## directory
```bash
├── validation
│   ├── A2C
│   └── A4C
├── models
│   ├── A2C_model.pt
│   └── A4C_model.pt
├── final_utils.py
└── Final_test.py
``` 
## How To Use
- For A2C task
```bash
python Final_test.py --data_path "validation" --data_type "A2C" --pretrained_path "models/A2C_model.pt"
```

- For A4C task
```bash
python Final_test.py --data_path "validation" --data_type "A4C" --pretrained_path "models/A4C_model.pt"
```
