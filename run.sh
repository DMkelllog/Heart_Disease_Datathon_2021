## pretrain


# Pretrain
python main.py --model_type 'caranet' --data_type 'A2C' --augmentation_type 3 --learning_rate 1e-3 --pretrained_path "pre_both_1_caranet_0.001" --pretrained_epoch "full"
python main.py --model_type 'caranet' --data_type 'A4C' --augmentation_type 3 --learning_rate 1e-3 --pretrained_path "pre_both_1_caranet_0.001" --pretrained_epoch "full"
