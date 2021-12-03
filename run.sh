## pretrain

for aug_t in 1;
    do 
    # python main.py --learning_rate 1e-4 --weight_decay 1e-10 --model_type 'unet' --data_type 'both' --augmentation_type $aug_t
    python main.py --learning_rate 1e-5 --weight_decay 1e-10 --model_type 'unet' --data_type 'both' --augmentation_type $aug_t
    python main.py --learning_rate 1e-6 --weight_decay 1e-10 --model_type 'unet' --data_type 'both' --augmentation_type $aug_t
done


# python main.py --learning_rate 1e-5 --weight_decay 1e-6 --model_type 'unet' --data_type 'both' --augmentation_type 2
# python main.py --learning_rate 1e-5 --weight_decay 1e-10 --model_type 'unet' --data_type 'both' --augmentation_type 2