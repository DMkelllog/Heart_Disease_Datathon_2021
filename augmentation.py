import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_training_augmentation(type=2):
    
    if type == 0:
        train_transform = [
            ToTensorV2(transpose_mask=True)
        ]

    if type == 1:
        train_transform = [
            A.OneOf(
                [
                    A.CLAHE(p=1),
                    A.RandomBrightness(p=1),
                    A.RandomContrast(p=1),
                ],
                p=0.5,
            ),
            A.ShiftScaleRotate(scale_limit=0, rotate_limit=0.1, shift_limit=0.1, p=0.5, border_mode=0),
            ToTensorV2(transpose_mask=True)
        ]

    if type == 2:
        train_transform = [
             A.OneOf(
                [
                    A.CLAHE(p=1),
                    A.RandomBrightness(p=1),
                    A.RandomContrast(p=1),
                ],
                p=0.5,
            ),
            A.HorizontalFlip(),
            A.Rotate(limit = 20),
            A.ShiftScaleRotate(shift_limit = 0.3),
            A.RandomScale(scale_limit = 0.3),
            ToTensorV2(transpose_mask=True)
        ]

    return A.Compose(train_transform)