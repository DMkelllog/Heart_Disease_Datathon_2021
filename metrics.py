import torch
import torch.nn as nn

# DSC
def get_DC(SR,GT,threshold=0.5):
    # DC : Dice Coefficient
    SR = SR > threshold
    GT = GT == torch.max(GT)

    Inter = torch.sum((SR+GT)==2)
    DC = float(2*Inter)/(float(torch.sum(SR)+torch.sum(GT)) + 1e-6)

    return DC

# JS
def get_JS(SR,GT,threshold=0.5):
    # JS : Jaccard similarity
    SR = SR > threshold
    GT = GT == torch.max(GT)
    
    Inter = torch.sum((SR+GT)==2)
    Union = torch.sum((SR+GT)>=1)
    
    JS = float(Inter)/(float(Union) + 1e-6)
    
    return JS

class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc

"""
for i, (images, GT) in enumerate(self.train_loader):
    # GT : Ground Truth

    images = images.to(self.device)
    GT = GT.to(self.device)

    # SR : Segmentation Result
    SR = self.unet(images)
    SR_probs = F.sigmoid(SR)
    SR_flat = SR_probs.view(SR_probs.size(0),-1)

    GT_flat = GT.view(GT.size(0),-1)
    loss = self.criterion(SR_flat,GT_flat)
    epoch_loss += loss.item()

    # Backprop + optimize
    self.reset_grad()
    loss.backward()
    self.optimizer.step()

    acc += get_accuracy(SR,GT)
    SE += get_sensitivity(SR,GT)
    SP += get_specificity(SR,GT)
    PC += get_precision(SR,GT)
    F1 += get_F1(SR,GT)
    JS += get_JS(SR,GT)
    DC += get_DC(SR,GT)
    length += images.size(0)

acc = acc/length
SE = SE/length
SP = SP/length
PC = PC/length
F1 = F1/length
JS = JS/length
DC = DC/length
"""