import torch.nn.functional as F

def dice_loss(pred, target, smooth=1.):
    # print(pred)
    pred = F.sigmoid(pred)
    # print(pred)
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=-1).sum(dim=-1)

    loss = (1 - ((2. * intersection + smooth) / (
                pred.sum(dim=-1).sum(dim=-1) + target.sum(dim=-1).sum(dim=-1) + smooth)))

    return loss.mean()