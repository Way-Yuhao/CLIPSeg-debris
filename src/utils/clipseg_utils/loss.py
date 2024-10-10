import torch.nn.functional as F

class BinaryCrossEntropyWithLogitsLoss:
    def __init__(self, weight=None, size_average=None, reduce=None, reduction="mean", pos_weight=None):
        self.weight = weight
        self.size_average = size_average
        self.reduce = reduce
        self.reduction = reduction
        self.pos_weight = pos_weight

    def __call__(self, input, target):
        return F.binary_cross_entropy_with_logits(
            input,
            target,
            weight=self.weight,
            size_average=self.size_average,
            reduce=self.reduce,
            reduction=self.reduction,
            pos_weight=self.pos_weight
        )
