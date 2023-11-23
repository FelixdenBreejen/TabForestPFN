import torch
import einops

class CrossEntropyLossExtraBatch(torch.nn.CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input, target):
        """
        Input has shape (batch_size, num_samples, num_classes)
        Target has shape (batch_size, num_samples)

        Compared to the original CrossEntropyLoss, accepts (batch_size, num_samples) as batch
        """

        input = einops.rearrange(input, 'b s c -> (b s) c')
        target = einops.rearrange(target, 'b s -> (b s)')

        return super().forward(input, target)



