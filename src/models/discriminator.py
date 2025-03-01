import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, img_channels=3, features=64):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                img_channels * 2,  # For concatenated input (real/fake + condition)
                features,
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )
        
        self.layers = nn.ModuleList([
            CNNBlock(features, features * 2, stride=2),
            CNNBlock(features * 2, features * 4, stride=2),
            CNNBlock(features * 4, features * 8, stride=1),
        ])
        
        self.final = nn.Conv2d(
            features * 8, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"
        )

    def forward(self, x, y):
        # x is the input image and y is the condition
        x = torch.cat([x, y], dim=1)  # Concatenate along the channel dimension
        x = self.initial(x)
        
        for layer in self.layers:
            x = layer(x)
            
        return self.final(x)


def test():
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    model = Discriminator(img_channels=3, features = 64)
    preds = model(x, y)
    print(model)
    print(preds.shape)


if __name__ == "__main__":
    test()