
import torch
import torch.nn as nn


class CNNBlock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(CNNBlock, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv3d(
				in_channels, out_channels, 3, 2, 1, bias=False, padding_mode="reflect"
			),
			nn.BatchNorm3d(out_channels),
			nn.LeakyReLU(0.2),
		)

	def forward(self, x):
		return self.conv(x)




class Discriminator(nn.Module):
	def __init__(self, in_channels=1, features=16):
		super().__init__()
		self.initial = nn.Sequential(
			nn.Conv3d(
				in_channels * 2,
				features,
				kernel_size=3,
				stride=2,
				padding=1,
				padding_mode="reflect",
			),
			nn.LeakyReLU(0.2),
		)
		self.down1 = CNNBlock(features, features * 2)
		self.down2 = CNNBlock(features * 2, features * 4)
		self.down3 = CNNBlock(features * 4, features * 8)
		self.out = nn.Conv3d(features * 8, 1, kernel_size=3, stride=(1, 2, 2), padding=1, padding_mode="reflect")


	def forward(self, x, y):
		x1 = torch.cat([x, y], dim=1)
		x2 = self.initial(x1)
		x3 = self.down1(x2)
		x4 = self.down2(x3)
		x5 = self.down3(x4)
		x6 = self.out(x5)
		return x6


if __name__ == "__main__":
	x = torch.randn((1, 1, 64, 320, 320))
	y = torch.randn((1, 1, 64, 320, 320))
	z = torch.randn((1, 1, 64, 320, 320))
	model = Discriminator(in_channels=1)
	#print(model)
	k = model(x, y, z)
	print(k.shape)