import torch
import torch.nn as nn
import numpy as np

class Down_Block(nn.Module):
	def __init__(self, in_channels, out_channels, kernel, pad, str, act="relu", use_dropout=False):
		super(Down_Block, self).__init__()
		self.Down_conv = nn.Sequential(
			nn.Conv3d(in_channels, out_channels, kernel, pad, str, bias=False, padding_mode="reflect"),
			nn.BatchNorm3d(out_channels),
			nn.ReLU if act == "relu" else nn.LeakyReLU(0.2),
		)
		self.use_dropout = use_dropout
		self.dropout = nn.Dropout(0.5)

	def forward(self, x):
		x = self.Down_conv(x)
		return self.dropout(x) if self.use_dropout else x



class Up_block(nn.Module):
	def __init__(self, in_channels, out_channels, kernel, pad, str, act="relu", use_dropout=False):
		super(Up_block, self).__init__()
		self.Up_conv = nn.Sequential(
			nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel, padding=pad, stride=str, bias=False),
			nn.BatchNorm3d(out_channels),
			nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),

		)
		self.use_dropout = use_dropout
		self.dropout = nn.Dropout(0.5)

	def forward(self, x):
		x = self.Up_conv(x)
		return self.dropout(x) if self.use_dropout else x




class Generator_encoder(nn.Module):
	def __init__(self, in_channels=1, features=64):
		super(Generator_encoder, self).__init__()
		self.initial_down = nn.Sequential(
			nn.Conv3d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
			nn.LeakyReLU(0.2),
		)
		self.down1 = Down_Block(features, features * 2, 4, 2, 1, act="Leaky", use_dropout=False)
		self.down2 = Down_Block(features * 2, features * 4, 4, 2, 1, act="Leaky", use_dropout=False)
		self.down3 = Down_Block(features * 4, features * 8, 4, 2, 1, act="Leaky", use_dropout=False)
		self.down4 = Down_Block(features * 8, features * 16, 4, 2, 1, act="Leaky", use_dropout=False)

		self.bottleneck = nn.Sequential(
			nn.Conv3d(features * 16, features * 16, 4, 2, 1),
			nn.ReLU()
		)


	def forward(self, x):
		d1 = self.initial_down(x)
		d2 = self.down1(d1)
		d3 = self.down2(d2)
		d4 = self.down3(d3)
		d5 = self.down4(d4)
		bottleneck = self.bottleneck(d5)
		skip_connect = [d1, d2, d3, d4, d5]
		return bottleneck, skip_connect


class Generator_decoder_liver(nn.Module):
	def __init__(self, in_channels=1, features=64):
		super(Generator_decoder_liver, self).__init__()

		self.up1 = Up_block(features * 16, features * 16, 4, 1, 2, act="relu", use_dropout=True)
		self.up2 = Up_block(features * 16 * 2, features * 8, 4, 1, 2, act="relu", use_dropout=True)
		self.up3 = Up_block(features * 8 * 2, features * 4, 4, 1, 2, act="relu", use_dropout=True)
		self.up4 = Up_block(features * 4 * 2, features * 2, 4, 1, 2, act="relu", use_dropout=True)
		self.up5 = Up_block(features * 2 * 2, features * 1, 4, 1, 2, act="relu", use_dropout=True)

		self.final_up = nn.Sequential(
			nn.ConvTranspose3d(features * 1 * 2, in_channels, kernel_size=4, stride=2, padding=1),
			nn.Tanh(),
		)


	def forward(self, bottleneck, skip_connection):
		up1 = self.up1(bottleneck)
		up2 = self.up2(torch.cat([up1, skip_connection[4]], 1))
		up3 = self.up3(torch.cat([up2, skip_connection[3]], 1))
		up4 = self.up4(torch.cat([up3, skip_connection[2]], 1))
		up5 = self.up5(torch.cat([up4, skip_connection[1]], 1))
		pre_img = self.final_up(torch.cat([up5, skip_connection[0]], 1))
		return pre_img


class Generator_decoder_mask(nn.Module):
	def __init__(self, in_channels=1, features=64):
		super(Generator_decoder_mask, self).__init__()

		self.up1 = Up_block(features * 16, features * 16, 4, 1, 2, act="relu", use_dropout=True)
		self.up2 = Up_block(features * 16 * 2, features * 8, 4, 1, 2, act="relu", use_dropout=True)
		self.up3 = Up_block(features * 8 * 2, features * 4, 4, 1, 2, act="relu", use_dropout=True)
		self.up4 = Up_block(features * 4 * 2, features * 2, 4, 1, 2, act="relu", use_dropout=True)
		self.up5 = Up_block(features * 2 * 2, features * 1, 4, 1, 2, act="relu", use_dropout=True)

		self.final_up_sig = nn.Sequential(
			nn.ConvTranspose3d(features * 1 * 2, in_channels, kernel_size=4, stride=2, padding=1),
			nn.Sigmoid(),
		)


	def forward(self, bottleneck, skip_connection):
		up1 = self.up1(bottleneck)
		up2 = self.up2(torch.cat([up1, skip_connection[4]], 1))
		up3 = self.up3(torch.cat([up2, skip_connection[3]], 1))
		up4 = self.up4(torch.cat([up3, skip_connection[2]], 1))
		up5 = self.up5(torch.cat([up4, skip_connection[1]], 1))
		pre_mask = self.final_up_sig(torch.cat([up5, skip_connection[0]], 1))
		return pre_mask

def test():
	x = torch.randn((1, 1, 64, 320, 320))
	model = Generator_encoder(1, 64)
	model2 = Generator_decoder_liver(1, 64)
	bl, sc = model(x)
	res = model2(bl, sc)
	print(res.shape)

def get_model(in_ch, feature):
	model = {}
	model['encoder'] = Generator_encoder(in_ch, feature)
	model['liver'] = Generator_decoder_liver(in_ch, feature)
	model['mask'] = Generator_decoder_mask(in_ch, feature)
	return model


if __name__ == "__main__":
	# model = Generator(1, 64)
	# print(model)
	# test()
	model = {}
	model['encoder'] = Generator_encoder(1, 1)
	model['decoder_liver'] = Generator_decoder_liver(1, 1)
	model['decoder_mask'] = Generator_decoder_mask(1, 1)
	x = torch.randn((1, 1, 64, 320, 320))
	b,s = model['encoder'](x)
	res = model['decoder_liver'](b,s)
	print(res.shape)