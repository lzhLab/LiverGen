import numpy as np

import torch
from utils import save_checkpoint_disc, save_checkpoint_gen, load_checkpoint_disc, load_checkpoint_gen, save_3Darray_as_png

import torch.nn as nn
import torch.optim as optim
import config
import cv2
from dice_score import dice_loss
from tensorboardX import SummaryWriter
from nii_dataset import get_liver_vessel
from torch.utils.data import DataLoader
from Generator_MTL import get_model





def get_RVD(pred, gt):
	return abs(pred.sum() - gt.sum()) / gt.sum()








def get_list_eve(inputlist):
	return sum(inputlist)/len(inputlist)







def val_fn_mtl(gen, val_loader):
    #这个代码主要评估的是生成的肝脏mask的准确程度
	val_loss_dice_list = []
	for m in gen:
		gen[m].eval()
	for idx_val, ((l2, v2, m2), id2, list2) in enumerate(val_loader):
		with torch.no_grad():
			liver, vessel, mask = l2.to(config.DEVICE), v2.to(config.DEVICE), m2.to(config.DEVICE)
			bl2, skc2 = gen['encoder'](vessel)
			mask_pre = gen['mask'](bl2, skc2)
			img_pre = gen['liver'](bl2, skc2)

		# mask = mask.squeeze(0).squeeze(0)
		# mask = mask.cpu().numpy().astype(np.uint8)
		# mask_pre = mask_pre.squeeze(0).squeeze(0)
		# mask_pre = mask_pre.cpu().numpy().astype(np.uint8)
		# mask_pre = np.where(mask_pre<=0.5, np.zeros_like(mask_pre), np.ones_like(mask_pre))


		try:
			mask_loss_Dice = dice_loss(mask_pre, mask)
			print(id2)
			print(mask_loss_Dice.mean().item())

			val_loss_dice_list.append(mask_loss_Dice.mean().item())
		except:
			pass



	val_loss_dice_eve = sum(val_loss_dice_list) / len(val_loss_dice_list)
	print(val_loss_dice_eve)




def main():
	gen = get_model(1, 32)
	for m in gen:
		gen[m].to(config.DEVICE)
	gen_para = []
	for m in gen:
		gen_para += gen[m].parameters()
	opt_gen = optim.Adam(gen_para, lr=config.LEARNING_RATE, betas=(0.5, 0.999))
	load_checkpoint_gen(r'./model/epoch_200/gen.pth.tar', gen, opt_gen, config.LEARNING_RATE,)

	
	val_dataset = get_liver_vessel(r'./valdata')
	val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

	#val_fn_mtl(gen, val_loader)

	save_3Darray_as_png(gen, val_loader, epoch=200, folder=r'./res/')


if __name__ == "__main__":
	main()