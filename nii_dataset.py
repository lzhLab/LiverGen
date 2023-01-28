import os

import numpy as np
import torch
from PIL import Image

from torch.utils.data import DataLoader, Dataset

import cv2


mean_liver = 17.5955
std_liver = 41.7803
mean_vessel = 0.5159
std_vessel = 8.6955

def normalize(in_put_array):

	normal_res = (in_put_array-127.5)/127.5
	return normal_res





def png2_3D_array(file_path, size):
	png_list = os.listdir(file_path)
	png_list.sort(key=lambda x:int(x[2:-4]))
	for i in range(len(png_list)):
		png_list[i] = int(png_list[i][2:-4])
	png_list.sort()
	array = []
	array = np.zeros([len(png_list), size, size], dtype='float32')
	for i in range(len(png_list)):
		png_name = 'p_' + str(png_list[i]) + '.png'

		img_path = os.path.join(file_path, png_name)
		img_array = Image.open(img_path)
		#img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

		if img_array.size != (size, size):
			img_array = img_array.resize((size, size))
		img_array_np = np.asarray(img_array)
		array[i, :, :] = img_array_np

	return array

class get_liver_vessel(Dataset):
	def __init__(self, data_path=None):
		if data_path==None or not os.path.exists(data_path):
			print('数据路径有误请检查')
		self.liver_dataset = os.path.join(data_path, 'liver')
		self.vessel_dataset = os.path.join(data_path, 'vessel')
		self.mask_dataset = os.path.join(data_path, 'mask')
		self.liver_p_id_list = os.listdir(self.liver_dataset)
		self.vessel_p_id_list = os.listdir(self.vessel_dataset)
		self.mask_p_id_list = os.listdir(self.mask_dataset)
		if len(self.liver_p_id_list) != len(self.vessel_p_id_list) or len(self.liver_p_id_list) != len(self.mask_p_id_list):
			print('肝脏与血管无法完全配成对！')
		#self.make_array = png2_3D_array()


	def __getitem__(self, idx):
		pat_id = self.liver_p_id_list[idx]
		pat_liver_path = os.path.join(self.liver_dataset, pat_id)
		pat_vessel_path = os.path.join(self.vessel_dataset, pat_id)
		pat_mask_path = os.path.join(self.mask_dataset, pat_id)
		png_name = os.listdir(pat_liver_path)
		png_name.sort(key=lambda x:int(x[2:-4]))



		liver_array= png2_3D_array(file_path=pat_liver_path, size=320)
		vessel_array = png2_3D_array(file_path=pat_vessel_path, size=320)
		mask_array = png2_3D_array(file_path=pat_mask_path, size=320)


		
		liver_array = normalize(liver_array)

		liver_array = torch.from_numpy(liver_array)



		vessel_array = vessel_array/255
		vessel_array = np.where(vessel_array < 0.5, np.zeros_like(vessel_array), np.ones_like(vessel_array))

		vessel_array = torch.from_numpy(vessel_array)



		mask_array = mask_array/255
		mask_array = np.where(mask_array < 0.5, np.zeros_like(mask_array), np.ones_like(mask_array))

		# #这里是用于对不太好的mask进行腐蚀
		# mask_array = cv2.morphologyEx(mask_array,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
		# mask_array = cv2.morphologyEx(mask_array, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

		mask_array = torch.from_numpy(mask_array)




		liver_array = liver_array.unsqueeze(0)
		vessel_array = vessel_array.unsqueeze(0)
		mask_array = mask_array.unsqueeze(0)
		return (liver_array, vessel_array, mask_array), pat_id, png_name

	def __len__(self):
		return len(self.liver_p_id_list)



if __name__ == "__main__":
	data_path = r'dataset/train'
	dataset = get_liver_vessel(data_path)
	loader = DataLoader(dataset, batch_size=1)
	for (l, v, m), id, list in loader:
		print ((l.shape), (v.shape))
