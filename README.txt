文件名介绍：



config.py:保存实验所用的部分超参数

dice_score.py:diceloss函数

Discriminator_model.py:判别器模型，通过修改通道乘法器的基本参数可以调整模型的宽度来适应不同大小的数据集

Generator_MTL.py:生成器模型，通过修改通道乘法器的基本参数可以调整模型的宽度来适应不同大小的数据集

min_norm_solvers.py:解决多任务学习的梯度归一化问题，实现计算不同任务的动态权重

nii_dataset.py:dataset类，用于将图片序列文件夹压成3D矩阵后打包成张量进入模型训练

utils.py:工具函数，用于将生成结果保存为图片序列，保存和载入训练权重等

train_MTL.py：训练代码

val_mtl.py：验证代码，可载入"./model"路径下训练好的生成器权重用于生成肝脏图片

./valdata：目录下保存了一个患者的真实血管，肝脏mask以及肝脏图片，将血管输入模型后可以生成出逼真的肝脏图片结果，保存在./res目录下

./model:保存权重