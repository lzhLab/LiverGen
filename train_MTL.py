import torch
from utils import save_checkpoint_disc, save_checkpoint_gen, load_checkpoint_disc, load_checkpoint_gen, save_3Darray_as_png

import torch.nn as nn
import torch.optim as optim
import config

from dice_score import dice_loss
from tensorboardX import SummaryWriter
from nii_dataset import get_liver_vessel
from Generator_MTL import get_model
from Discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from min_norm_solvers import MinNormSolver, gradient_normalizers
torch.backends.cudnn.benchmark = True



def train_fn_mtl(disc, gen, loader, opt_disc, opt_gen, loss_fn, BCE, writer, train_step):



    loop = tqdm(loader, leave=True)
    for idx, ((y, x, m), id, list) in enumerate(loop):
        x = x.to(config.DEVICE)#vessel_input
        label = {}
        label['liver'] = y.to(config.DEVICE)#label_liver
        label['mask'] = m.to(config.DEVICE)#label_mask


        loss_data = {}
        grads = {}
        scale = {}
        out = {}



        opt_gen.zero_grad()
        with torch.no_grad():
            x_data = x.data
            bl, skc = gen['encoder'](x_data)
        bl_clone = bl.data.clone().requires_grad_(True)

        for t in ['liver', 'mask']:
            opt_gen.zero_grad()
            out[t] = gen[t](bl_clone, skc)
        
        
            
        
        for t in ['liver', 'mask']:
            if t == 'mask':
                loss = (loss_fn[t](out[t], label[t]) + dice_loss(out[t], label[t]))
            else:
                loss = loss_fn[t](out[t], label[t])
            loss_data[t] = loss.data
            loss.backward()

            grads[t] = []

            grads[t].append(bl_clone.grad.data.clone().requires_grad_(False))
            bl_clone.grad.data.zero_()
        gn = gradient_normalizers(grads, loss_data, 'loss+')
        for t in ['liver', 'mask']:
            for gr_i in range(len(grads[t])):
                grads[t][gr_i] = grads[t][gr_i] / gn[t]
            sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in ['liver', 'mask']])
            for i, t in enumerate(['liver', 'mask']):
                scale[t] = float(sol[i])




        #train disc
        x_data = x.data
        bl, skc = gen['encoder'](x_data)
        for t in ['liver', 'mask']:
            out[t] = gen[t](bl_clone, skc)

        final_res = torch.where(out['mask']>=0.5, out['liver'], torch.zeros_like(out['liver'])-1)
        D_real = disc(x, label['liver'])
        D_real_loss = BCE(D_real,torch.ones_like(D_real))
        D_fake = disc(x, final_res.detach())
        D_fake_loss = BCE(D_fake, torch.zeros_like(D_fake))
        k = (D_fake_loss + D_real_loss)
        D_loss = k / 2

        opt_disc.zero_grad()
        D_loss.backward()
        opt_disc.step()


        #train generator
        gen_loss = {}

        D_fake = disc(x, final_res)
        G_fake_loss = BCE(D_fake, torch.ones_like(D_fake))
        for i, t in enumerate(['liver', 'mask']):
            if t == 'mask':
                out_dice = dice_loss(out[t], label[t])
                out_bce = loss_fn[t](out[t], label[t])
                out_edge = (out_dice + out_bce) / 2
                gen_loss[t] = out_edge
            else:
                out_l1 = loss_fn[t](out[t], label[t])
                gen_loss[t] = out_l1
            loss_data[t] = gen_loss[t].data
            if i > 0:
                loss = loss + scale[t]*gen_loss[t]
            else:
                loss = scale[t]*gen_loss[t]
        G_loss = loss * 100 + G_fake_loss
        opt_gen.zero_grad()
        G_loss.backward()
        opt_gen.step()

        if idx % 1 == 0:
            writer.add_scalar('train__loss_edge', out_edge.mean().item(), train_step)
            writer.add_scalar('train__loss_bce', out_bce.mean().item(), train_step)
            writer.add_scalar('train__loss_dice', out_dice.mean().item(), train_step)
            writer.add_scalar('train__loss_l1', out_l1.mean().item(), train_step)
            writer.add_scalar('train__loss_Gfake', G_fake_loss.mean().item(), train_step)
            writer.add_scalar('train__loss_G', G_loss.mean().item(), train_step)
            writer.add_scalar('train__loss_D', D_loss.mean().item(), train_step)
            train_step = train_step + 1

            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
                D_loss = D_loss.mean().item(),
                G_loss=G_loss.mean().item(),

                G_fake_loss=G_fake_loss.mean().item(),
                BCE=out_bce.mean().item(),
                L1=out_l1.mean().item(),
                scarl_l = scale['liver'],
                scarl_m = scale['mask'],

            )


def val_fn_mtl(gen, val_loader, writer, val_step, loss_fn, BCE):
    val_loss_bce_list = []
    val_loss_dice_list = []
    val_loss_l1_list = []
    for m in gen:
        gen[m].eval()
    for idx_val, ((l2, v2, m2), id2, list2) in enumerate(val_loader):
        liver, vessel, mask = l2.to(config.DEVICE), v2.to(config.DEVICE), m2.to(config.DEVICE)
        bl2, skc2 = gen['encoder'](vessel)
        mask_pre = gen['mask'](bl2,skc2)
        img_pre = gen['liver'](bl2,skc2)
        mask_loss_BCE = loss_fn['mask'](mask_pre, mask)
        mask_loss_Dice = dice_loss(mask_pre, mask)
        img_loss_l1 = loss_fn['liver'](img_pre, liver)
        val_loss_bce_list.append(mask_loss_BCE.mean().item())
        val_loss_dice_list.append(mask_loss_Dice.mean().item())
        val_loss_l1_list.append(img_loss_l1.mean().item())

    val_loss_bce_eve = sum(val_loss_bce_list) / len(val_loss_bce_list)
    val_loss_dice_eve = sum(val_loss_dice_list) / len(val_loss_dice_list)
    val_loss_l1_eve = sum(val_loss_l1_list) / len(val_loss_l1_list)

    writer.add_scalar('val_bce_eve', val_loss_bce_eve, val_step)
    writer.add_scalar('val_dice_eve', val_loss_dice_eve, val_step)
    writer.add_scalar('val_l1_eve', val_loss_l1_eve, val_step)











def main():

    gen = get_model(1, 32)
    for m in gen:
        gen[m].to(config.DEVICE)

    gen_para = []

    for m in gen:
        gen_para += gen[m].parameters()

    disc = Discriminator(1).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(gen_para, lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    writer = SummaryWriter(logdir='runs/')

    loss_fn = {}
    loss_fn['liver'] = nn.L1Loss()

    loss_fn['mask'] = nn.BCELoss()
    BCE = nn.BCEWithLogitsLoss()

    if config.LOAD_MODEL:
        load_checkpoint_gen(
            r"./model", gen, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint_disc(
            r"./model", disc, opt_disc, config.LEARNING_RATE,
        )

    train_dataset = get_liver_vessel(data_path=config.TRAIN_DIR)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )


    val_dataset = get_liver_vessel(data_path=config.VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)

    test_dataset = get_liver_vessel(data_path=config.TEST_DIR)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    

    
    train_step = 0
    for epoch in range(config.NUM_EPOCHS):
        val_step = epoch
        train_fn_mtl(
            disc, gen, train_loader, opt_disc, opt_gen, loss_fn, BCE, writer, train_step=(epoch * (int(1483/config.BATCH_SIZE)+1) + train_step)
        )
        val_fn_mtl(gen, val_loader, writer, val_step, loss_fn, BCE)
        
        
        if epoch % 5 == 0:
            save_3Darray_as_png(gen, test_loader, epoch, folder=r'./res')
        if epoch % 50 == 0:
            save_checkpoint_gen(gen, opt_gen, floder=config.CHECKPOINT_GEN + "epoch_" + str(epoch))
            save_checkpoint_disc(disc, opt_disc, floder=config.CHECKPOINT_DISC + "epoch_" + str(epoch))
            



    

if __name__ == "__main__":
    main()
