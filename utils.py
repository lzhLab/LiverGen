import numpy as np
import torch
import config
#from torchvision.utils import save_image
import os
import cv2





def img_clip(img):
    xmax = max(map(max, img))
    xmin = min(map(min, img))

    for x in range(0, 320):
        for y in range(0, 320):
            if img[x][y] != 0:
                img[x][y] = round(255 * (img[x][y] - xmin) / (xmax - xmin))
            else:
                pass

    return img

def make_mask(img):
    for x in range(0, 320):
        for y in range(0, 320):
            if img[x][y] <= 0.5:
                img[x][y] = 0
            else:
                img[x][y] = 255

    return img


def make_tahn(img):
    for x in range(0, 320):
        for y in range(0, 320):
            if img[x][y] <= 0:
                img[x][y] = 0
    return img


def save_img_form_array(array, name_list, path):
    if not os.path.exists(path):
        os.makedirs(path)

    for i in range(0, 64):
        img = array[i, :, :]

        img = (img*127.5) + 127.5

        img_name = str(name_list[i])[2:-3]
        save_path = os.path.join(path, img_name)
        cv2.imwrite(save_path, img)

def save_final_res_form_array(array, name_list, path):
    if not os.path.exists(path):
        os.makedirs(path)

    for i in range(0, 64):
        img = array[i, :, :]
        img = (img*127.5) + 127.5

        img_name = str(name_list[i])[2:-3]
        save_path = os.path.join(path, img_name)
        cv2.imwrite(save_path, img)



def save_mask_form_array(array, name_list, path):
    if not os.path.exists(path):
        os.makedirs(path)

    for i in range(0, 64):
        img = array[i, :, :]

        img = img*255
        img_name = str(name_list[i])[2:-3]
        save_path = os.path.join(path, img_name)
        cv2.imwrite(save_path, img)



def save_3Darray_as_png(gen, val_loader, epoch, folder):
    new_folder = os.path.join(folder, 'epoch' + str(epoch))
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    for idx_val, ((liver, vessel, mask), id, list) in enumerate(val_loader):

        save_path_liver = os.path.join(new_folder, str(id)[2:-3] + '/liver')

        save_path_mask = os.path.join(new_folder, str(id)[2:-3] + '/mask')

        save_path_finalres = os.path.join(new_folder, str(id)[2:-3] + '/final_res')
        if not os.path.exists(save_path_liver):
            os.makedirs(save_path_liver)
        if not os.path.exists(save_path_mask):
            os.makedirs(save_path_mask)
        liver, vessel = liver.to(config.DEVICE),vessel.to(config.DEVICE)
        for m in gen:
            gen[m].eval()

        with torch.no_grad():


            bottlneck, skc = gen['encoder'](vessel)
            liver_pred = gen['liver'](bottlneck, skc)
            mask_pred = gen['mask'](bottlneck, skc)

            liver_pred = liver_pred.squeeze(0).squeeze(0)
            liver_pred = liver_pred.cpu().numpy()
            mask_pred = mask_pred.squeeze(0).squeeze(0)
            mask_pred = mask_pred.cpu().numpy()

            final_res = np.where(mask_pred < 0.5, np.zeros_like(liver_pred)-1, liver_pred)
            save_img_form_array(liver_pred, list, save_path_liver)
            save_mask_form_array(mask_pred, list, save_path_mask)
            save_final_res_form_array(final_res, list, save_path_finalres)


def save_checkpoint_disc(model, optimizer, floder):
    print("=> Saving checkpoint disc")
    if not os.path.exists(floder):
        os.makedirs(floder)
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, floder + "/disc.pth.tar")

    

def save_checkpoint_gen(model, optimizer, floder):
    print("=> Saving checkpoint generator")
    if not os.path.exists(floder):
        os.makedirs(floder)
    checkpoint = {
        "state_dict_encoder": model["encoder"].state_dict(),
        "state_dict_liver": model["liver"].state_dict(),
        "state_dict_mask": model["mask"].state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, floder + "/gen.pth.tar")



def load_checkpoint_disc(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def load_checkpoint_gen(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")

    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model["encoder"].load_state_dict(checkpoint["state_dict_encoder"])
    model["liver"].load_state_dict(checkpoint["state_dict_liver"])
    model["mask"].load_state_dict(checkpoint["state_dict_mask"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def test_fn():
    array = np.zeros((128, 128, 128), dtype='uint8')
    name_list = r'./valdata/liver/10290094_1'
    name_list = os.listdir(name_list)
    path = r'./res/testsave/'
    save_img_form_array(array, name_list, path)


if __name__ == "__main__":
    test_fn()