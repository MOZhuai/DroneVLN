import torch

from learning.models.semantic_map.pinhole_camera_inv import PinholeCameraProjectionModuleGlobal
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import numpy as np



def test_affine(img, pose):
    map_projection = PinholeCameraProjectionModuleGlobal(32, 32, 30, 128, 72, 90)
    grid_maps = map_projection(pose)
    affine_img = F.grid_sample(img, grid_maps)
    print(img.shape, ", ", affine_img.shape)
    plt.imshow(img[0])
    plt.show()
    plt.imshow(affine_img[0])
    plt.show()


def show_fpvs(img):
    # img_ss = b_images[0].squeeze().permute(1,2,0).cpu().detach();plt.imshow(img_ss);plt.axis('off');plt.show()
    # plt.imshow(img[0])
    # plt.axis('off')
    # plt.show()
    # 创建一个随机的三通道张量（图像）
    if isinstance(img, torch.Tensor):
        img = img.cpu().detach()
    # 将张量转换为NumPy数组
    image_np = np.array(img)

    # 使用OpenCV绘制图像
    cv2.imshow('Tensor Image', image_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def show_img(img, title="None", save=False, axis=False):
    if isinstance(img, torch.Tensor):
        img = img.cpu().detach()
    if len(img.shape) == 4 and not save:
        vmin = 100000
        vmax = -100000
        for img_item in img:
            vmin = min(vmin, torch.min(img_item[0]))
            vmax = max(vmax, torch.max(img_item[0]))
        for img_item in img:
            plt.imshow(img_item[0], vmax=vmax, vmin=vmin)
            if not axis:
                plt.axis('off')
            plt.show()
    elif save:
        path = "../dataset/diffusion_compare/"
        for idx, img_item in enumerate(img):
            plt.title(title)
            plt.imshow(img_item[0])
            plt.savefig(path+title+str(idx))
            plt.show()
    elif len(img.shape) == 3:
        plt.title(title)
        plt.imshow(img[0])
        plt.show()
    elif len(img.shape) == 2:
        plt.title(title)
        plt.imshow(img)
        plt.show()
    else:
        print("Error Img Show: ", title)
