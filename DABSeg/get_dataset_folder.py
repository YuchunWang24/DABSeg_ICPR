
import os
'''
def get_brats_folder(dataset_folder, mode):
    assert mode in ["train","train_val", "test"]
   
    if mode == "train":
        return os.path.join(dataset_folder, "brats2021", "train")
    elif mode == "train_val":
        return os.path.join(dataset_folder, "brats2021", "val")
    elif mode == "test" :
        return os.path.join(dataset_folder, "brats2021", "test")

'''

def get_brats_folder(dataset_folder, mode):
    assert mode in ["train","train_val", "test"]
    # 你的代码目录:  /public/home/wyc/fusion-seg/my-fu-seg/CKD_Brury/11.13/CKD-TransBTS-main
    # 真实数据目录:  /public/home/wyc/fusion-seg/my-fu-seg/CKD_Brury/11.13/CKD-TransBTS-main/dataset/brats2021/...
    base = os.path.join(dataset_folder, "dataset", "brats2020")

    if mode == "train":
        return os.path.join(base, "train")
    elif mode == "train_val":
        return os.path.join(base, "val")
    elif mode == "test":
        return os.path.join(base, "test")
