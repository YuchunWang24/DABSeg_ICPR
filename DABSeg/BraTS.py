import torch
import os
from torch.utils.data.dataset import Dataset
from get_dataset_folder import get_brats_folder
from utils import pad_or_crop_image, minmax, load_nii, pad_image_and_label, listdir

class BraTS(Dataset):
    def __init__(self, patients_dir, patient_ids, mode, clean_patients_dir=None):
        super(BraTS, self).__init__()
        self.patients_dir = patients_dir              # S2 模糊数据根目录
        self.clean_patients_dir = clean_patients_dir  # 修改：S0 干净数据根目录
        self.patients_ids = patient_ids
        self.mode = mode
        self.datas = []
        self.pattens = ["_t1", "_t1ce", "_t2", "_flair"]
        if mode == "train" or mode == "train_val" or mode == "test":
            self.pattens += ["_seg"]
        for patient_id in patient_ids:
            paths = [f"{patient_id}{patten}.nii.gz" for patten in self.pattens]
            patient = dict(
                id=patient_id,
                t1=paths[0], t1ce=paths[1],
                t2=paths[2], flair=paths[3],
                seg=paths[4] if mode in ["train", "train_val", "val", "test", "visualize"] else None
            )
            self.datas.append(patient)

    def __getitem__(self, idx):
        patient = self.datas[idx]
        patient_id = patient["id"]
        crop_list = []
        pad_list = []

        # 1) 读取 S2 模糊图像
        patient_image_blur_dict = {
            key: torch.tensor(load_nii(f"{self.patients_dir}/{patient_id}/{patient[key]}"))
            for key in patient if key not in ["id", "seg"]
        }
        patient_image_blur = torch.stack(
            [patient_image_blur_dict[key] for key in patient_image_blur_dict]
        )  # [4, D, H, W]

        # 2) 读取标签（直接用 S2 目录下的 seg）
        patient_label = torch.tensor(
            load_nii(f"{self.patients_dir}/{patient_id}/{patient['seg']}").astype("int8")
        )

        # 3) 若提供 clean_patients_dir，则同时读取 S0 干净图像
        patient_image_clean = None
        if self.clean_patients_dir is not None and (
            self.mode == "train" or self.mode == "train_val" or self.mode == "test"
        ):
            patient_image_clean_dict = {
                key: torch.tensor(load_nii(f"{self.clean_patients_dir}/{patient_id}/{patient[key]}"))
                for key in patient if key not in ["id", "seg"]
            }
            patient_image_clean = torch.stack(
                [patient_image_clean_dict[key] for key in patient_image_clean_dict]
            )  # [4, D, H, W]

        # 4) 标签转 ET/TC/WT
        if self.mode == "train" or self.mode == "train_val" or self.mode == "test":
            et = patient_label == 4
            tc = torch.logical_or(patient_label == 1, patient_label == 4)
            wt = torch.logical_or(tc, patient_label == 2)
            patient_label = torch.stack([et, tc, wt])  # [3, D, H, W]

        # 5) 去掉黑边（基于模糊图像计算 nonzero 区域）
        nonzero_index = torch.nonzero(torch.sum(patient_image_blur, axis=0) != 0)
        z_indexes, y_indexes, x_indexes = nonzero_index[:, 0], nonzero_index[:, 1], nonzero_index[:, 2]
        zmin, ymin, xmin = [max(0, int(torch.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
        zmax, ymax, xmax = [int(torch.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]

        patient_image_blur = patient_image_blur[:, zmin:zmax, ymin:ymax, xmin:xmax].float()
        if patient_image_clean is not None:
            patient_image_clean = patient_image_clean[:, zmin:zmax, ymin:ymax, xmin:xmax].float()
        patient_label = patient_label[:, zmin:zmax, ymin:ymax, xmin:xmax]

        # 6) min-max 归一化（对 blur/clean 都做）
        for i in range(patient_image_blur.shape[0]):
            patient_image_blur[i] = minmax(patient_image_blur[i])
            if patient_image_clean is not None:
                patient_image_clean[i] = minmax(patient_image_clean[i])

        # 7) 统一尺寸到 (128, 128, 128)
        if self.mode == "train" or self.mode == "train_val":
            if patient_image_clean is not None:
                # 为保证 S2/S0 对齐，分别对它们调用 pad_or_crop_image，
                # 因为原始尺寸一致，而且这个函数是确定性的，所以裁剪 / 填充方式会一致
                label_for_blur = patient_label
                label_for_clean = patient_label.clone()

                patient_image_blur, patient_label, pad_list, crop_list = pad_or_crop_image(
                    patient_image_blur, label_for_blur, target_size=(128, 128, 128)
                )
                patient_image_clean, _, _, _ = pad_or_crop_image(
                    patient_image_clean, label_for_clean, target_size=(128, 128, 128)
                )
            else:
                patient_image_blur, patient_label, pad_list, crop_list = pad_or_crop_image(
                    patient_image_blur, patient_label, target_size=(128, 128, 128)
                )

        elif self.mode == "test":
            d, h, w = patient_image_blur.shape[1:]
            pad_d = (128 - d) if 128 - d > 0 else 0
            pad_h = (128 - h) if 128 - h > 0 else 0
            pad_w = (128 - w) if 128 - w > 0 else 0
            patient_image_blur, patient_label, pad_list = pad_image_and_label(
                patient_image_blur, patient_label,
                target_size=(d + pad_d, h + pad_h, w + pad_w)
            )
            # test 阶段暂时不需要 S0 来算 loss，可以不 pad patient_image_clean

        return dict(
            patient_id=patient["id"][0],  # 保持你原来的写法
            image=patient_image_blur,     # S2 模糊图像
            label=patient_label,
            clean=patient_image_clean,    # 修改：新增 S0 干净图像（train / train_val 会用到）
            nonzero_indexes=((zmin, zmax), (ymin, ymax), (xmin, xmax)),
            box_slice=crop_list,
            pad_list=pad_list
        )

    def __len__(self):
        return len(self.datas)


def get_datasets(dataset_folder, mode):
    dataset_folder_blur = get_brats_folder(dataset_folder, mode)
    print("最终使用的模糊数据路径:", dataset_folder_blur)

    assert os.path.exists(dataset_folder_blur), "Dataset Folder Does Not Exist (blur)"

    # 修改：自动推 S0 路径，这里假设你用 brats2020_S0 作为 S0 目录名
    clean_folder = None
    if "brats2020" in dataset_folder_blur:
        clean_folder = dataset_folder_blur.replace("brats2020", "brats2020_S0")
        if not os.path.exists(clean_folder):
            print("警告: 未找到 S0 干净数据目录, clean_patients_dir = None, 只能训练分割不加 L_rec")
            clean_folder = None
        else:
            print("最终使用的干净数据路径:", clean_folder)

    patients_ids = [x for x in listdir(dataset_folder_blur)]
    return BraTS(dataset_folder_blur, patients_ids, mode, clean_patients_dir=clean_folder)
