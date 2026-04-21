import argparse
import os
import numpy as np
import random
import torch
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from BraTS import get_datasets
from models.model import DeblurCKD
from models import DataAugmenter
from utils import (
    mkdir, save_best_model, save_seg_csv, cal_dice, cal_confuse,
    save_test_label, AverageMeter, save_checkpoint
)
from torch.backends import cudnn
from monai.metrics.hausdorff_distance import HausdorffDistanceMetric
from monai.metrics.meandice import DiceMetric
from monai.inferers import sliding_window_inference
import torch.nn as nn
import torch.nn.functional as F

# 去模糊重建损失的权重，保持和主实验一致
LAMBDA_REC = 0.1
class WeightedDiceLoss(torch.nn.Module):
    """
    三通道加权 DiceLoss：
    通道顺序默认是 [ET, TC, WT]，对应权重 class_weights = (w_et, w_tc, w_wt)
    loss = 1 - sum(w_c * dice_c) / sum(w_c)
    """
    def __init__(self, class_weights=(2.0, 1.0, 1.0), smooth=1e-5):
        super().__init__()
        w = torch.tensor(class_weights, dtype=torch.float32)
        self.register_buffer("class_weights", w)
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits:  [B, 3, D, H, W]，未过 sigmoid 的网络输出
        targets: [B, 3, D, H, W]，0/1 或 bool 的 ET/TC/WT one-hot
        """
        # 关键：全部对齐到 logits 的 device
        device = logits.device
        targets = targets.to(device)

        # 概率
        probs = torch.sigmoid(logits).float()
        targets = targets.float()

        B, C = probs.shape[0], probs.shape[1]

        # 展平成 [B, C, N]
        probs = probs.view(B, C, -1)
        targets = targets.view(B, C, -1)

        # 每个 batch、每个通道算 Dice
        intersection = (probs * targets).sum(dim=-1)          # [B, C]
        denom = probs.sum(dim=-1) + targets.sum(dim=-1)       # [B, C]
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)  # [B, C]

        # 先在 batch 维度上取平均，得到每个通道一个 dice
        dice_per_class = dice.mean(dim=0)                     # [C]

        # 权重也对齐到同一 device
        w = self.class_weights.to(device)
        w = w / (w.sum() + 1e-8)

        dice_weighted = (w * dice_per_class).sum()            # 标量
        loss = 1.0 - dice_weighted
        return loss


parser = argparse.ArgumentParser(description='BraTS')
parser.add_argument('--exp-name', default="CKD_etw2", type=str)
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--dataset-folder', default="", type=str,
                    help="Please reference the README file for the detailed dataset structure.")
parser.add_argument('--workers', default=1, type=int)
parser.add_argument('--end-epoch', default=50, type=int)
parser.add_argument('--batch-size', default=2, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--devices', default="0", type=str)
parser.add_argument('--resume', default=False, type=bool)  # 这里的 resume 只管 checkpoint，不影响从 CKD 加载
parser.add_argument('--tta', default=True, type=bool, help="test time augmentation")
parser.add_argument('--seed', default=1)
parser.add_argument('--val', default=1, type=int)


def init_randon(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def init_folder(args):
    args.base_folder = mkdir(os.path.dirname(os.path.realpath(__file__)))
    args.dataset_folder = mkdir(os.path.join(args.base_folder, args.dataset_folder))

    args.best_folder = mkdir(f"{args.base_folder}/best_model/{args.exp_name}")
    args.writer_folder = mkdir(f"{args.base_folder}/writer/{args.exp_name}")
    args.pred_folder = mkdir(f"{args.base_folder}/pred/{args.exp_name}")
    args.checkpoint_folder = mkdir(f"{args.base_folder}/checkpoint/{args.exp_name}")
    args.csv_folder = mkdir(f"{args.base_folder}/csv/{args.exp_name}")

    print(f"The code folder are located in {os.path.dirname(os.path.realpath(__file__))}")
    print(f"The dataset folder located in {args.dataset_folder}")


def load_state_from_ckd_main(model, device):
    """
    从主实验 CKD 的 best_model.pkl 初始化权重，用作微调起点
    """
    base_folder = os.path.dirname(os.path.realpath(__file__))
    ckd_best = os.path.join(base_folder, "best_model", "CKD", "best_model.pkl")
    assert os.path.exists(ckd_best), f"主实验 CKD 的 best_model 不存在: {ckd_best}"

    print(f"从主实验 CKD 加载初始化权重: {ckd_best}")
    state = torch.load(ckd_best, map_location=device)

    # 有些情况下可能存成 {'model': state_dict}
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]

    model_is_dp = isinstance(model, nn.DataParallel)
    has_module = False
    for k in state.keys():
        if isinstance(k, str) and k.startswith("module."):
            has_module = True
            break

    if model_is_dp and (not has_module):
        state = {"module." + k: v for k, v in state.items()}
    if (not model_is_dp) and has_module:
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"load_state_dict from CKD done, missing={len(missing)}, unexpected={len(unexpected)}")
    return model


def main(args):
    writer = SummaryWriter(args.writer_folder)

    device_ids = [int(x) for x in str(args.devices).split(',')]
    print("使用的 GPU:", device_ids)
    main_device = torch.device(f"cuda:{device_ids[0]}")
    args.main_device = main_device

    model = DeblurCKD(
        embed_dim=32,
        output_dim=3,
        img_size=(128, 128, 128),
        patch_size=(4, 4, 4),
        in_chans=1,
        depths=[2, 2, 2],
        num_heads=[2, 4, 8, 16],
        window_size=(7, 7, 7),
        mlp_ratio=4.,
        stem_in_channels=4,
        stem_mid_channels=16,
        stem_norm_type='instance'
    ).to(main_device)

    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)

    # 用加权 DiceLoss 替换原来的 DiceLoss
    criterion = WeightedDiceLoss(
        class_weights=(2.0, 1.0, 1.0)
    ).to(main_device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5, amsgrad=True)

    if args.mode == "train":
        # 1) 先从 CKD 主实验的 best_model 初始化
        model = load_state_from_ckd_main(model, main_device)

        # 2) 正常构建当前实验的数据集和 dataloader
        train_dataset = get_datasets(args.dataset_folder, "train")
        train_val_dataset = get_datasets(args.dataset_folder, "train_val")
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            drop_last=True
        )
        train_val_loader = torch.utils.data.DataLoader(
            train_val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers
        )

        # 3) 进入微调训练
        train_manager(args, train_loader, train_val_loader, model, criterion, optimizer, writer)

    elif args.mode == "test":
        print("start test")
        best_path = os.path.join(args.best_folder, "best_model.pkl")
        assert os.path.exists(best_path), f"best model not found: {best_path}"

        state = torch.load(best_path, map_location=main_device)
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]

        model_is_dp = isinstance(model, nn.DataParallel)
        has_module = any(isinstance(k, str) and k.startswith("module.") for k in state.keys())
        if model_is_dp and (not has_module):
            state = {"module." + k: v for k, v in state.items()}
        if (not model_is_dp) and has_module:
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

        model.load_state_dict(state, strict=False)
        model.eval()

        test_dataset = get_datasets(args.dataset_folder, "test")
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers
        )
        test(args, "test", test_loader, model, writer)


def train_manager(args, train_loader, train_val_loader, model, criterion, optimizer, writer):
    best_loss = np.inf
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.end_epoch,
        eta_min=1e-5
    )
    start_epoch = 0

    print(f"start train from epoch = {start_epoch}")

    for epoch in range(start_epoch, args.end_epoch):
        model.train()
        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)

        train_loss, train_rec_loss = train(train_loader, model, criterion, optimizer, scheduler, epoch, writer)

        if (epoch + 1) % args.val == 0:
            model.eval()
            with torch.no_grad():
                train_val_loss = train_val(train_val_loader, model, criterion, epoch, writer)
                if train_val_loss < best_loss:
                    best_loss = train_val_loss
                    save_best_model(args, model)
        else:
            train_val_loss = best_loss

        save_checkpoint(
            args,
            dict(
                epoch=epoch,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict()
            )
        )
        print(f"epoch = {epoch}, "
              f"train_loss = {train_loss:.6f}, "
              f"train_rec_loss = {train_rec_loss:.6f}, "
              f"train_val_loss = {train_val_loss:.6f}, "
              f"best_loss = {best_loss:.6f}")

    print("finish train epoch")


def train(data_loader, model, criterion, optimizer, scheduler, epoch, writer):
    train_loss_meter = AverageMeter('Loss', ':.4e')
    rec_loss_meter = AverageMeter('RecLoss', ':.4e')

    for i, data in enumerate(data_loader):
        # 这里不要再 empty_cache 了，避免显卡利用率乱跳
        label = data["label"].to(model.device if hasattr(model, "device") else data["label"].device)
        images = data["image"].cuda()
        clean = data["clean"].cuda()

        optimizer.zero_grad()

        seg_pred, v_deblur_hat = model(images, return_deblur=True)

        loss_seg = criterion(seg_pred, label)
        loss_rec = F.l1_loss(v_deblur_hat, clean)
        loss = loss_seg + LAMBDA_REC * loss_rec

        train_loss_meter.update(loss.item())
        rec_loss_meter.update(loss_rec.item())

        loss.backward()
        optimizer.step()

    scheduler.step()

    writer.add_scalar("loss/train_total", train_loss_meter.avg, epoch)
    writer.add_scalar("loss/train_rec", rec_loss_meter.avg, epoch)

    return train_loss_meter.avg, rec_loss_meter.avg


def train_val(data_loader, model, criterion, epoch, writer):
    train_val_loss_meter = AverageMeter('Loss', ':.4e')
    for i, data in enumerate(data_loader):
        label = data["label"].cuda()
        images = data["image"].cuda()
        pred = model(images)
        train_val_loss = criterion(pred, label)
        train_val_loss_meter.update(train_val_loss.item())
    writer.add_scalar("loss/train_val", train_val_loss_meter.avg, epoch)
    return train_val_loss_meter.avg


def inference(model, input, batch_size, overlap):
    def _compute(input_inner):
        return sliding_window_inference(
            inputs=input_inner,
            roi_size=(128, 128, 128),
            sw_batch_size=batch_size,
            predictor=model,
            overlap=overlap
        )
    return _compute(input)


def test(args, mode, data_loader, model, writer=None):
    metrics_dict = []
    haussdor = HausdorffDistanceMetric(include_background=True, percentile=95)
    meandice = DiceMetric(include_background=True)
    for i, data in enumerate(data_loader):
        patient_id = data["patient_id"][0]
        inputs = data["image"]
        targets = data["label"].cuda()
        pad_list = data["pad_list"]
        nonzero_indexes = data["nonzero_indexes"]
        inputs = inputs.cuda()
        model.cuda()
        with torch.no_grad():
            if args.tta:
                predict = torch.sigmoid(inference(model, inputs, batch_size=2, overlap=0.6))
                p = torch.sigmoid(inference(model, inputs.flip(dims=(2,)), batch_size=2, overlap=0.6))
                predict += p.flip(dims=(2,))
                p = torch.sigmoid(inference(model, inputs.flip(dims=(3,)), batch_size=2, overlap=0.6))
                predict += p.flip(dims=(3,))
                p = torch.sigmoid(inference(model, inputs.flip(dims=(4,)), batch_size=2, overlap=0.6))
                predict += p.flip(dims=(4,))
                p = torch.sigmoid(inference(model, inputs.flip(dims=(2, 3)), batch_size=2, overlap=0.6))
                predict += p.flip(dims=(2, 3))
                p = torch.sigmoid(inference(model, inputs.flip(dims=(2, 4)), batch_size=2, overlap=0.6))
                predict += p.flip(dims=(2, 4))
                p = torch.sigmoid(inference(model, inputs.flip(dims=(3, 4)), batch_size=2, overlap=0.6))
                predict += p.flip(dims=(3, 4))
                p = torch.sigmoid(inference(model, inputs.flip(dims=(2, 3, 4)), batch_size=2, overlap=0.6))
                predict += p.flip(dims=(2, 3, 4))

                predict = predict / 8.0
            else:
                predict = torch.sigmoid(inference(model, inputs, batch_size=2, overlap=0.6))

        targets = targets[:, :,
                          pad_list[-4]:targets.shape[2] - pad_list[-3],
                          pad_list[-6]:targets.shape[3] - pad_list[-5],
                          pad_list[-8]:targets.shape[4] - pad_list[-7]]
        predict = predict[:, :,
                          pad_list[-4]:predict.shape[2] - pad_list[-3],
                          pad_list[-6]:predict.shape[3] - pad_list[-5],
                          pad_list[-8]:predict.shape[4] - pad_list[-7]]
        predict = (predict > 0.5).squeeze()
        targets = targets.squeeze()
        dice_metrics = cal_dice(predict, targets, haussdor, meandice)
        confuse_metric = cal_confuse(predict, targets, patient_id)
        et_dice, tc_dice, wt_dice = dice_metrics[0], dice_metrics[1], dice_metrics[2]
        et_hd, tc_hd, wt_hd = dice_metrics[3], dice_metrics[4], dice_metrics[5]
        et_sens, tc_sens, wt_sens = confuse_metric[0][0], confuse_metric[1][0], confuse_metric[2][0]
        et_spec, tc_spec, wt_spec = confuse_metric[0][1], confuse_metric[1][1], confuse_metric[2][1]
        metrics_dict.append(dict(
            id=patient_id,
            et_dice=et_dice, tc_dice=tc_dice, wt_dice=wt_dice,
            et_hd=et_hd, tc_hd=tc_hd, wt_hd=wt_hd,
            et_sens=et_sens, tc_sens=tc_sens, wt_sens=wt_sens,
            et_spec=et_spec, tc_spec=tc_spec, wt_spec=wt_spec
        ))
        full_predict = np.zeros((155, 240, 240))
        predict = reconstruct_label(predict)
        full_predict[slice(*nonzero_indexes[0]),
                     slice(*nonzero_indexes[1]),
                     slice(*nonzero_indexes[2])] = predict
        save_test_label(args, mode, patient_id, full_predict)
    save_seg_csv(args, mode, metrics_dict)


def reconstruct_label(image):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    c1, c2, c3 = image[0], image[1], image[2]
    image = (c3 > 0).astype(np.uint8)
    image[(c2 == False) * (c3 == True)] = 2
    image[(c1 == True) * (c3 == True)] = 4
    return image


if __name__ == "__main__":
    args = parser.parse_args()
    for arg in vars(args):
        print(format(arg, "<20"), format(str(getattr(args, arg)), "<"))

    if torch.cuda.device_count() == 0:
        raise RuntimeWarning("Can not run without GPUs")

    init_randon(args.seed)
    init_folder(args)

    device_ids = [int(x) for x in str(args.devices).split(",")]
    torch.cuda.set_device(device_ids[0])

    main(args)
