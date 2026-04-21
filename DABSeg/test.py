import argparse
import os
import numpy as np
import random
import torch
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from BraTS_S2 import get_datasets

# 修改1：这里从 CKD 换成 DeblurCKD（里面已经包含 DeblurStem + CKD）
from models.model import DeblurCKD

from models import DataAugmenter
from utils import (
    mkdir, save_best_model, save_seg_csv, cal_dice, cal_confuse,
    save_test_label, AverageMeter, save_checkpoint
)
from torch.backends import cudnn
from monai.metrics.hausdorff_distance import HausdorffDistanceMetric
from monai.metrics.meandice import DiceMetric
from monai.losses.dice import DiceLoss
from monai.inferers import sliding_window_inference
import torch.nn as nn

# 用于从真实目录拿 test case id
from get_dataset_folder import get_brats_folder


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in ("yes", "true", "t", "1", "y"):
        return True
    if v in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser(description='BraTS')
parser.add_argument('--exp-name', default="CKD_lrec02", type=str)
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--dataset-folder', default="", type=str,
                    help="Please reference the README file for the detailed dataset structure.")
parser.add_argument('--workers', default=1, type=int, help="The value of CPU's num_worker")
parser.add_argument('--end-epoch', default=500, type=int, help="Maximum iterations of the model")
parser.add_argument('--batch-size', default=2, type=int)
parser.add_argument('--lr', default=1e-4, type=float)

# devices 必须是 str，支持 "0,1"
parser.add_argument('--devices', default="0", type=str)

# bool 参数用 str2bool，确保 --tta False 生效
parser.add_argument('--resume', default=False, type=str2bool)
parser.add_argument('--tta', default=True, type=str2bool, help="test time augmentation")

parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--val', default=1, type=int, help="Validation frequency of the model")


def extract_patient_id(data):
    pid = data.get("patient_id", None)

    if isinstance(pid, str):
        return pid

    try:
        import numpy as _np
        if isinstance(pid, _np.ndarray):
            pid = pid.tolist()
    except Exception:
        pass

    if isinstance(pid, (list, tuple)):
        if len(pid) == 1 and isinstance(pid[0], str):
            return pid[0]

        # 字符列表 -> 拼回字符串
        if all(isinstance(x, str) for x in pid):
            if all(len(x) == 1 for x in pid):
                return "".join(pid)
            return pid[0]

        if len(pid) > 0:
            return extract_patient_id({"patient_id": pid[0]})

    return str(pid)


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

    # 给 utils.save_test_label 用
    args.label_folder = mkdir(f"{args.base_folder}/label/{args.exp_name}")

    print(f"The code folder are located in {os.path.dirname(os.path.realpath(__file__))}")
    print(f"The dataset folder located in {args.dataset_folder}")


def load_model_safely(model, ckpt_path, device):
    """
    兼容两类权重结构：
    1) best_model.pkl: OrderedDict(state_dict)
    2) checkpoint.pth.tar: dict，权重在 ckpt["model"]
    自动处理单卡/多卡的 module. 前缀
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    state = ckpt
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            state = ckpt["model"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]

    model_is_dp = isinstance(model, nn.DataParallel)

    state_has_module = False
    if isinstance(state, dict):
        for k in state.keys():
            if isinstance(k, str) and k.startswith("module."):
                state_has_module = True
                break

    if isinstance(state, dict):
        if state_has_module and not model_is_dp:
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        elif (not state_has_module) and model_is_dp:
            state = {"module." + k: v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"weight loaded from: {ckpt_path}")
    print(f"load_state_dict done, missing={len(missing)}, unexpected={len(unexpected)}")

    return model, ckpt


def resolve_checkpoint_path(args):
    p1 = os.path.join(args.checkpoint_folder, "checkpoint.pth.tar")
    p2 = os.path.join(args.checkpoint_folder, "checkpont.pth.tar")
    if os.path.exists(p1):
        return p1
    if os.path.exists(p2):
        return p2
    return p1


# 修改2：这里构建的是 DeblurCKD（前面有 DeblurStem，后面是 CKD）
def build_model(main_device, device_ids):
    """
    和训练用的 main.py 保持一致：
    model = DeblurCKD(
        embed_dim=32, output_dim=3, img_size=(128,128,128),
        patch_size=(4,4,4), in_chans=1, depths=[2,2,2],
        num_heads=[2,4,8,16], window_size=(7,7,7),
        mlp_ratio=4., stem_in_channels=4, stem_mid_channels=16,
        stem_norm_type='instance'
    )
    """
    model = DeblurCKD(
        embed_dim=32,
        output_dim=3,
        img_size=(128, 128, 128),
        patch_size=(4, 4, 4),
        in_chans=1,
        depths=[2, 2, 2],
        num_heads=[2, 4, 8, 16],
        window_size=(7, 7, 7),
        mlp_ratio=4.0,
        stem_in_channels=4,
        stem_mid_channels=16,
        stem_norm_type='instance',
    ).to(main_device)

    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)

    return model


def main(args):
    writer = SummaryWriter(args.writer_folder)

    device_ids = [int(x) for x in str(args.devices).split(',')]
    print("使用的 GPU:", device_ids)

    main_device = torch.device(f"cuda:{device_ids[0]}")
    args.main_device = main_device

    model = build_model(main_device, device_ids)
    criterion = DiceLoss(sigmoid=True).to(main_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5, amsgrad=True)

    if args.mode == "train":
        # 虽然现在主要用 test，但这里逻辑保持不变
        train_dataset = get_datasets(args.dataset_folder, "train")
        train_val_dataset = get_datasets(args.dataset_folder, "train_val")

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, drop_last=True
        )
        train_val_loader = torch.utils.data.DataLoader(
            train_val_dataset, batch_size=1, shuffle=False, num_workers=args.workers
        )

        train_manager(args, train_loader, train_val_loader, model, criterion, optimizer, writer)

    elif args.mode == "test":
        print("start test")

        # 只用 best_model.pkl，不再管 checkpoint
        best_path = os.path.join(args.best_folder, "best_model.pkl")
        if not os.path.exists(best_path):
            raise FileNotFoundError(f"best model not found: {best_path}")

        # 这里加载的就是你训练 50 轮后保存的 DeblurCKD 权重，
        # 里面包含了 DeblurStem 的参数，因此推理时真正走的是：
        #   S2 模糊四模态 -> DeblurStem3D -> CKD -> 分割输出
        model, _ = load_model_safely(model, best_path, main_device)
        model.eval()

        test_dataset = get_datasets(args.dataset_folder, "test")
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=args.workers
        )

        test(args, "test", test_loader, model)


def train_manager(args, train_loader, train_val_loader, model, criterion, optimizer, writer):
    best_loss = np.inf
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.end_epoch, eta_min=1e-5
    )
    start_epoch = 0

    if args.resume:
        ckpt_path = resolve_checkpoint_path(args)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

        print(f"正在加载已有模型权重: {ckpt_path}")
        model, checkpoint = load_model_safely(model, ckpt_path, args.main_device)

        if isinstance(checkpoint, dict):
            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if "scheduler" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler"])
            if "epoch" in checkpoint:
                start_epoch = int(checkpoint["epoch"]) + 1

        print(f"已加载 epoch = {start_epoch - 1} 的权重，将从 epoch = {start_epoch} 开始继续训练。")

    print(f"start train from epoch = {start_epoch}")

    for epoch in range(start_epoch, args.end_epoch):
        model.train()
        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)

        train_loss = train(args, train_loader, model, criterion, optimizer, scheduler, epoch, writer)

        train_val_loss = np.nan
        if (epoch + 1) % args.val == 0:
            model.eval()
            with torch.no_grad():
                train_val_loss = train_val(args, train_val_loader, model, criterion, epoch, writer)
                if train_val_loss < best_loss:
                    best_loss = train_val_loss
                    save_best_model(args, model)

        save_checkpoint(args, dict(
            epoch=epoch,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict()
        ))

        print(f"epoch = {epoch}, train_loss = {train_loss}, train_val_loss = {train_val_loss}, best_loss = {best_loss}")

    print("finish train epoch")


def train(args, data_loader, model, criterion, optimizer, scheduler, epoch, writer):
    train_loss_meter = AverageMeter('Loss', ':.4e')

    for i, data in enumerate(data_loader):
        torch.cuda.empty_cache()

        data_aug = DataAugmenter().to(args.main_device)

        label = data["label"].to(args.main_device)
        images = data["image"].to(args.main_device)

        images, label = data_aug(images, label)

        pred = model(images)
        loss = criterion(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_meter.update(loss.item())

    scheduler.step()
    torch.cuda.empty_cache()

    writer.add_scalar("loss/train", train_loss_meter.avg, epoch)
    return train_loss_meter.avg


def train_val(args, data_loader, model, criterion, epoch, writer):
    train_val_loss_meter = AverageMeter('Loss', ':.4e')

    for i, data in enumerate(data_loader):
        label = data["label"].to(args.main_device)
        images = data["image"].to(args.main_device)

        pred = model(images)
        loss = criterion(pred, label)
        train_val_loss_meter.update(loss.item())

    writer.add_scalar("loss/train_val", train_val_loss_meter.avg, epoch)
    return train_val_loss_meter.avg


def inference(model, input, batch_size, overlap):
    def _compute(inp):
        return sliding_window_inference(
            inputs=inp, roi_size=(128, 128, 128),
            sw_batch_size=batch_size, predictor=model, overlap=overlap
        )
    return _compute(input)


# 从真实 test 目录读取病人 id 列表
def get_test_case_ids(args):
    # 优先使用项目根作为 dataset_folder
    root = get_brats_folder(args.base_folder, "test")

    # 如果目录不存在，兜底用 args.dataset_folder
    if not os.path.isdir(root):
        root = get_brats_folder(args.dataset_folder, "test")

    ids = []
    try:
        for d in os.listdir(root):
            p = os.path.join(root, d)
            if os.path.isdir(p):
                ids.append(d)
    except Exception:
        pass

    ids.sort()
    return ids


def test(args, mode, data_loader, model):
    metrics_dict = []
    haussdor = HausdorffDistanceMetric(include_background=True, percentile=95)
    meandice = DiceMetric(include_background=True)

    if not hasattr(args, "label_folder"):
        args.label_folder = mkdir(os.path.join(args.base_folder, "label", args.exp_name))

    if not hasattr(args, "_test_case_ids"):
        args._test_case_ids = get_test_case_ids(args)

    for i, data in enumerate(data_loader):
        patient_id = extract_patient_id(data)

        if (not isinstance(patient_id, str)) or len(patient_id) <= 1:
            if i < len(args._test_case_ids):
                patient_id = args._test_case_ids[i]
            else:
                patient_id = f"case_{i:04d}"

        inputs = data["image"].to(args.main_device)
        targets = data["label"].to(args.main_device)
        pad_list = data["pad_list"]
        nonzero_indexes = data["nonzero_indexes"]

        with torch.no_grad():
            if args.tta:
                predict = torch.sigmoid(inference(model, inputs, batch_size=2, overlap=0.6))

                p = torch.sigmoid(inference(model, inputs.flip(dims=(2,)), batch_size=2, overlap=0.6)).flip(dims=(2,))
                predict += p
                p = torch.sigmoid(inference(model, inputs.flip(dims=(3,)), batch_size=2, overlap=0.6)).flip(dims=(3,))
                predict += p
                p = torch.sigmoid(inference(model, inputs.flip(dims=(4,)), batch_size=2, overlap=0.6)).flip(dims=(4,))
                predict += p
                p = torch.sigmoid(inference(model, inputs.flip(dims=(2, 3)), batch_size=2, overlap=0.6)).flip(dims=(2, 3))
                predict += p
                p = torch.sigmoid(inference(model, inputs.flip(dims=(2, 4)), batch_size=2, overlap=0.6)).flip(dims=(2, 4))
                predict += p
                p = torch.sigmoid(inference(model, inputs.flip(dims=(3, 4)), batch_size=2, overlap=0.6)).flip(dims=(3, 4))
                predict += p
                p = torch.sigmoid(inference(model, inputs.flip(dims=(2, 3, 4)), batch_size=2, overlap=0.6)).flip(dims=(2, 3, 4))
                predict += p

                predict = predict / 8.0
            else:
                predict = torch.sigmoid(inference(model, inputs, batch_size=2, overlap=0.6))

        targets = targets[:, :,
                          pad_list[-4]:targets.shape[2]-pad_list[-3],
                          pad_list[-6]:targets.shape[3]-pad_list[-5],
                          pad_list[-8]:targets.shape[4]-pad_list[-7]]
        predict = predict[:, :,
                          pad_list[-4]:predict.shape[2]-pad_list[-3],
                          pad_list[-6]:predict.shape[3]-pad_list[-5],
                          pad_list[-8]:predict.shape[4]-pad_list[-7]]

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
        predict_np = reconstruct_label(predict)
        full_predict[slice(*nonzero_indexes[0]),
                     slice(*nonzero_indexes[1]),
                     slice(*nonzero_indexes[2])] = predict_np

        save_test_label(args, patient_id, full_predict)

    save_seg_csv(args, mode, metrics_dict)


def reconstruct_label(image):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    c1, c2, c3 = image[0], image[1], image[2]
    out = (c3 > 0).astype(np.uint8)
    out[(c2 == False) * (c3 == True)] = 2
    out[(c1 == True) * (c3 == True)] = 4
    return out


if __name__ == '__main__':
    args = parser.parse_args()
    for arg in vars(args):
        print(format(arg, '<20'), format(str(getattr(args, arg)), '<'))

    if torch.cuda.device_count() == 0:
        raise RuntimeWarning("Can not run without GPUs")

    init_randon(args.seed)
    init_folder(args)

    device_ids = [int(x) for x in str(args.devices).split(',')]
    torch.cuda.set_device(device_ids[0])

    # 只在 test 模式补丁 utils.get_brats_folder
    if args.mode == "test":
        import utils as _utils
        from get_dataset_folder import get_brats_folder as _real_get_brats_folder

        def _patched_get_brats_folder(dataset_folder=None, mode=None, **kwargs):
            # 兼容 utils.py 内部的写法：get_brats_folder(mode="test")
            if mode is None:
                mode = kwargs.get("mode", "test")
            if dataset_folder is None:
                dataset_folder = args.base_folder
            return _real_get_brats_folder(dataset_folder, mode)

        _utils.get_brats_folder = _patched_get_brats_folder

    main(args)
