"""
@author: yuchuang,zhaojinmiao
@time:
@desc: 直接可以用“GGLNet_LESPS_coarse_300.pth.tar”权重进行推理
"""
import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
from net import Net
from dataset import *
from metrics_mm import *
import os
from tqdm import tqdm
import numpy as np
import cv2
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_gaussian, create_pairwise_bilateral

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser(description="PyTorch LESPS test")
parser.add_argument("--model_names", default=['GGLNet'], nargs='+',
                    help="model_name: 'ACM', 'ALCNet', 'DNANet','MLCLNet_small','MLCLNet_base','ALCLNet','GGLNet'")
parser.add_argument("--pth_dirs", default=['./GGLNet_LESPS_coarse_300.pth.tar', ],
                    nargs='+', help="checkpoint dir, default=None")
parser.add_argument("--dataset_names", default=['WP_IRSTD_9000'], nargs='+',
                    help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'SIRST3', 'NUDT-SIRST-Sea'")
parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--img_norm_cfg_mean", default=64.181, type=float,
                    help="specific a mean value img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--img_norm_cfg_std", default=24.502, type=float,
                    help="specific a std value img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")

parser.add_argument("--dataset_dir", default='./datasets/', type=str, help="train_dataset_dir")
parser.add_argument("--save_img", default=True, type=bool, help="save image of or not")
parser.add_argument("--save_img_dir", type=str, default='./results/', help="path of saved image")
parser.add_argument("--save_log", type=str, default='./log/', help="path of saved .pth")
parser.add_argument("--threshold", type=float, default=0.01)
parser.add_argument("--threshold_2", type=float, default=0.15)
parser.add_argument("--patchSize", type=int, default=256, help="Training patch size, default: 512")
parser.add_argument("--test_batch_size", type=int, default=32, help="Training patch size, default: 16")

global opt
opt = parser.parse_args()
## Set img_norm_cfg
if opt.img_norm_cfg_mean != None and opt.img_norm_cfg_std != None:
    opt.img_norm_cfg = dict()
    opt.img_norm_cfg['mean'] = opt.img_norm_cfg_mean
    opt.img_norm_cfg['std'] = opt.img_norm_cfg_std

###可调节敏感度策略 (Adjustable sensitivity strategy)
def target_PD(copy_mask, target_mask):
    copy_contours, _ = cv2.findContours(copy_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    target_contours, _ = cv2.findContours(target_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    overwrite_contours = []
    un_overwrite_contours = []

    target_index_sets = []
    for target_contour in target_contours:
        target_contour_mask = np.zeros(copy_mask.shape, np.uint8)
        cv2.fillPoly(target_contour_mask, [target_contour], (255))
        target_index = np.where(target_contour_mask == 255)
        target_index_XmergeY = set(target_index[0] * 1.0 + target_index[1] * 0.0001)
        target_index_sets.append(target_index_XmergeY)

    for copy_contour in copy_contours:
        copy_contour_mask = np.zeros(copy_mask.shape, np.uint8)
        cv2.fillPoly(copy_contour_mask, [copy_contour], (255))
        copy_index = np.where(copy_contour_mask == 255)
        copy_index_XmergeY = set(copy_index[0] * 1.0 + copy_index[1] * 0.0001)

        overlap_found = False
        for target_index_XmergeY in target_index_sets:
            if not copy_index_XmergeY.isdisjoint(target_index_XmergeY):
                overwrite_contours.append(copy_contour)
                overlap_found = True
                break

        if not overlap_found:
            un_overwrite_contours.append(copy_contour)

    for un_overwrite_c in un_overwrite_contours:
        temp_contour_mask = np.zeros(target_mask.shape, np.uint8)
        cv2.fillPoly(temp_contour_mask, [un_overwrite_c], (255))
        temp_mask = measure.label(temp_contour_mask, connectivity=2)
        coord_image = measure.regionprops(temp_mask)
        (y, x) = coord_image[0].centroid
        target_mask[int(y), int(x)] = 255

    return target_mask


def convert_prob_map(single_channel_prob_map):
    foreground_prob = single_channel_prob_map[0]
    background_prob = 1 - foreground_prob
    double_channel_prob_map = np.stack([background_prob, foreground_prob], axis=0)
    return double_channel_prob_map


def apply_dense_crf_binary(image, prob_map, sdims_gaussian=(1, 1), compat_gaussian=3, sdims_bilateral=(20, 20),
                           compat_bilateral=10, srgb=(5, 5, 5)):
    num_classes = 2
    h, w = image.shape[:2]

    if image.shape[2] != 3:
        raise ValueError("Image must have 3 channels (H, W, 3)")
    d = dcrf.DenseCRF2D(w, h, num_classes)
    unary = unary_from_softmax(prob_map)
    d.setUnaryEnergy(unary)
    feats_gaussian = create_pairwise_gaussian(sdims=sdims_gaussian, shape=(h, w))
    d.addPairwiseEnergy(feats_gaussian, compat=compat_gaussian, kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
    feats_bilateral = create_pairwise_bilateral(sdims=sdims_bilateral, schan=srgb, img=image, chdim=2)
    d.addPairwiseEnergy(feats_bilateral, compat=compat_bilateral, kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(1)
    refined_prob_map = np.array(Q).reshape((num_classes, h, w))
    return refined_prob_map


def test_pred(img, net, batch_size):
    b, c, h, w = img.shape
    patch_size = opt.patchSize
    stride = opt.patchSize

    if h > patch_size and w > patch_size:
        img_unfold = F.unfold(img, kernel_size=patch_size, stride=stride)
        img_unfold = img_unfold.reshape(b, c, patch_size, patch_size, -1).permute(0, 4, 1, 2, 3)
        patch_num = img_unfold.size(1)

        preds_list = []
        for i in range(0, patch_num, batch_size):
            end = min(i + batch_size, patch_num)
            batch_patches = img_unfold[:, i:end, :, :, :].reshape(-1, c, patch_size, patch_size)
            batch_patches = Variable(batch_patches.float())
            batch_preds = net.forward(batch_patches)
            preds_list.append(batch_preds)
        preds_unfold = torch.cat(preds_list, dim=0).permute(1, 2, 3, 0)
        preds_unfold = preds_unfold.reshape(b, -1, patch_num)
        preds = F.fold(preds_unfold, kernel_size=patch_size, stride=stride, output_size=(h, w))
    else:
        preds = net.forward(img)

    return preds


def test():
    test_set = InferenceSetLoader(opt.dataset_dir, opt.train_dataset_name, opt.test_dataset_name, opt.img_norm_cfg)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)

    net = Net(model_name=opt.model_name, mode='test').cuda()
    net.load_state_dict(torch.load(opt.pth_dir)['state_dict'])
    net.eval()

    eval_mIoU = mIoU()
    eval_PD_FA = PD_FA()
    with torch.no_grad():
        for idx_iter, (img, size, img_dir) in tqdm(enumerate(test_loader)):
            img = Variable(img).cuda()

            image_1 = img
            image_2 = torch.flip(img, [2])
            image_3 = torch.flip(img, [3])

            pred_1 = test_pred(image_1, net, batch_size=opt.test_batch_size)
            img_out_1 = pred_1.cpu().data.numpy()
            img_out_1 = img_out_1[0, :, :size[0], :size[1]]

            pred_2 = test_pred(image_2, net, batch_size=opt.test_batch_size)
            img_out_2 = torch.flip(pred_2, [2]).cpu().data.numpy()
            img_out_2 = img_out_2[0, :, :size[0], :size[1]]

            pred_3 = test_pred(image_3, net, batch_size=opt.test_batch_size)
            img_out_3 = torch.flip(pred_3, [3]).cpu().data.numpy()
            img_out_3 = img_out_3[0, :, :size[0], :size[1]]

            img_np = (img_out_1 + img_out_2 + img_out_3) / 3

            if opt.save_img == True:
                pred_fg = np.expand_dims(img_np, axis=0)
                prob_map = convert_prob_map(pred_fg)
                img_oriain_singal = np.array(
                    Image.open(opt.dataset_dir + opt.dataset_names[0] + '/images/' + img_dir[0] + '.png').convert(
                        "RGB"))
                refined_prob_map = apply_dense_crf_binary(img_oriain_singal, prob_map)
                refined_segmentation = refined_prob_map[1]
                refined_segmentation = np.where(refined_segmentation >= opt.threshold, 255, 0).astype(np.uint8)
                pred_copy_0 = np.where(img_np >= opt.threshold_2, 255, 0).astype(np.uint8)
                pred_copy_0 = np.squeeze(pred_copy_0, axis=0)
                binary_np_0 = target_PD(pred_copy_0, refined_segmentation)
                binary_np = binary_np_0
                print(np.max(binary_np))
                if binary_np.ndim == 3 and binary_np.shape[0] == 1:
                    binary_np = binary_np[0]
                img_save = Image.fromarray(binary_np)
                save_dir = os.path.join(os.getcwd(), 'mask')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                img_save_path = os.path.join(save_dir, img_dir[0] + '.png')
                img_save.save(img_save_path)


if __name__ == '__main__':
    for pth_dir in opt.pth_dirs:
        opt.train_dataset_name = pth_dir.split('/')[0]
        print(pth_dir)
        for dataset_name in opt.dataset_names:
            opt.test_dataset_name = dataset_name
            opt.pth_dir = pth_dir
            print(opt.test_dataset_name)
            for model_name in opt.model_names:
                if model_name in pth_dir:
                    opt.model_name = model_name
            test()
        print('\n')
