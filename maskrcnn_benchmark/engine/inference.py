# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os

import torch
from tqdm import tqdm

from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from .bbox_aug import im_detect_bbox_aug
import ipdb
import cv2
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
import glob
import numpy as np
from maskrcnn_benchmark.config.paths_catalog import DatasetCatalog


def compute_on_dataset(model, data_loader, device, bbox_aug, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        with torch.no_grad():
            if timer:
                timer.tic()
            if bbox_aug:
                output = im_detect_bbox_aug(model, images, device)
            else:
                output = model(images.to(device))
            if timer:
                if not device.type == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        bbox_aug=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):  
    # ipdb.set_trace()
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, device, bbox_aug, inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )
    # ipdb.set_trace()
    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)


def iou_cal(pred,gt):
    # ipdb.set_trace()
    if len(gt.shape) == 3:
        gt = gt[:,:,0]
    pred = pred[0,0]
    assert pred.shape == gt.shape
    ii = (pred*(gt>0.5)).sum()
    uu = pred.sum()+(gt>0.5).sum()
    iou = ii/(uu - ii)
    return iou*100


class Resize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = self.min_size
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        return image


def build_transform(cfg):
    """
    Creates a basic transformation that was used to train the models
    """
    # cfg = self.cfg

    # we are loading images with OpenCV, so we don't need to convert them
    # to BGR, they are already! So all we need to do is to normalize
    # by 255 if we want to convert to BGR255 format, or flip the channels
    # if we want it to be in RGB in [0-1] range.
    if cfg.INPUT.TO_BGR255:
        to_bgr_transform = T.Lambda(lambda x: x * 255)
    else:
        to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
    )
    min_size = cfg.INPUT.MIN_SIZE_TEST
    max_size = cfg.INPUT.MAX_SIZE_TEST
    transform = T.Compose(
        [
            T.ToPILImage(),
            Resize(min_size, max_size),
            T.ToTensor(),
            to_bgr_transform,
            normalize_transform,
        ]
    )
    return transform


def compute_prediction(cfg, model, transforms, original_image, masker):
    """
    Arguments:
        original_image (np.ndarray): an image as returned by OpenCV

    Returns:
        prediction (BoxList): the detected objects. Additional information
            of the detection properties can be found in the fields of
            the BoxList via `prediction.fields()`
    """
    # ipdb.set_trace()
    # apply pre-processing to image
    image = transforms(original_image)
    # convert to an ImageList, padded so that it is divisible by
    # cfg.DATALOADER.SIZE_DIVISIBILITY
    image_list = to_image_list(image, cfg.DATALOADER.SIZE_DIVISIBILITY)
    image_list = image_list.to(cfg.MODEL.DEVICE)
    # compute predictions
    with torch.no_grad():
        predictions = model(image_list)
    cpu_device = torch.device("cpu")
    predictions = [o.to(cpu_device) for o in predictions]

    # always single image is passed at a time
    prediction = predictions[0]

    # reshape prediction (a BoxList) into the original image size
    height, width = original_image.shape[:-1]
    prediction = prediction.resize((width, height))

    if prediction.has_field("mask"):
        # if we have masks, paste the masks in the right position
        # in the image, as defined by the bounding boxes
        masks = prediction.get_field("mask")
        # always single image is passed at a time
        masks = masker([masks], [prediction])[0]
        prediction.add_field("mask", masks)
    return prediction


def select_top1_predictions(confidence_threshold, predictions):
    """
    Select only predictions which have a `score` > self.confidence_threshold,
    and returns the predictions in descending order of score

    Arguments:
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores`.

    Returns:
        prediction (BoxList): the detected objects. Additional information
            of the detection properties can be found in the fields of
            the BoxList via `prediction.fields()`
    """
    scores = predictions.get_field("scores")
    keep = torch.nonzero(scores > confidence_threshold).squeeze(1)
    predictions = predictions[keep]
    scores = predictions.get_field("scores")
    _, idx = scores.sort(0, descending=True)
    if len(predictions) > 1:
        # ipdb.set_trace()
        return predictions[torch.tensor([0])]
    else:
        return predictions[idx]


def infer_iou(cfg,model,confidence_threshold=0.7):

    data_dir = DatasetCatalog.DATA_DIR
    dataname = cfg.DATASETS.TEST
    attrs = DatasetCatalog.DATASETS[dataname[0]]
    valimgfolder = os.path.join(data_dir, attrs["img_dir"])
    foldnum = valimgfolder.split('fold')[-1].split('_')[0]
    if foldnum in ['1','2','3']:
        gtmaskroot = '/mnt/data1/ZYDdata/dataset/movie111_cls/clean_for_convnext/'
        gtmaskfolder = gtmaskroot+'fold{}/val/inner'.format(foldnum)
    else:
        gtmaskfolder = '/mnt/data1/ZYDdata/dataset/for_unionmaskpred/foldall/train/mask'

    show_mask_heatmaps = False
    mask_threshold = -1 if show_mask_heatmaps else 0.5
    masker = Masker(threshold=mask_threshold, padding=1)

    transforms = build_transform(cfg)
    
    imgpathlist = glob.glob(os.path.join(valimgfolder,'**.jpg'))
    ioulist = []
    flaglist = []
    for imgpath in imgpathlist:
        image = cv2.imread(imgpath)
        predictions = compute_prediction(cfg,model,transforms,image,masker)
        top_predictions = select_top1_predictions(confidence_threshold,predictions)

        name = os.path.split(imgpath)[-1]
        maskname = name.replace('.jpg','.png')
        gtmask = cv2.imread(os.path.join(gtmaskfolder, maskname))[:,:,0]

        if len(top_predictions)>0:
            masks = top_predictions.get_field("mask").numpy()
            iou = iou_cal(masks, gtmask)
            flag = 1
        else:
            iou = 0
            flag = 0
        ioulist.append(iou)
        flaglist.append(flag)
    
    meaniou = np.array(ioulist).mean()
    meaniou_predbbox = np.array(ioulist)[np.where(np.array(flaglist) == 1)].mean()
    bbox_recall = 100*(np.array(flaglist)==1).sum() / len(flaglist) # 预测有病灶的
    bbox_pos = 100*(np.array(ioulist)> 0.1).sum() / len(flaglist) # 预测对病灶位置（near
    bbox_seg = 100*(np.array(ioulist)> 0.5).sum() / len(flaglist) # 预测对病灶区域

    return {
        'meaniou': meaniou,
        'meaniou_predbbox': meaniou_predbbox,
        'bbox_recall': bbox_recall,
        'bbox_pos': bbox_pos,
        'bbox_seg': bbox_seg}
    
