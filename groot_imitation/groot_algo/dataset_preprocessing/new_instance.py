from groot_imitation.groot_algo.dino_features import DinoV2ImageProcessor
from groot_imitation.groot_algo.sam_operator import SAMOperator
from groot_imitation.segmentation_correspondence_model.scm import SegmentationCorrespondenceModel
from groot_imitation.groot_algo.misc_utils import resize_image_to_same_shape, get_palette
import numpy as np
import cv2
from PIL import Image

class NewInstance:
    def __init__(self, cfg, first_frame, first_frame_annotation):
        self.dinov2 = DinoV2ImageProcessor(cfg=cfg)
        self.sam_operator = SAMOperator(checkpoint=cfg.sam_checkpoint, device=cfg.device, sam_config_file=cfg.sam_config_file)
        self.sam_operator.init()
        self.scm_module = SegmentationCorrespondenceModel(dinov2=self.dinov2, sam_operator=self.sam_operator)

        self.first_frame = first_frame.copy()
        self.first_frame_annotation = first_frame_annotation.copy()
        
        self.xmem_input_size = first_frame.shape[:2]

    def get_new_instance_annotation(self, new_first_frame):
        first_frame = resize_image_to_same_shape(self.first_frame, new_first_frame)
        first_frame_annotation = resize_image_to_same_shape(self.first_frame_annotation, first_frame)
        #Save first_frame_annotation which is a mask
        new_first_frame_annotation = self.scm_module(new_first_frame, first_frame, first_frame_annotation)
        new_first_frame_annotation = resize_image_to_same_shape(new_first_frame_annotation, new_first_frame)
        new_first_frame_annotation[first_frame_annotation == first_frame_annotation.max()] = first_frame_annotation.max()
        new_first_frame = resize_image_to_same_shape(new_first_frame, reference_size=self.xmem_input_size)
        new_first_frame_annotation = resize_image_to_same_shape(np.array(new_first_frame_annotation), reference_size=self.xmem_input_size)
        return new_first_frame_annotation