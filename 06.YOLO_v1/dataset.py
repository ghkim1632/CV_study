import torch
import os
import pandas as pd
from PIL import Image
from pycocotools.coco import COCO
import numpy as np

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, annotation, data_dir, s=7, b=2, c=20, transforms=None):
        super().__init__()
        self.data_dir = data_dir
        # coco annotation 불러오기
        self.coco = COCO(annotation)
        self.predictions = {
            "images": self.coco.dataset['images'].copy(),
            "categories": self.coco.dataset["categories"].copy(),
            "annotations": None
        }
        
        # S x S grid 영역
        self.s = s
        # 각 그리드별 bounding box 개수
        self.b = b
        # class num
        self.c = c
        self.transforms = transforms
        
    def __len__(self):
        return len(self.coco.getImgIds())
    
    def __getitem__(self, index):
        # 이미지 아이디 가져오기
        image_id = self.coco.getImgIds(imgId=index)
        # 이미지 정보 가져오기
        image_info = self.coco.loadImgs(image_id)[0]
        # 이미지 로드
        img_path = os.path.join(self.data_dir, image_info['file_name'])
        image = Image.open(img_path)
        
        # 어노테이션 파일 로드
        ann_ids = self.coco.getAnnIds(imgIds=image_info['id'])
        anns = self.coco.loadAnns(ann_ids)
        
        # 박스 가져오기
        bbox = np.array([x['bbox'] for x in anns])
        
        # 레이블 가져오기
        labels = np.array([x['category_id'] for x in anns])
        
        # 박스 단위를 0~1로 조정
        boxes = []
        for box, label in zip(bbox, labels):
             boxes = []
        for box, label in zip(bbox, labels):
            boxes.append([label, box[0]/512, box[1]/512, box[2]/512, box[3]/512]) # *이미지 크기를 512로 resize한 상태에서.
                                
        boxes = torch.tensor(boxes)
    
        if self.transforms:
            # image = self.transforms(image)
            image,boxes = self.transforms(image, boxes)
            
        # 바운딩 박스를 그리드 단위(s x s)로 변환
        label_matrix = torch.zeros((self.s, self.s, self.c + 5 * self.b)) # 7 x 7 x (클래스 개수+5*박스 개수)
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)
            
            # x와 y가 어느 셀에 있는지 위치 계산 // i, j는 박스가 위치하는 row, column
            i, j = int(self.s * y), int(self.s * x)
            x_cell, y_cell = self.s * x - j, self.s * y - i
            
            # 높이, 너비도 그리드 기준으로 바꿈
            width_cell, height_cell = (width * self.s, height * self.s,)
            
            # 그리드 당 객체는 하나로 제한된다.
            if label_matrix[i, j, self.c] == 0: # 박스가 없을 때
                label_matrix[i, j, self.c] = 1 # 해당 그리드에 박스가 있다고 표시해준다.
                
                # 그리드 당 박스 좌표
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                
                label_matrix[i, j, self.c + 1:self.c + 5] = box_coordinates
                
                # class label을 one-hot encoding -> 1
                label_matrix[i, j, class_label] = 1
                
            return image, label_matrix