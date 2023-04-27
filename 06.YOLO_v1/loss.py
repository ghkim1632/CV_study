import torch
import torch.nn as nn
from utils import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self, s=7, b=2, c=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        
        self.s = s
        self.b = b
        self.c = c
        self.lambda_noobj = 0.5 # no object에 대한 하이퍼파라미터
        self.lambda_coord = 5 # coordinate에 대한 하이퍼파라미터
    
    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.s, self.b, self.c+self.b*5) # 7x7x20 feature map flatten
        
        # 예측한 2개의 bounding box의 iou계산. iou값이 더 큰 박스를 선택하기 위해.
        # 첫번째 bounding box와 target과 iou 계산 
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25]) # [..., 21:25] 첫번째 bounding box, *class개수 이후
        # 두번째 bounding box와 target과 iou 계산
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25]) # [..., 26:30] 두번째 bounding box
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        iou_maxes, bestbox = torch.max(ious, dim=0) # bounding box 두개 중 더 큰 IoU를 가진 박스. -> (value, index)
        exists_box = target[..., 10].unsqueeze(3)  # 해당 grid cell에 ground-truth가 존재하는지 여부 (1 : 존재, 0 : 존재x)


        # ======================== #
        #     Localization Loss    #
        # ======================== #

        # 각 그리드셀의 각 박스가 object를 포함하고 있을 때 중심점/w, h의 regression loss 계산 

        # box_predictions : IoU 더 큰 값의 bounding box
        # ground-truth가 존재할 때 iou가 더 큰 박스를 가져온다
        box_predictions = exists_box * ( 
            (
                bestbox * predictions[..., 26:30]         # IoU가 더 큰 박스가 두번째 박스일 때
                + (1 - bestbox) * predictions[..., 21:25] # IoU가 더 큰 박스가 첫번째 박스일 때
            )
        )

        box_targets = exists_box * target[..., 21:25] 

        # width, height 루트 씌우기
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        # MSE loss
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # ======================== #
        #      Confidence Loss     #
        # ======================== #
        
        # confidence loss는 object가 있을 때, 없을 때 나눠서 계산 (exists_box: object 존재 유무)

        ### For Object Loss ###
        
        # pred_box : IoU가 큰 box의 confidence score
        pred_box = (
            bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21]
        )
        # MSE Loss
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21]),
        )
        
        ### For No Object Loss ###
        # object가 없을 때는 두개의 bounding box 모두 계산

        # 첫번째 bounding box의 MSE loss
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )
        # 두번째 bounding box의 MSE loss
        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        # ======================== #
        #    Classification Loss   #
        # ======================== #

        # MSE loss
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2,), # 박스가 존재할 때 prediction
            torch.flatten(exists_box * target[..., :20], end_dim=-2,), # 박스가 존재할 때 target
        )

        # ======================== #
        #         Final Loss       #
        # ======================== #

        loss = (
            self.lambda_coord * box_loss           # localization loss
            + object_loss                          # confidence loss (object 있을 때)
            + self.lambda_noobj * no_object_loss   # confidence loss (object 없을 때)
            + class_loss                           # classification loss
        )

        return loss
