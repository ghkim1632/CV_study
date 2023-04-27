# 필요한 라이브러리 로드
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as PT
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import *
from model import *
from loss import *
from utils import *

# 시드 고정
seed = 123
torch.manual_seed(seed)

# 하이퍼파라미터 정의
learning_rate = 2e-5
device = 'cuda' if torch.cuda.is_available else 'cpu'
BATCH_SIZE = 16 # 논문에서는 64
weight_decay = 0
epochs = 100
NUM_WORKERS = 1
PIN_MEMORY = True
load_model = False
# load_model_file = 'overfit.pth.tar'
# img_dir = "./kaggle/input/pascal-voc-2012/voc2012/VOC2012/JPEGImages/"
# label_dir = "./kaggle/input/pascal-voc-2012/voc2012/VOC2012/Annotations/"


# transforms 정의
# albumentation을 이용해도 된다.
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes

# transoform init
transform = Compose([transforms.Resize(448, 448), transforms.ToTensor(),])



def train_fn(train_loader, model, optimizer, scheduler, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []
    
    for batch_idx, (x, y) in enumerate(loop):
        # x: image, y: bounding box
        x, y = x.to(device), y.to(device)
        
        # yolo 모델의 output
        out = model(x)
        
        # output과 ground truth를 이용해 loss 계산
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # progress bar 업데이트
        loop.set_postfix(loss=loss.item())
        
    return sum(mean_loss)/len(mean_loss)


def main():
    # model 생성
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(device)
    
    # opimizer 설정
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # loss function
    loss_fn = YoloLoss()
    
    # scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 70 ], gamma=0.1)


    # pretrained weight 사용할 시에 동작
    if load_model:
        load_checkpoint(torch.load(load_model_file), model, optimizer)
    
    # train dataset 생성
    tarin_dataset = CustomDataset(
        "./data/train.json",
        "./data",
        trainsforms=transform,
    )
    
    # test dataset 생성
    test_dataset = CustomDataset(
        "./data/test.json",
        "./data",
        transforms=transform,
    )
    
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True, drom_last=True,
    )
    
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True, drom_last=True,
    )
    
    low_loss = 1000
    for epoch in range(epochs):
        
        # model train
        loss = train_fn(train_loader, model, optimizer, scheduler, loss_fn)
        print(f"#{epoch+1}: Mean loss was {loss}")
        
        # checkpoint 저장
        if loss < low_loss:
            torch.save(model.state_dict(), './yolo_s.pth')
            low_loss = loss
        
        # 학습된 모델로 test dataset의 prediction box와 target box 생성.
        # 모델과 test loader로부터 box를 예측.그리드 단위의 박스를 원래 크기로 바꿈.
        pred_boxes, target_boxes = get_bboxes(
            test_loader, model, iou_threshold=0.5, threshold=0.4
        )
        
        # model이 얼마나 정확히 예측했는지 mAP 계산
        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        
        print(f"Train mAP: {mean_avg_prec}")
        
        # model train
        train_fn(train_loader, model, optimizer, loss_fn)
    
    
if __name__ = "__main__":
    main()