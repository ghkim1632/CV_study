# Faster R-CNN

## object detection 관련 개념  

### - object detection의 2가지 방식

1. **2-satge 방식**
    
    물체가 있을 법한 위치를 찾은 후(region proposal)에 각각의 위치에 클래스 부여(classfication, regression)
    
    regression: 이미지 내 사물이 존재하는 바운딩 박스를 예측하는 문제
    
    ex. R-CNN, Fast R-CNN, Faster R-CNN
    
2. **1-stage 방식**
    
    위치를 찾는 문제와 분류 문제를 한번에 해결
    
    ex. YOLO
  

### - **region proposal 방식**

- 물체가 있을 법한 위치를 찾는 방법
    
    ![Untitled](Faster%20R-CNN%2011cbd9cf8e85425592e9811431aad308/Untitled.png)


    - sliding window
    
        원본 이미지를 window 사이즈로 잘라내어 Localization network에 입력 후 이동을 반복하는 방식으로 물체의 위치 파악
    
        ex. Faster R-CNN
    
    ![Untitled](Faster%20R-CNN%2011cbd9cf8e85425592e9811431aad308/Untitled%201.png)
    
    - selective search
    
        인접한 영역끼리 유사성을 측정해서 큰 영역으로 통합된다.
    
        ex. R-CNN, Fast R-CNN
    

### 성능 평가 지표

- 두 바운딩 박스가 겹치는 비율인 iou(Intersection over Union)을 성능 평가 지표로 사용하는 경우가 많다.
- 예를 들어 
mAP@0.5이면 정답과 예측의 IoU가 50%이상일 때 정답으로 판정했다는 의미이다.

**NMS(Non Maximum Suppression)**

- 여러개의 바운딩 박스가 겹쳐있는 경우 하나로 합치는 방법으로, 제일 큰 것을 제외하고 나머지를 제거한다.

**RoI pooling**

![Untitled](Faster%20R-CNN%2011cbd9cf8e85425592e9811431aad308/Untitled%203.png)

- 고정된 크기의 feature vector를 찾기 위해서 임의로 나눈 RoI 영역에 대해서 max pooling 진행하는 방식이다.


---


## **Faster R-CNN**

Faster R-CNN은 물체가 있을 법한 위치를 찾고(region proposal) 각 위치의 클래스(classfication, regression)를 확인하는 2-stage 방식의 object detection 모델이다.

기존의 Fast R-CNN를 발전시킨 모델로, region proposal 단계에서 소요되는 시간을 단축하기 위해서 Region Proposal Network를 제안하였다.

![Untitled](Faster%20R-CNN%2011cbd9cf8e85425592e9811431aad308/Untitled%204.png)

## **Abstract**


region proposal 단계를 GPU에서 시행할 수 있게하는 RPN(Region proposal network) 제안하였다.

CNN으로 특징 추출하고 RPN에서 제안된 부분을 위주로 classification, regression을 진행한다.

논문에서는 RPN과 classifier를 번갈아가며 학습 진행하여 출력부터 입력까지 back propagation 가능한 end-to-end 모델을 구축하였다.

- full-image conv features를 공유
- simultaneously predicts object bounds, objectness scores at each point
- trained end-to-end. 전체가 GPU 상에서 이루어진다.
- 성능: VGG 모델에서는 5fps 처리

## **Introduction**

Region Proposal 방식 중 selective Search는 segmentation을 하는 것처럼 super pixel을 묶어서 위치를 찾는다. 이러한 방식은 CPU에서 이미지당 2초가 걸린다. Edge boxes를 이용하여 이미지당 0.2초로 이미지 처리 속도가 빨라지긴 했지만 여전히 region proposal step은 detection network만큼 많은 시간이 소요된다.

이는 region proposal 단계가 CPU에서 이루어지기 때문이다. Region proposal 속도를 개선시키기 위해서는 이 단계를 GPU에서 시행할 필요가 있는데, 문제는 GPU에 re-implement하면 down-stream detection network를 무시해서 연산을 공유할 수 없다는 것이다.

이 논문은 Region Proposal Networks(RPNs)과 Fast RCNN를 사용하여 conv layer 공유하여 연산을 줄이는 방식을 제안한다.

Faster R-CNN에서 주목할 부분들은 **RPN**, **anchor box**, **RPNs와 Faster R-CNN의 결합**이다.

- **RPN**
    
    RPN은 Deep nueral network에서 regison proposal을 진행하는 방식이다. Detection network의 연산에 비하면 region proposal 연산은 cost-free하다.
    
- **anchor box**
    
    Anchor box는 다양한 크기와 비율을 가진 bounding box이다.
    
- **RPNs와 Fast RCNN의 결합**
    
    Faster RCNN은 RPNs와 Fast RCNN을 일련의 과정으로 구성한다. RPN과 Fast RCNN을 연결하기 위해서 본 논문에서는 proposals를 고정한 채 RPN과 Fast RCNN을 번갈아가면서 fine-tuning한다. 그 결과 RPNs with Fast RCNN이 Selective Search with Fast RCNN보다 성능이나 연산량에 있어서 더 좋았다.
    

<details>
<summary>Related work</summary>
<div markdown="1">

1. **Object Proposals**
    
    region proposal 방법
    
    - grouping super-pixels ex. Selective Search
    - sliding window
    
    외부 모듈로 사용되어 detector와 분리됨
    
2. **Deep Networks for Object Detection**
    
    <u>**objectness score**</u>
    
    - R-CNN 방식은 CNN을 end-to-end로 학습
    - object category나 background로 region proposals 분류
    - 물체의 위치는 예측x
    - region proposal module에 정확성 의존
    
    <u>**object bound**</u>
    
    deep network를 object bounding box 예측에 사용하는 방법
    
    - **OverFeat method**
        - fc(fully conneted) layer가 box 좌표 예측(localization)을 위해 학습됨
        - fc layer는 conv layer가 되어 classs-specific object 탐지
    - **MultiBox method**
        - 마지막 fc layer가 다양한 box 예측
        - R-CNN의 proposal로 사용
        - single image crop이나 multiple large image crop에 사용
        - propowal과 detection net간 feature sharingX
    - **DeepMask method**
        - segmentation proposals 학습
</div>        
</details>

## **Faster R-CNN**

- **RPN**: fully conv net로 region proposal 진행. Fast R-CNN이 어디를 봐야 하는지=물체가 어디에 있는지 알려줌

- **Fast R-CNN**: RPN의 region proposal 이용.

**1. Region Proposal Networks**

![61AE479F-7FE2-42DD-9CD0-9D9E216DAAF0.jpeg](Faster%20R-CNN%2011cbd9cf8e85425592e9811431aad308/61AE479F-7FE2-42DD-9CD0-9D9E216DAAF0.jpeg)

RPN에서는 마지막 conv layer의 conv feature map output 위로 samll net를 sliding window 방식으로 연산한다. 각 위치에 대해 intermediate feature를 뽑고 classification과 regression을 진행한다. 여기서의 classification은 물체가 있는지 없는지(object/background)를 분류하는 문제이고, regression은 region proposal의 좌표를 구한다.

- input: 이미지 사이즈 상관x

- output: proposals + score 출력

Fast RCNN과 연산을 공유하기 위해(conv layer 공유) fully conv net를 만들었고, region proposal을 생성하기 위해 마지막 conv layer의 conv feature map output 위로 samll net를 슬라이드한다.


<u>**Anchors**</u>

RPN에서는 k개의 anchor box 이용한다. Sliding window 방식으로 region proposal을 진행할 때는 일반적으로 일정한 크기의 bounding box를 이용한다. 그러나 이 방식으로는 다양한 객체의 크기를 포착하지 못할 수 있다. 그래서 해당 논문에서는 size를 128, 256, 512 세 가지로 두고 이들을 각각 1:1, 1:2, 2:1 세 가지 비율로 된 anchor boxes를 참조하여 bouding box를 분류하고 찾는다.

이미지 크기 변경 and/or 비율 변경의 경우가 성능이 더 좋았다. translation-invariant(이미지를 이동하거나 다른 형태로 바뀌어도 변함x)한 특성을 바탕으로 모델 크기와 파라미터 개수를 줄여 PASCAL VOC와 같은 작은 데이터셋에서도 overfitting 위험을 줄일 수 있었다.

> multi scale anchor is a key component for sharing features  </br>k: a number of maximum possible proposals for each location.

<u>**multi-scale prediction의 방법**</u>

1. based on image/feature pyramids

    이미지 크기 변경하면서 스케일에 대해 feature map 계산

2. using sliding windows of multiple scale(and/or aspect ratios)

    이미지 크기 변경 + 비율 변경

    여러 크기와 비율의 anchor boxes를 참조하여 bouding box를 분류하고 찾는다.

<u>**Loss Function**</u>

각 anchor는 object, background로 classification된다. object로 분류하는 기준은 IoU값 최대일 때(모든 case에서 0.7이하일 때 대비) 또는  IoU가 0.7이상일 때이다. IoU가 0.3이하일 때는 background로 분류하였다.

![6E29C473-0C90-412B-AAFE-F0CF349DB49A.jpeg](Faster%20R-CNN%2011cbd9cf8e85425592e9811431aad308/6E29C473-0C90-412B-AAFE-F0CF349DB49A.jpeg)

+ cls loss: log loss(NLL). object vs not object

+ reg loss: smooth L1

    ![Untitled](Faster%20R-CNN%2011cbd9cf8e85425592e9811431aad308/Untitled%205.png)

    Loss function에서 '+' 뒤편의 식은 anchor가 positive일 때만 activated된다는 것을 보여준다.

    ![0AC21E8F-F86E-4332-8C59-40ECA16CC606.jpeg](Faster%20R-CNN%2011cbd9cf8e85425592e9811431aad308/0AC21E8F-F86E-4332-8C59-40ECA16CC606.jpeg)

    ```python
    ### Object or not loss
    rpn_cls_loss = F.cross_entropy(rpn_scores, gt_rpn_scores.long(), ignore_index=-1)
    print(rpn_cls_loss)

    ### location loss
    mask = gt_rpn_scores > 0
    mask_target_format_locs = gt_rpn_format_locs[mask]
    mask_pred_format_locs = rpn_format_locs[mask]

    print(mask_target_format_locs.shape)
    print(mask_pred_format_locs.shape)

    x = torch.abs(mask_target_format_locs - mask_pred_format_locs)
    rpn_loc_loss = ((x<0.5).float()*(x**2)*0.5 + (x>0.5).float()*(x-0.5)).sum()
    print(rpn_loc_loss)
    ```

<u>**Training RPN**</u>

backpropagation과 SGD로 end-to-end 학습 가능

256개 anchor만 학습. positive와 negative anchor의 비율 1:1 (positive가 128개 이하라면 negative로 부족한 부분 채움)

zero-mean Gaussian distribution 표준분산 0.01로 가중치 초기화

60k 미니배치 동안 learning rate 0.001, 다음 20k 미니배치동안 0.0001

SGD momentum 0.9, weight decay 5*1e-4

### **2. Sharing Features for RPN and Fast R-CNN**

RPN과 Fast RCNN의 conv layer 연결: feature map 공유
<u>**feature map을 공유하는 방식**</u>

1. Alternating training

RPN 학습 → proposals 이용해서 Fast R-CNN학습 → Fast R-CNN으로 RPN 초기화

: 논문에서 사용한 방식

2. Approximate joint training

RPN과 Fast R-CNN 결합하여 forward 때는 region proposal을 고정하고, backward propagation 때 RPN loss와 Fast R-CNN loss 결합하여 사용한다.

proposal box의 미분을 무시하기 때문에 '근사'라고 이름붙였다.

Alternating training과 결과는 비슷한데 training time을 25-50% 줄였다고 한다.

3. Non-approximate joint training


<u>**4-Step Alternating Training**</u>

1. RPN 학습

ImageNet으로 학습된 네트워크를 region proposal task로 fine tuned end-to-end

2. Fast R-CNN 학습

RPN으로 생성된 proposal 사용하여 FAST R-CNN 학습한다. ImageNet으로 학습된 네트워크를 사용하였고, 아직까진 conv layer를 공유하지 않은 상태이다.

3. Fast R-CNN으로 RPN 학습 초기화

공유된 conv layer는 고정시키고 RPN만 fine-tune한다.

4. 3의 과정과 반대
공유된 conv layer는 고정시키고 Fast RCNN unique한 부분만 fine-tune한다.

### **3. Implementation Details**

이미지의 짧는 면을 600px로 re-scale하여 사용하였다. anchor의 크기는 128, 256, 512으로 두었고, 각각을 1:1, 1:2, 2:1 비율을 가진 anchor로 만들었다. cross-boundary anchor는 학습 과정에서는 제외시키고 테스트할때는 사용하였다. RPN proposal이 너무 겹치는 경우를 방지하기 위해 cls score IoU 0.7 이상을 기준으로 non-maximum suppression(NMS) 적용하였다.


## Conclusion

RPNs는 효율적이고 정확한 region proposal generation이다. region proposal quality와 전체적인  object detection 정확도를 향상시킨다.

Faster R-CNN에서는 region proposal 과정을 수행하는 RPN의 conv features를 detection networt(Fast R-CNN)과 공유함으로써 cost-free를 이루어냄은 물론 거의 실시간 object detection이 가능하게 되었다.

---

### Reference

논문 전반 리뷰: 객체 검출(Object Detection) 딥러닝 기술: R-CNN, Fast R-CNN, Faster R-CNN 발전 과정 핵심 요약, [https://youtu.be/jqNCdjOB15s](https://youtu.be/jqNCdjOB15s)

sliding window: [https://velog.io/@cha-suyeon/딥러닝-Object-Detection-Sliding-Window-Convolution](https://velog.io/@cha-suyeon/%EB%94%A5%EB%9F%AC%EB%8B%9D-Object-Detection-Sliding-Window-Convolution)

region proposal, anchor box: [https://rubber-tree.tistory.com/133](https://rubber-tree.tistory.com/133)
