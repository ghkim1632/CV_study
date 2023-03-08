# U-Net

## U-Net: Convolutional Networks for Biomedical Image Segmentation


> abstract : 이 논문은 image augmentation을 통해 작은 데이터 셋으로 얼마나 효율적으로 학습 할 수 있는지 설명합니다. 

### Architecture
- 작은 데이터 셋으로 어떻게 효율적인 학습을 했는지 설명합니다.
- 지도학습에서 성능을 높이는 방법으로는 Krizhevsky et al이 고안한 돌파구인 거대한 모델을 만드는 것이었다. 
- 특히 이미지 분류 모델을 single class label을 가지만 의료계를 비롯한 많은 업계에서는 다양한 테스크, 위치를 포함하는 결과물을 열망했다. 
- 예측 이미지의 각각의 픽셀에 class를 예측하기 위해서 __sliding window__ 방식이 사용되었다. 

![image.png](.\fig1.png)

__encoder__
- U-Net 구조는 각각의 파란 박스가 multi-channel feature map에 해당한다. 
- 하얀 박스는 복사된 feature map에 해당한다. 
- U-Net 아키텍처의 encoding 부분은 2x2 max pooling사용하여 해상도가 2배 감소
- conv 연산으로 채널 크기는 2배 증가
- 일반적인 cnn모델처럼 다음 형식을 반복
    - conv -> ReLU -> MaxPooling
- 실제로는 encoder 부분은 pre-train된 분류모델을 사용하기도 한다. 

__decoder__
- 2x2 convolution(up-convolution) 사용하여 해상도가 2배 증가
- conv 연산으로 채널 크기는 2배 감소
- 수축 경로(contracting path)에서 처리된 feature채널 레벨에서 연결 map을 잘라내어 가져와 수행


### Overlap-tile
![image](.\fig2.png)
- FCN 특성상 입력 이미지의 해상도에는 제한이 없음
- 하지만 U-Net 구조상 출력 이미지의 해상도가 입력 이미지보다 작음
- 실제로 노란색 영역의 segmentation이 필요하면 더 큰 범위(파란색 영역)의 패치를 삽입함
- 이미지의 경계 부분은 extrapolation을 사용합니다. 

### training
- U-Net은 segmentation을 위한 네트워크이므로, 픽셀 단위로 소프트맥스를 사용한다.
![image](.\fig3.png)

- 학습을 위해 다음의 cross-entropy 손실을 사용(true label만 고려하므로, 일반 cross entropy공식과 동일)
![image](.\fig4.png)


__W(x)__
 - 세포를 명확히 구분하기 위해 작은 분리 경계를 학습함
 - w(x)는 인접한 셀 사이에 있는 배경 레이블에 대하여 높은 가중치를 부여함
  ![image](.\fig5.png)

### data augmentation
- rotation, crop, flip 등 일반적인 data augmentation기법 사용
- 추가적으로 __elastic deformation__ 방법 사용

__elastic deformation 이란?__
![image](.\fig6.png)