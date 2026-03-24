```python
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

x_data = [[80, 220, 6300], [75, 167, 4500], [86, 210, 7500], [110, 330, 9000],
          [95, 280, 8700], [67, 190, 6800], [79, 210, 5000], [98, 250, 7200]]
y_data = [2, 3, 1, 0, 0, 3, 2, 1]

x_train = torch.FloatTensor(x_data)
y_train = torch.LongTensor(y_data)

x_norm = (x_train - x_train.mean(dim=0)) / x_train.std(dim=0)

y_one_hot = F.one_hot(y_train, num_classes=4).float() 

W2 = torch.zeros((3, 4), requires_grad=True)
b2 = torch.zeros(1, requires_grad=True) 
optimizer2 = optim.SGD([W2, b2], lr=0.1)

costs2 = []

for epoch in range(10000):
    optimizer2.zero_grad()
    z = x_norm.matmul(W2) + b2 
    cost2 = F.cross_entropy(z, y_one_hot)
    cost2.backward()
    optimizer2.step()
    costs2.append(cost2.item())
```

### 문제 원인 1
다중 분류 모델(클래스 4개) 반환값을 이항 분류 함수인 Sigmoid로 처리
Cell 6 코드에서 예측값 산출 시 `z = torch.sigmoid(x_train.matmul(W1) + b1)`를 선언하고 이를 `F.cross_entropy(z, y_train)`에 대입했습니다. `F.cross_entropy` 내부에는 이미 다중 분류 목적의 Softmax 함수가 내장되어 있습니다. 값의 범위를 0~1로 압축하는 Sigmoid 연산을 거친 z 벡터가 Softmax에 중복 투입되면 Logit 차이가 왜곡되어 Loss 값이 1.386에서 하강하지 못합니다.

### 해결방안
예측값 변수 z에서 Sigmoid 연산을 제외한 순수 선형 결합 수식 `z = x_train.matmul(W1) + b1`로 교체했습니다.

### 결과
`F.cross_entropy`의 다중 확률 계산식에 실수 범위의 Logit 벡터가 그대로 전달되어 Cost 연산이 수식에 맞게 수행되었습니다.


### 문제 원인 2
특성(Feature) 간 수치 스케일이 최대 134배 차이 발생
x_data의 3개 열(Feature)의 수치를 보면 첫 번째 피처(x_1)의 값은 67~110 범위에 분포하지만, 세 번째 피처(x_3)의 값은 4500~9000 범위에 분포해 최저점(67)과 최고점(9000) 기준 스케일 차이가 약 134배입니다. 이를 조정하지 않고 `lr=0.00001`로 학습을 진행하면 경사하강법이 최적해에 수렴하지 못합니다.

### 해결방안
`x_norm = (x_train - x_train.mean(dim=0)) / x_train.std(dim=0)`을 적용하여, x_train 텐서의 평균은 0, 표준편차는 1이 되도록 정규화(Standardization)시켰습니다.

### 결과
3개의 피처 모두 동일한 스케일의 분산을 갖게 되어 `lr=0.1`의 높은 학습률 환경에서도 모델 Parameter(W, b)가 편향 없이 수렴했습니다.


### 문제 원인 3
옵티마이저 변수명 충돌로 인한 W1 기반 기울기(Gradient) 초기값 미적용 및 누적 현상
Cell 12의 학습 루프에서 `optimizer = optim.SGD([W, b], lr=0.1)`를 선언하고 `optimizer.zero_grad()`로 초기화했지만 가중치 예측 및 업데이트는 각각 `z = x_norm.matmul(W1) + b1` 과 `optimizer1.step()`으로 수행했습니다. 그 결과 W1 및 b1에 할당된 10,000번 수량의 기울기가 초기화 없이 누적 연산되었습니다. 더하여 W1과 b1의 재초기화 코드가 누락되어 이전 셀 실행에 따른 `Cost: 0.035300`의 오차가 이관되었습니다.

### 해결방안
목표 가중치 변수인 W2와 b2를 `torch.zeros()`로 수식 윗단에서 0으로 명확히 초기화했고 옵티마이저를 `optimizer2` 단일 개체로 재선언해 `optimizer2.zero_grad()`와 `optimizer2.step()`의 적용 대상을 일치시켰습니다.

### 결과
가중치 값이 0으로 초기화됨에 따라 다중 분류 4개 클래스에 대한 균등 확률인 $\ln(0.25)$의 음수 계산식 도출 비용인 초기 Cost 1.386294부터 올바르게 에폭 계산이 재시작되었습니다. 기울기 또한 매 에폭 1회 단위로 비워져서 Cost가 9000 에폭 시점에 0.053087로 안정적으로 하강했습니다.
