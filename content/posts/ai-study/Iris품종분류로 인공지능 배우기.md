# 🌱 인공지능 첫걸음: Iris 품종 분류로 배우는 머신러닝 워크플로우

> **“모두가 딥러닝을 말하지만, 나만은 ‘Hello AI’도 버벅인다.”**
> 이 블로그는 그런 당신(=나)을 위한 엔드 투 엔드 튜토리얼이다.

---

## 🎯 목표

Iris 품종 분류라는 세계에서 가장 친근한 머신러닝 예제로, 인공지능 모델의 **기본 워크플로우**를 직접 따라 해보는 것이 목표.
(당신이 모르는 수학은 잠시 묻어두자. 파이썬만 있으면 된다. 아니, 있어야 한다. 제발.)

---

## 🧰 1단계: 최소한의 워크플로우

### 🔧 1. 환경 설정 (이거 안 하면 기계가 말을 안 들어요)

* Python 3.7 이상 (3.12? 그런 건 아무도 안 써요… 아직은)
* 가상환경 생성:

  ```bash
  python -m venv myenv
  source myenv/bin/activate  # Windows라면 myenv\Scripts\activate
  ```
* 필수 라이브러리 설치:

  ```bash
  pip install numpy pandas scikit-learn matplotlib
  ```

> ✨ **주의:** Anaconda 쓴다고 자랑하지 마세요. 모두가 한 번은 깔았다가 후회합니다.

---

### ❓ 2. 문제 정의: 회귀냐 분류냐 그것이 문제로다

* 회귀: 숫자를 예측 (예: 집값)
* 분류: 카테고리를 예측 (예: 꽃 품종)

우리의 목표는 **꽃을 보면 그 품종을 맞추는 분류 문제**.

---

### 📦 3. 데이터 준비

* `scikit-learn`의 내장 **Iris 데이터셋**을 사용
* 실제 프로젝트에서는 CSV 수집, 클렌징, 요약 정리 등 다양한 경험이 기다리니 가능한 일은 다 해보자.

---

### 🧼 4. 데이터 전처리

* Iris는 깨끗한 공주님 같은 데이터라 전처리할 것이 거의 없음
* 하지만 보통은:

  * 결측치 체크 및 처리
  * 범주형 변수 인코딩
  * 특성 스케일링 (standardization / normalization)

---

### 🧠 5. 모델 선택

* 분류 문제니까 `LogisticRegression`부터 시작
* 그냥 이름만 "회귀"지 분류도 잘함. 이름에 속지 마라.

---

### 🏋️ 6. 모델 학습

* `fit()` 한 줄이면 뭔가 여러 가지를 한 느낌을 줄 수 있음
* 현실은 데이터를 나누어서 train/validation/test로 관리해야 진짜 프로

---

### 🧪 7. 모델 평가

* 정확도(accuracy), 혼동 행렬(confusion matrix)로 성능 확인
* “0.95 나왔어요!” 라고 좋아하다가, 전부 setosa만 맞춘 모델일 수도 있다는 거, 알지?

---

## 🌸 2단계: Iris 품종 분류 예제

```python
# 1) 라이브러리 불러오기
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# 2) 데이터 로드
iris = load_iris()
X = iris.data
y = iris.target

# 3) 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 4) 모델 정의 및 학습
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 5) 예측 및 평가
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"테스트 정확도: {acc:.2f}")
print("혼동 행렬:")
print(cm)
```

---

## 🧵 코드 설명 (초심자를 위한 아주 친절한 해설)

* `load_iris()`로 꽃 데이터 불러오기 (아쉽게도 진짜 꽃은 안 옴)
* `train_test_split()`으로 80:20 비율 나누기
* `LogisticRegression()` 모델 생성 → `fit()`으로 학습
* `predict()`로 결과 예측 → `accuracy_score()`, `confusion_matrix()`로 평가

---

## 🚀 마무리: 이 한 파일이면 뭐든 시작할 수 있다

딱 이 한 파일만 실행해보면 머신러닝의 전체 흐름을 맛볼 수 있다.

**데이터 불러오기 → 전처리 → 모델 학습 → 평가 → 눈물**

머신러닝 입문은 이렇게 시작된다.
기억해라, 결국 당신은 TensorFlow와 PyTorch 사이에서 길을 잃게 될 것이다.

