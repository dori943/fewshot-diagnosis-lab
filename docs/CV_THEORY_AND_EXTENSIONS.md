# 이 미션 속 Computer Vision 이론 정리 + 확장 로드맵

> 이 미션 코드에서 쓴 기술이 **왜** 동작하는지 정리하고, CV 연구/실무로 넓혀 갈 다음 주제를 묶었다.
> 각 절의 `📍코드` 표시는 레포에서 해당 개념이 구현된 위치다.

---

## Part 1. 모델 진단의 이론적 뼈대

### 1.1 편향–분산 분해
제곱오차 기준 기대 일반화 오차는

```
E[(y − f̂(x))²] = Bias[f̂]²  +  Var[f̂]  +  σ²(줄일 수 없는 잡음)
```

- **Bias**: 모델 가설 공간이 정답 함수를 표현하지 못해서 생기는 오차 → Train Loss가 높다.
- **Variance**: 학습 데이터 샘플링에 따라 모델이 크게 흔들리는 정도 → Train/Validation 격차(Gap).
- **σ² (Bayes error)**: 어떤 모델로도 줄일 수 없는 오차. 사람도 구분 못 하는 32×32 이미지, 일별 주가 잡음이 대표적.

Andrew Ng(*Machine Learning Yearning*)의 실무형 분해:

| 항목 | 정의 | 크면 할 일 |
|---|---|---|
| Avoidable bias | Train 오차 − Bayes(≈사람 수준) 오차 | 더 큰 모델, 더 오래 학습, 더 나은 구조 |
| Variance | Validation 오차 − Train 오차 | 데이터↑, 정규화, 증강 |
| Data mismatch | (Train 분포 Validation) 오차 − Train 오차 | 분포를 맞추는 데이터 수집/합성 |

📍코드 `src/common/utils.py::diagnose_bias_variance` — 기준 Loss(ln C, 또는 평균 예측 MSE)를 Bayes error 대신 상한으로 써서 규칙화했다. Track B 리포트에서 "rel_train≈1은 Bayes error 탓일 수 있다"고 경고하는 이유가 이것이다.

### 1.2 Double Descent와 현대 딥러닝
전통적 U자 곡선과 달리, 파라미터가 매우 많은 모델은 보간 임계점을 지나면 오히려 Test 오차가 다시 줄어드는 **double descent**가 관찰된다(Belkin 2019, Nakkiran 2019). 사전학습 + 정규화된 큰 모델이 Few-shot에서도 잘 되는 현상과 연결된다. "파라미터가 많으면 무조건 과적합"이라는 직관은 진단 시 곡선으로 확인해야 하는 이유다.

### 1.3 Loss 곡선을 읽는 법
| 모양 | 의미 |
|---|---|
| Train·Validation 모두 높고 평평 | High Bias / 학습률 너무 작음 / 동결 범위 과다 |
| Train↓ 계속, Validation은 최저점 후 ↑ | 과적합 시작 → Early Stopping 지점 |
| Validation이 Train보다 낮음 | Dropout·증강이 Train Loss에만 걸린 효과 (📍코드는 eval 모드 Train Loss 재계산으로 보정) |
| 심한 진동 | 학습률 과대, 배치 작음, Validation 셋 작음 |

---

## Part 2. CNN — 이미지에 맞는 귀납적 편향

### 2.1 합성곱의 3가지 가정
1. **지역성(Locality)**: 픽셀은 가까운 픽셀과 관련이 크다 → 작은 커널(3×3).
2. **가중치 공유(Weight sharing)**: 같은 패턴 검출기를 모든 위치에 적용 → 파라미터 수 급감.
3. **평행이동 등변성(Translation equivariance)**: 입력이 움직이면 feature map도 같이 움직임. Pooling/전역 평균은 여기에 근사적 **불변성**을 더한다.

이 귀납적 편향 덕분에 CNN은 데이터가 적어도 잘 학습된다. 반대로 ViT는 이런 가정이 약해 대규모 사전학습이 필요하다.

### 2.2 수용 영역과 계층적 특징
층이 깊어질수록 수용 영역(receptive field)이 넓어져 **엣지 → 텍스처 → 부분 → 객체**로 추상화된다(Zeiler & Fergus 2014). 전이 학습에서 "앞단은 고정, 뒷단만 재학습"이 합리적인 근거다.

### 2.3 이 미션에서 쓰는 Backbone
| 모델 | 핵심 아이디어 | 비고 |
|---|---|---|
| **ResNet** (He 2016) | Residual connection `y = F(x) + x` → 기울기 소실 없이 깊게 | ResNet18: 11.2M 파라미터, stage = layer1~4 |
| **EfficientNet** (Tan & Le 2019) | 깊이·너비·해상도를 함께 키우는 compound scaling, MBConv + Squeeze-and-Excitation | B0: 5.3M, 모바일 친화 |
| BatchNorm | 미니배치 통계로 정규화, running mean/var 저장 | **Few-shot에서 함정** (2.4 참고) |

### 2.4 BatchNorm과 Freezing의 함정
`requires_grad=False`만 하면 가중치는 고정돼도 `model.train()`에서 BN의 running 통계는 **계속 갱신**된다. 작은 데이터의 통계로 ImageNet 통계를 덮어쓰면 특징이 망가진다.
📍코드 `track_a/train.py::set_train_mode` — Linear probing/Partial에서 동결 구간 BN을 eval로 고정.

### 2.5 CNN vs RNN — 구조적 차이
| | CNN | RNN (LSTM/GRU) | Transformer |
|---|---|---|---|
| 데이터 가정 | 공간적 지역성, 격자 구조 | 순차적 의존, 시간 순서 | 집합 + 위치 인코딩 |
| 파라미터 공유 | 공간 위치 간 | 시점 간 | 토큰 간 (attention 가중치는 입력 의존) |
| 장거리 의존 | 깊이로 확장 | 게이트로 완화, 여전히 어려움 | 한 층에서 전 구간 연결 (O(n²)) |
| 병렬화 | 높음 | 낮음 (순차 계산) | 높음 |
| 대표 도메인 | 이미지, 영상 프레임, 스펙트로그램 | 센서/금융 시계열, 짧은 시퀀스 | 언어, ViT, 장기 시계열 |

LSTM은 입력·망각·출력 게이트와 셀 상태로 기울기 흐름을 보존하고, GRU는 게이트를 2개(update, reset)로 줄여 파라미터가 적다. 1D-CNN/TCN도 시계열에 쓰이며, 반대로 ConvLSTM은 영상(시공간)에 쓰인다 — 경계는 "데이터 구조에 맞는 귀납적 편향"이다.

---

## Part 3. 전이 학습 (Transfer Learning)

### 3.1 왜 전이가 되는가
Yosinski et al.(2014)은 앞단 층의 특징은 범용적이고, 뒷단으로 갈수록 원 과제에 특화됨을 보였다. ImageNet 1.28M장으로 학습한 앞단은 다른 자연 이미지 과제에서도 유용하다.

### 3.2 전략 스펙트럼
| 전략 | 학습 대상 | 장점 | 위험 |
|---|---|---|---|
| Scratch | 전체 (랜덤 초기화) | 도메인 완전 적합 가능 | 소량 데이터에선 과적합/학습 실패 |
| **Linear Probing** | 분류기만 | 분산 작음, 빠름 | 특징이 도메인과 안 맞으면 High Bias |
| **Partial** | 마지막 stage + 분류기 | 균형 | 어디까지 풀지 실험 필요 |
| **Full Fine-tuning** | 전체 (작은 LR) | 최고 성능 잠재력 | 소량 데이터에서 특징 왜곡, 과적합 |

📍코드 `track_a/train.py::build_model`, `param_groups` (backbone LR × 0.1 = **차등 학습률**, ULMFiT의 discriminative fine-tuning).

### 3.3 알아두면 좋은 연구 포인트
- **LP-FT** (Kumar et al., ICLR 2022): 랜덤 Head로 바로 Full FT하면 초기 큰 기울기가 사전학습 특징을 왜곡해 OOD 성능이 떨어진다 → **Linear probe 먼저 → Full FT**. 이 레포에선 `outputs/track_a/linear_probe/best.pt`를 불러와 `full`을 시작하도록 `train.py`에 로딩 한 줄을 추가하면 실험 가능.
- **해상도 불일치**: CIFAR 32×32를 224로 업샘플하면 ImageNet 특징과 통계가 다르다. FixRes(Touvron 2019)는 Train/Test 해상도 차이가 성능에 미치는 영향을 분석했다. `--img_size`를 바꿔 비교해 보면 좋은 소실험이 된다.
- **Catastrophic forgetting**: Full FT가 원래 지식을 잃는 현상. L2-SP(사전학습 가중치 쪽으로 정규화), EWC 등이 대응책.
- **도메인 간극**: 의료/위성/산업 이미지는 ImageNet과 거리가 멀어 Linear probing이 약하다 → 도메인 내 자기지도 사전학습이 유리.

---

## Part 4. Few-shot Learning

### 4.1 문제 정의
N-way K-shot: N개 클래스, 클래스당 K장. 이 미션은 5-way 40-shot으로 "적은 데이터의 지도학습"에 가깝다.

### 4.2 주요 접근
| 계열 | 대표 | 아이디어 |
|---|---|---|
| 전이 + 미세조정 | Baseline++ (Chen 2019, *A Closer Look at Few-shot Classification*) | 좋은 사전학습 + 단순 fine-tune이 복잡한 메타학습과 대등 |
| Metric learning | Siamese, **Prototypical Networks** (Snell 2017) | 클래스 평균 임베딩(prototype)과의 거리로 분류 |
| Meta-learning | MAML (Finn 2017) | 몇 step만에 적응하는 초기값 학습 |
| Foundation model | **CLIP** zero-shot, **DINOv2** linear probe | 대규모 사전학습 표현으로 0~수 장만으로 분류 |

최근 실무 결론: **강한 사전학습 표현 + Linear probe/가벼운 FT**가 가장 비용 대비 효과가 크다.

---

## Part 5. 데이터 증강 — 오차 분석과 짝을 이루는 처방

### 5.1 종류
| 분류 | 기법 | 대응하는 실패 태그 |
|---|---|---|
| 기하 | RandomResizedCrop, Flip, Rotation, Affine | 위치/크기 변화, 일부만 보임 |
| 광도 | ColorJitter(밝기·대비·채도), Grayscale | **데이터 품질: 어두운 조명** |
| 화질 | GaussianBlur, JPEG 압축, 노이즈 | 데이터 품질: 흐림·저해상도 |
| 가림 | RandomErasing, Cutout | **배경/잡음: 객체 가림** |
| 혼합 | Mixup, CutMix | 결정 경계 매끄럽게, 과신 완화(모델 한계) |
| 자동 탐색 | AutoAugment, RandAugment, TrivialAugment | 정책 자동화 |

📍코드 `track_a/train.py::build_transforms("strong")` — 태그별 처방 증강을 묶었다.

### 5.2 원칙
- **라벨 보존**: 증강이 클래스 의미를 바꾸면 안 된다 (숫자 6/9 회전, 색이 클래스인 꽃 분류에 hue 증강 과다).
- **오차 분석 → 증강** 순서: 무작정 강한 증강은 오히려 Bias를 키운다. 태그 통계로 근거를 만들고, `compare`로 **태그별 회복률**을 측정한다.
- 시계열은 다르다: 금융 시계열에 Jitter/Time-warp는 자기상관·변동성 군집 구조를 깨뜨린다 (미션 금지사항).

---

## Part 6. 오차 분석 (Error Analysis)

### 6.1 Ng식 오차 분석 스프레드시트
1. Validation/Test의 오분류 표본 50~100개를 **직접 본다**.
2. 실패 원인을 태그로 단다 (중복 가능).
3. 태그 비율 = **해당 문제를 완벽히 고쳤을 때 얻을 수 있는 성능 상한(ceiling)**.
   예: 오분류의 8%만 "흐림"이면, 흐림을 완벽히 해결해도 오차가 8%만 줄어든다 → 우선순위 낮음.

📍코드 `track_a/error_analysis.py` — 자동 힌트(밝기·선명도·엣지 밀도·신뢰도)는 **보조**이고, 최종 태그는 사람이 단다(Human-in-the-loop).

### 6.2 함께 보면 좋은 도구
- **혼동행렬**: 특정 클래스 쌍(cat↔dog)에 오류가 몰리면 '클래스 유사성' → 세밀 분류(fine-grained) 기법 필요.
- **신뢰도 분석 / Calibration**: "틀렸는데 확신도 0.95"가 많으면 과신(overconfidence). Expected Calibration Error(ECE), Reliability diagram, **Temperature scaling**(Guo 2017)으로 보정.
- **라벨 오류 탐지**: Confident Learning(Northcutt 2021, `cleanlab`) — 예측 확률과 주어진 라벨의 불일치로 오라벨 후보 검출. CIFAR/ImageNet 테스트셋에도 라벨 오류가 존재함이 보고되었다.
- **Grad-CAM** (Selvaraju 2017): 모델이 이미지의 어디를 보고 판단했는지 히트맵. 객체 대신 배경(풀밭→말, 하늘→새)을 보면 **shortcut learning**(Geirhos 2020) → '배경/잡음' 태그의 객관적 근거가 된다.

---

## Part 7. 시계열 쪽 이론 요약 (Track B)
- **베이스라인이 강한 이유**: 주가는 랜덤워크에 가깝다 → Naive `ŷ(t)=y(t-1)`가 최적에 가까움. 딥러닝이 이를 못 이기는 것은 실패가 아니라 **정상적인 발견**이며, 이걸 정량적으로 보여주는 것이 미션의 핵심.
- **지연(lag) 착시**: 가격 수준 예측 그래프가 "잘 맞아 보이는" 것은 하루 늦게 따라가기 때문. 수익률/방향 정확도로 보면 실체가 드러난다.
- **Diebold-Mariano 검정**: 두 예측의 오차 차이가 통계적으로 유의한지 검정 (📍`baselines.py::diebold_mariano`).
- **Attention**: Bahdanau(2015) additive attention — 과거 각 시점에 가중치를 줘 중요한 시점을 선택. 📍`models.py::AdditiveAttention`, 결과 `attention_weights.png`에서 최근 시점에 몰리는지 확인.

---

## Part 8. 하이퍼파라미터 탐색
- **Random search > Grid search** (Bergstra & Bengio 2012): 중요한 파라미터는 소수라서, Grid는 중요하지 않은 축에 예산을 낭비한다. Random은 각 축마다 서로 다른 값을 시험한다.
- **로그 스케일 샘플링**: 학습률·weight decay는 10배 단위로 효과가 달라진다.
- **Coarse-to-Fine**: 넓게 뿌린 뒤 상위 영역으로 좁힌다. 📍`hparam_search.py::narrow`
- 다음 단계: **Bayesian Optimization / TPE**(Optuna), **Hyperband / ASHA**(나쁜 trial 조기 중단), Population Based Training.
- 영향도 분석: RandomForest 중요도, fANOVA, Optuna `plot_param_importances`.

---

## Part 9. 확장 로드맵 (CV 관심자용)

난이도 ★(쉬움)~★★★(연구급). 위에서부터 이 레포에 바로 붙일 수 있는 순서다.

### 9.1 이 레포에 바로 붙이기
| 주제 | 내용 | 난이도 |
|---|---|---|
| **Grad-CAM 오차 분석** | 오분류 이미지에 히트맵을 겹쳐 '배경/잡음' 태그를 근거화 (`pytorch-grad-cam`) | ★ |
| **LP-FT** | linear_probe 가중치로 full 초기화 → Full FT와 비교 | ★ |
| **Mixup / CutMix / RandAugment** | 증강 정책 비교 표 | ★ |
| **Calibration** | ECE 측정 + Temperature scaling 전후 비교 | ★ |
| **Test-Time Augmentation (TTA)** | 여러 증강 예측 평균 | ★ |
| **cleanlab 라벨 오류 탐지** | '라벨 오류' 태그를 자동 후보화 후 사람 확인 | ★★ |
| **K-shot 곡선** | K=5,10,20,40에서 전략별 성능 → "데이터 크기별 최적 Freeze 범위"를 그래프로 증명 | ★★ |
| **Backbone 비교** | ResNet vs EfficientNet vs ConvNeXt vs ViT/DINOv2 (`timm`) | ★★ |

Grad-CAM 최소 예시:
```python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
cam = GradCAM(model=m, target_layers=[m.layer4[-1]])
heat = cam(input_tensor=x)[0]                   # (H, W) 0~1
vis = show_cam_on_image(img_float_rgb, heat, use_rgb=True)
```

### 9.2 Foundation model 시대의 Few-shot
| 주제 | 포인트 | 난이도 |
|---|---|---|
| **CLIP zero/few-shot** | 텍스트 프롬프트로 분류, 이미지-텍스트 임베딩 정렬. Linear probe와 비교 | ★★ |
| **DINOv2 특징 + k-NN / Linear** | 자기지도 ViT 특징의 강력함 확인 | ★★ |
| **PEFT** | LoRA, Adapter, Visual Prompt Tuning(VPT) — 소량 파라미터만 학습 | ★★ |
| **Prototypical Network** | 에피소드 학습으로 metric learning 직접 구현 | ★★ |

### 9.3 데이터가 부족할 때의 고급 전략
| 주제 | 포인트 | 난이도 |
|---|---|---|
| **자기지도 학습** | SimCLR, MoCo, BYOL, **MAE**, DINO — 라벨 없는 도메인 이미지로 사전학습 | ★★★ |
| **준지도 학습** | Pseudo-label, **FixMatch** — 소량 라벨 + 대량 비라벨 | ★★ |
| **Active Learning** | 불확실성 높은 샘플부터 라벨링 → 오차 분석의 자동화 버전 | ★★ |
| **합성 데이터** | Diffusion 기반 증강, 도메인 랜덤화 | ★★★ |
| **Domain Adaptation** | DANN 등 소스/타깃 분포 정렬 (Data mismatch 대응) | ★★★ |

### 9.4 신뢰성 (실무 필수)
- **OOD 탐지**: Max softmax, Energy score, Mahalanobis — "모르는 것을 모른다고 말하기".
- **Robustness 벤치마크**: ImageNet-C(노이즈·블러·날씨 손상)로 '데이터 품질' 태그를 체계적으로 평가.
- **Fairness / Slice 분석**: 조건(조명·배경)별 성능 슬라이스 — 오차 분석 태그를 평가 지표로 승격.

### 9.5 도메인 프로젝트로 확장
| 도메인 | 과제 | 대표 기술/데이터 |
|---|---|---|
| **스마트팩토리** | 불량 검출 (정상 데이터만 많음) | 이상 탐지 **PatchCore, PaDiM, EfficientAD**, 데이터셋 **MVTec AD, VisA**; 엣지 배포는 지식 증류 + 양자화(ONNX/TensorRT) |
| 검출/분할 | 위치까지 찾기 | YOLO 계열, DETR, Mask R-CNN, **SAM**(프롬프트 분할) |
| 의료 | 소량·고해상도·라벨 비용 큼 | 도메인 자기지도 사전학습, U-Net, Calibration 중요 |
| 영상/시계열 융합 | 센서 + 카메라 | ConvLSTM, Video Transformer, 멀티모달 융합 |

### 9.6 Track B 쪽 확장 (참고)
- Walk-forward(rolling-origin) 검증, 다변량 입력(거래량·지수·금리), 방향 정확도·수익률 기반 백테스트.
- 모델: TCN, N-BEATS, Temporal Fusion Transformer, PatchTST, 시계열 foundation model(Chronos, TimesFM).
- 확률 예측: Quantile loss, 예측 구간(conformal prediction).

---

## Part 10. 추천 학습 순서
1. 이 레포 결과로 **K-shot 곡선 + Grad-CAM 오차 분석**을 완성 (포트폴리오 핵심 그림 2장)
2. CLIP / DINOv2 Linear probe를 같은 Few-shot 셋에 붙여 "Foundation model vs ImageNet CNN" 비교
3. MVTec AD로 PatchCore 재현 → 스마트팩토리 도메인 프로젝트
4. 논문 읽기: ResNet → ViT → SimCLR/MAE/DINO → CLIP → LP-FT → PatchCore

## 참고 문헌
- He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016
- Tan & Le, *EfficientNet*, ICML 2019
- Yosinski et al., *How transferable are features in deep neural networks?*, NeurIPS 2014
- Zeiler & Fergus, *Visualizing and Understanding Convolutional Networks*, ECCV 2014
- Kumar et al., *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution*, ICLR 2022
- Howard & Ruder, *ULMFiT*, ACL 2018 (discriminative fine-tuning)
- Chen et al., *A Closer Look at Few-shot Classification*, ICLR 2019
- Snell et al., *Prototypical Networks*, NeurIPS 2017 · Finn et al., *MAML*, ICML 2017
- Radford et al., *CLIP*, ICML 2021 · Oquab et al., *DINOv2*, 2023 · He et al., *MAE*, CVPR 2022
- Zhang et al., *mixup*, ICLR 2018 · Yun et al., *CutMix*, ICCV 2019 · Cubuk et al., *RandAugment*, 2020
- Selvaraju et al., *Grad-CAM*, ICCV 2017 · Geirhos et al., *Shortcut Learning*, Nature MI 2020
- Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017 · Northcutt et al., *Confident Learning*, JAIR 2021
- Bergstra & Bengio, *Random Search for Hyper-Parameter Optimization*, JMLR 2012
- Bahdanau et al., *Neural Machine Translation by Jointly Learning to Align and Translate*, ICLR 2015
- Roth et al., *Towards Total Recall in Industrial Anomaly Detection (PatchCore)*, CVPR 2022
- Andrew Ng, *Machine Learning Yearning*
