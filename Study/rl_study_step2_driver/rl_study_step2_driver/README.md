# RL study step 2 — drowsy driver model

이번 단계 목표는 **RL 없이 운전자 모델 자체를 이해하고 구현하는 것**이다.

핵심 구조는 아래와 같다.

\[
\text{perception delay}
\rightarrow
\text{raw steering intent}
\rightarrow
\text{LPF}
\rightarrow
\text{motor delay}
\rightarrow
\delta_{\mathrm{sfa,cmd}}
\]

그리고 실제 차량에 들어가는 driver-equivalent road-wheel steering은

\[
\delta_{\mathrm{drv}}^{\mathrm{rwa}}
=
\frac{\delta_{\mathrm{sfa}}}{i_s}
\]

이다.

## 파일 구성

- `vehicle.py` : 1단계의 최소 SBW lateral plant
- `geometry.py` : obstacle ellipse, barrier function
- `driver.py` : split-delay drowsy-driver model
- `test_driver_reaction.py` : driver-only 시뮬레이션과 결과 저장

## 운전자 모델 수식

### 1. 지연된 인지
차량이 실제로 보고 있는 상태가 아니라, perception delay가 지난 상태를 사용한다.

\[
\tilde x_k = x_{k-d_p}
\]

여기서 `d_p`는 perception delay step 수다.

### 2. 차선 유지 항
지연된 lateral error와 heading error로 차선 유지 성분을 만든다.

\[
u_{\text{lane},k}
=
K_y (y_{\text{ref}} - \tilde y_k)
+
K_\psi (-\tilde\psi_k)
\]

### 3. 장애물 회피 항
장애물까지의 지연된 상대거리

\[
d_x = x_o - \tilde x_k, \qquad d_y = \tilde y_k - y_o
\]

를 써서 repulsive steering term을 만든다.

코드에서는

\[
u_{\text{obs},k}
=
s \cdot K_{\text{obs}}
\cdot \text{hazard}
\cdot \text{proximity}
\cdot \text{lateral\_amp}
\]

형태로 쓴다.

- `s` : 회피 방향 부호
- `hazard = 1 - d_x / d_{\text{detect}}`
- `proximity` : 장애물에 가까워질수록 커지는 항
- `lateral_amp` : obstacle centreline 가까울수록 커지는 항

### 4. burst reaction
지연된 barrier 값이 작아지면 burst multiplier를 건다.

\[
u_{\text{raw},k}
=
u_{\text{lane},k}
+
\beta_k u_{\text{obs},k}
\]

\[
\beta_k =
\begin{cases}
K_{\text{burst}}, & h(\tilde x_k, \tilde y_k) \le h_{\text{trig}} \\
1, & \text{otherwise}
\end{cases}
\]

### 5. low-pass filter
손이 한 번에 확 따라가지 않도록 LPF를 건다.

\[
u_{\text{lpf},k}
=
(1-\alpha) u_{\text{lpf},k-1}
+
\alpha u_{\text{raw},k}
\]

### 6. motor delay
최종 steering-wheel command는 motor delay buffer를 지난 값이다.

\[
\delta_{\mathrm{sfa,cmd},k} = u_{\text{lpf},k-d_m}
\]

## 실행

```bash
cd /mnt/data/rl_study_step2_driver
python test_driver_reaction.py
```

생성 파일:

- `artifacts/driver_step2_log.csv`
- `artifacts/driver_step2_preview.png`

## 이번 단계에서 확인할 것

1. `perceived_x`와 실제 `x`가 다른 이유
2. `raw_cmd -> lpf_cmd -> sfa_cmd` 로 갈수록 왜 늦어지는지
3. `delta_sfa`가 먼저 움직이고, `delta_drv_rwa = delta_sfa / steering_ratio`가 더 작게 따라오는 이유
4. driver-only에서는 왜 **반응은 하지만 늦어서 못 피하는지**

## 다음 단계

다음 단계에서는 이 driver와 1단계 vehicle을 묶어서

\[
\delta_{\text{cmd}}
=
\lambda\,\delta_{\text{drv}}^{\mathrm{rwa}}
+
(1-\lambda)\,\delta_{\text{auto}}^{\mathrm{rwa}}
\]

형태의 **fixed-\(\lambda\)** shared control 실험으로 넘어가면 된다.
