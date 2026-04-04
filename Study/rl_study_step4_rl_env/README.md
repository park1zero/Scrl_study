# Step 4 — RL Environment without Safety Filter

이번 단계의 목표는 **authority allocation을 RL 문제로 정확히 정의하는 것**이다.

이 단계에서는 `safety filter`를 **의도적으로 넣지 않는다**. 즉 RL action이 직접 authority를 바꾼다.

## 1. 핵심 식

최종 road-wheel steering command는

\[
\delta_{\mathrm{cmd},k}
=
\lambda_k \, \delta_{\mathrm{drv},k}^{\mathrm{rwa}}
+
(1-\lambda_k) \, \delta_{\mathrm{auto},k}^{\mathrm{rwa}}
\]

이고, RL action은 steering angle이 아니라 authority 변화율이다.

\[
a_k \in [-1,1]
\]

\[
\lambda_{k+1}
=
\operatorname{clip}\left(
\lambda_k + \dot\lambda_{\max} \Delta t \, a_k,
0,
1
\right)
\]

이 단계에서는

\[
\lambda_{k+1}^{\mathrm{safe}} = \Pi_{\mathcal S}(\lambda_{k+1})
\]

같은 safety projection을 **쓰지 않는다**.

## 2. 파일 설명

- `vehicle.py`  
  minimal SBW dynamic bicycle model
- `driver.py`  
  split-delay drowsy-driver model
- `automation.py`  
  간단한 obstacle-avoidance steering law
- `env.py`  
  RL environment (`reset`, `step`, observation, reward, done)
- `policies.py`  
  random / driver-hold / full-takeover / heuristic baseline authority policies
- `inspect_observation.py`  
  observation과 한 step transition을 출력해서 이해하는 용도
- `rollout_policy_demo.py`  
  몇 가지 authority policy를 환경에서 굴려보고 CSV와 플롯 저장

## 3. Observation

현재 observation은 13차원이다.

1. `y_norm`
2. `psi_norm`
3. `vy_norm`
4. `r_norm`
5. `delta_rwa_norm`
6. `delta_swa_norm`
7. `delta_drv_rwa_norm`
8. `delta_auto_rwa_norm`
9. `lambda`
10. `h_norm`
11. `ttc_norm`
12. `dx_norm`
13. `dy_norm`

## 4. Reward

reward는 아래 구조를 쓴다.

\[
r_k
=
 w_p \Delta x_k
- w_y y_k^2
- w_\psi \psi_k^2
- w_h [h^\star-h_k]_+^2
- w_\lambda (1-\lambda_k)^2
- w_{\Delta\lambda}(\Delta\lambda_k)^2
- w_c (\delta_{\mathrm{cmd},k}-\delta_{\mathrm{drv},k}^{\mathrm{rwa}})^2
\]

그리고 terminal bonus / penalty를 추가한다.

- collision: 큰 음수 패널티
- road departure: 음수 패널티
- success: 양의 보너스

즉 이 reward는
- 안전하게 obstacle을 피하고
- 필요 이상으로 authority를 빼앗지 말고
- authority를 갑자기 바꾸지 말고
- driver와 너무 크게 충돌하지 말라

는 목적을 동시에 담고 있다.

## 5. 실행 방법

```bash
cd /mnt/data/rl_study_step4_rl_env
python inspect_observation.py
python rollout_policy_demo.py
```

생성 파일:
- `artifacts/step4_policy_preview.png`
- `artifacts/step4_policy_summary.csv`
- `artifacts/step4_driver_hold_log.csv`
- `artifacts/step4_full_takeover_log.csv`
- `artifacts/step4_heuristic_log.csv`
- `artifacts/step4_random_log.csv`

## 6. 이 단계에서 꼭 이해해야 할 것

이 환경에서는 RL이 **차를 직접 조향하지 않는다**.

RL이 학습하는 것은 오직

\[
\text{"지금 authority } \lambda \text{를 얼마나 빨리 줄일까/늘릴까?"}
\]

뿐이다.

즉 이 프로젝트는 end-to-end driving RL이 아니라,

**shared control supervisory RL**

이다.

## 7. 다음 단계

다음은 이 환경 위에 아주 작은 RL 학습기를 얹는 단계다.

추천 순서:
1. random action rollout
2. heuristic authority baseline
3. tiny actor network
4. replay buffer
5. SAC 또는 PPO
