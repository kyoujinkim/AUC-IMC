# AUC-IMC — Smart Consensus

애널리스트의 개별 EPS 추정치를 결합하고, 과거 오차와 추정 시차·시장 환경에 따른 편향을 보정해 **기업·업종의 개선 컨센서스**를 연구하는 프로젝트입니다. 한국·미국, 연간·분기 자료를 다룹니다.


## 연구 결과

세 보고서는 한국 연간·미국 연간·미국 분기 컨센서스를 각각 평가합니다. AUC는 **Adaptive Uncertainty Curve**로 기간 불확실성을 조정하고, IMC는 **Iterated Mean Combination**으로 추정 편의를 보정합니다. 시장·기간·표준화 방법이 다르므로 아래 오차의 절대 크기를 서로 직접 비교하지 않습니다.

### 한국 연간 컨센서스

FY2003~FY2023 EPS를 대상으로 2002-01-01~2024-03-31 컨센서스를 사용하고 EPS를 FY−2 BPS로 표준화했습니다. 과거 10년으로 회귀계수를 추정하되, 5년 이상 자료가 없으면 EW로 대체했습니다. 표 A1의 AUC-IMC는 EW 대비 MAFE를 **발표 6분기 전 3.14%, 7분기 전 2.60%** 줄였습니다. 반면 발표 직전~3분기 전에서는 EW보다 오차가 커서, 개선 효과는 예측 시차에 따라 달랐습니다.

출처: [240809_한국_컨센서스_개선모델.pdf](https://github.com/kyoujinkim/AUC-IMC/blob/master/240809_%ED%95%9C%EA%B5%AD_%EC%BB%A8%EC%84%BC%EC%84%9C%EC%8A%A4_%EA%B0%9C%EC%84%A0%EB%AA%A8%EB%8D%B8.pdf) — PDF 18, 38쪽 — 본문 18쪽, 표 A1.

### 미국 연간 컨센서스

보고서는 1995~2023년 각 연도 시가총액 상위 300기업을 분석합니다. 표 A1~A2의 EPS(FY1)/BPS(FY0) 기준 MAFE와 EW 대비 오차 감소율은 다음과 같습니다.

| 실적 발표까지 시차 | EW MAFE | AUC-IMSE MAFE | 보고된 오차 감소율 | 표시 MAFE로 계산한 감소율 |
|---|---:|---:|---:|---:|
| 1분기 | 5.44 | 4.94 | 10.20% | 약 9.19% |
| 2분기 | 5.99 | 5.51 | 8.70% | 약 8.01% |
| 3분기 | 6.76 | 6.13 | 10.30% | 약 9.32% |

감소율은 보고서 주석상 `1 − MAFE(model) / MAFE(EW)`의 상대 개선율이며, 주가 예측 정확도나 정확도의 %p 개선이 아닙니다. 다만 표 A1~A2의 표시 MAFE로 계산한 감소율은 보고된 감소율과 일치하지 않습니다. 이 차이는 소수 둘째 자리 반올림만으로 설명하기 어려워, 집계 방식·원시 값·표기 오류 여부를 확인하기 전까지 원 수치와 계산값을 구분합니다.

출처: [241218_미국_컨센서스_개선모델.pdf](https://github.com/kyoujinkim/AUC-IMC/blob/master/241218_%EB%AF%B8%EA%B5%AD_%EC%BB%A8%EC%84%BC%EC%84%9C%EC%8A%A4_%EA%B0%9C%EC%84%A0%EB%AA%A8%EB%8D%B8.pdf) — PDF 6, 30쪽 — 분석 범위 6쪽, 표 A1~A2.

### 미국 분기 컨센서스

Bluffer 여부에 따라 모형을 나눈 Mixed Model의 보고된 추정치 오차 개선율은 다음과 같습니다.

| 예측 시차 | AUC-EW | AUC-IMSE | Mixed Model |
|---|---:|---:|---:|
| 당분기 | 6.2% | 6.4% | 9.1% |
| 1분기 전 | 1.4% | 8.0% | 4.4% |
| 2분기 전 | 7.1% | 7.3% | 10.0% |
| 3분기 전 | 6.5% | 6.7% | 9.2% |
| 4분기 전 | 5.6% | 5.8% | 8.5% |

Mixed Model은 여러 시차에서 개선됐지만 **1분기 전에는 AUC-IMSE보다 낮았습니다**. 단순 서프라이즈 롱숏 전략은 2014년 이후 특히 숏 포지션에서 성과가 약해졌다고 보고돼, 컨센서스 오차 개선과 투자 초과성과를 구분해야 합니다.

출처: [250415_미국_분기_컨센서스_모델.pdf](https://github.com/kyoujinkim/AUC-IMC/blob/master/250415_%EB%AF%B8%EA%B5%AD_%EB%B6%84%EA%B8%B0_%EC%BB%A8%EC%84%BC%EC%84%9C%EC%8A%A4_%EB%AA%A8%EB%8D%B8.pdf) — PDF 17, 18, 19쪽 — 그림 21, 투자 활용과 한계 18~19쪽.

### appendix 참고논문의 결론

아래는 방법론의 배경이 되는 외부 연구이며 이 저장소의 재현 성과가 아닙니다.

| 첨부 논문 | 결론에서 확인한 내용 | 출처 |
|---|---|---|
| Forecasting earnings with combination of analyst forecasts | 회귀 기반 결합과 편의 조정이 단순 컨센서스보다 오차를 줄였으며, 단일 단계 조정은 필요한 시계열 길이가 짧다는 장점이 있음 | [1-s2.0-S0927539822000512-main.pdf](https://github.com/kyoujinkim/AUC-IMC/blob/master/appendix/1-s2.0-S0927539822000512-main.pdf) — PDF 21, 22쪽 |
| Forecast Combination With Entry and Exit of Experts | 예측자 진입·이탈로 불균형 패널이 발생하면 복잡한 가중치 추정의 불확실성이 커지며, 단순 편의 조정이 유용할 수 있음 | [Capistrn-ForecastCombinationEntry-2009.pdf](https://github.com/kyoujinkim/AUC-IMC/blob/master/appendix/Capistrn-ForecastCombinationEntry-2009.pdf) — PDF 13, 14쪽 |
| Parsimonious Modeling of Yield Curves | 적은 수의 모수로 금리곡선의 형태를 표현하고, 표본 만기 범위 밖 장기채 가격과의 높은 상관을 확인 | [Parsimonious Modeling of Yield Curves.pdf](https://github.com/kyoujinkim/AUC-IMC/blob/master/appendix/Parsimonious%20Modeling%20of%20Yield%20Curves.pdf) — PDF 16쪽 |
| Uncertainty and the Term Structure of Interest Rates | 금융·거시·정책 불확실성이 금리곡선과 연결되며, 불확실성 충격에 곡선의 수준·기울기·곡률이 반응 | [Uncertainty and the Term Structure of.pdf](https://github.com/kyoujinkim/AUC-IMC/blob/master/appendix/Uncertainty%20and%20the%20Term%20Structure%20of.pdf) — PDF 28쪽 |
| Yield Curve and Financial Uncertainty: Evidence Based on US Data | 미국 1962~2018 월간 자료에서 불확실성 충격 후 금리가 낮아지고 단기물이 더 크게 반응해 기울기가 일시적으로 확대 | [Yield Curve and Financial Uncertainty Evidence Based on.pdf](https://github.com/kyoujinkim/AUC-IMC/blob/master/appendix/Yield%20Curve%20and%20Financial%20Uncertainty%20Evidence%20Based%20on.pdf) — PDF 8쪽 |
| Forecasting Corporate Bond Returns with a Large Set of Predictors: An Iterated Combination Approach | 반복 가중 결합이 비교 모형보다 회사채 수익률의 표본 내·외 예측을 개선 | [lin et al.pdf](https://github.com/kyoujinkim/AUC-IMC/blob/master/appendix/lin%20et%20al.pdf) — PDF 19쪽 |
| Estimating the Yield Curve Using the Nelson–Siegel Model: A Ridge Regression Approach | 기울기·곡률 요인의 다중공선성이 계수 불안정을 유발하며 ridge 정규화가 추정·예측 안정성에 도움 | [smoothed_bootstrap_-_nelson-siegel_revisited_june_2010.pdf](https://github.com/kyoujinkim/AUC-IMC/blob/master/appendix/smoothed_bootstrap_-_nelson-siegel_revisited_june_2010.pdf) — PDF 35쪽 |

실적 발표치가 보정 모형의 학습 시점에 실제로 알려져 있었는지, 결측·애널리스트 진입과 이탈·투자 거래비용을 어떻게 처리했는지는 재현 평가에서 별도로 점검해야 합니다.

## 방법

```text
개별 애널리스트 EPS 추정 이력 + 실제 실적·기업정보
  → 시점·회계기간 정렬
  → 추정 시차와 오차 특성 분석
  → 여러 결합 모형과 보정 모형 비교
  → 종목별 모형 선택·컨센서스 생성
  → 업종 집계·오차·추정 변화 분석
```

`src/enh_EPS.py`에는 `EW`, `PBest`, `IMSE`, `BAM`, `IMC` 및 `_adp` 보정 모형, 혼합 모형 선택 로직이 있습니다. 정확한 가중치와 보정식은 각 모형 구현을 기준으로 비교해야 합니다. 단순 평균과 개선 모형은 같은 기준일·대상·자료 가용 범위에서 평가하는 것이 중요합니다.

## 구조

| 파일 | 역할 |
|---|---|
| `run.py` | 설정·기준일별 추정과 집계 |
| `src/enh_EPS.py` | 컨센서스 결합·편향 보정·모형 선택 |
| `src/funs.py` | 자료 구성·보조 계산 |
| `proc_USListSetup.py`, `proc_usdata.py` | 미국 유니버스·추정 이력 준비 |
| `calc_et.py`, `calc_surprise_ratio.py` | 추가 평가·서프라이즈 관련 분석 |
| 연구 PDF 3개 | 한국·미국·미국 분기 컨센서스 연구 자료 |

## 설치·입력

```bash
python -m pip install -r requirements.txt
```

주요 입력은 `data/<kr|us>/consenlist/*.csv`와 `infos.xlsx`입니다. 기업정보 워크북은 `industry_map`, `share`, `name` 시트 등을 읽습니다. 수집·부가 함수는 별도 설정 파일과 금융데이터 서비스 접근을 요구할 수 있습니다. 실제 자격증명은 비공개 설정으로 관리합니다.

현재 `run.py`는 `src.signals` 등의 추가 모듈도 참조합니다. 공개 소스 목록에서 해당 파일이 준비됐는지 먼저 확인해야 하며, 필요한 모듈·원천 자료 없이 복제만으로 실행된다고 가정해서는 안 됩니다.

## 실행 형태

입력과 추가 모듈을 준비한 뒤 저장소 루트에서 실행합니다.

```bash
python run.py --setting US_Q --quarter 3Q26AS --prddate 2026-10-30 --n-fy 1 --outdir result/us_example
```

분기·기준일은 실제 보유 자료에 맞게 변경합니다.

| 옵션 | 의미 |
|---|---|
| `--setting` | `KR_Q`, `US_Q`, `KR_Y`, `US_Y` |
| `--quarter`, `--n-fy` | 분기 추정 시작점과 추정 분기 수 |
| `--prddate` | 정보·예측 기준일 |
| `--q-basis` | 분기 자료 기준 `QBtw` 또는 `CQBtw` |
| `--reuse_cache`, `--reuse_ucurve`, `--reuse_result` | 기존 자료·계수·결과 재사용 |
| `--pit` | 실제 실적 가림을 이용한 시점 제어 실험 |
| `--outdir` | 별도 결과 저장 경로 |

`--pit`는 기간 말 이후 일정 일수로 실적 이용 가능 시점을 근사합니다. 실제 공시 시각을 사용한 완전한 빈티지 데이터베이스와 같지 않습니다. 연간 설정에는 소스 안의 고정 회계연도 목록도 있으므로 분석 연도에 맞춰 확인해야 합니다.

## 산출물·활용

종목·업종별 추정 CSV, `total_ts.csv`, `coeff.csv`, 보정계수 관련 파일을 저장합니다. 실적 기대 변화, 컨센서스 편향, 업종별 추정 추세 분석의 입력으로 활용할 수 있습니다. 추정 오차가 줄어드는 것과 투자 초과수익이 발생하는 것은 별도 검증 문제입니다.

원천 자료의 공급사·공시 시차·회계기준·통화·적자기업 처리·표본 선택에 따라 결과가 달라집니다. PDF·기업별 추정 이력·시세의 재사용 권한은 코드와 별도로 확인하세요.
