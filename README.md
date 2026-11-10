# Edit-then-Merge

**지식 편집(MEMIT)으로 만든 편집 모델을 원본 모델과 병합(Model Merging)해, 다중 지식 편집에서 생기는 간섭과 성능 붕괴를 완화하는 연구**의 실험 코드입니다.

기본 아이디어는 다음과 같습니다.

1. 원본 GPT2-XL에 편집할 지식(MultiCounterFact) N개를 **MEMIT으로 한 번에 편집**해 편집 모델을 만듭니다.
2. 편집 모델을 **원본 모델과 병합**합니다. 병합은 Task Vector 기반 방법(Task Arithmetic, TIES, DARE-TIES, DARE-Linear)을 사용합니다.
3. 병합 모델의 편집 성능을 **MEMIT 편집 모델(baseline)** 과 비교합니다.

---

## 📁 전체 폴더 구조

```
Edit-then-Merge/
├── README.md                  # (이 문서)
├── memit/                     # 지식 편집 + 평가 + 분석 (memit 환경, 이 폴더에서 실행)
│   ├── tester.py              # 편집/평가 실행 진입점 (--mode edit | merge_test | org_test)
│   ├── model_editor.py        # ModelEditor: 편집·예측·평가·로그·모델 저장
│   ├── common_util.py         # 파일 I/O 등 유틸
│   ├── merge_runner.py        # mergekit으로 편집 모델 + 원본 모델 병합 (mergekit 환경, 하이퍼파라미터 grid 탐색)
│   ├── sample_json.py         # MultiCounterFact에서 N개 랜덤 샘플링
│   ├── summary_result.py      # logs/hclt/* 병합 결과 집계 + 하이퍼파라미터 분석
│   ├── all_summary.py         # run별 메트릭(Score, Efficacy …) 요약 표 출력
│   ├── memit/                 # MEMIT 편집 알고리즘
│   ├── rome/                  # ROME (MEMIT이 layer_stats·repr_tools를 사용)
│   ├── dsets/                 # MultiCounterFact 데이터셋 로더
│   ├── experiments/           # 편집 성능 평가 함수 eval_utils_counterfact.py, summarize.py
│   ├── util/                  # nethook, generate 등 유틸
│   ├── hparams/MEMIT/gpt2-xl.json  # MEMIT 하이퍼파라미터 (편집 레이어 13~17)
│   ├── globals.yml            # 결과/데이터/통계 경로 설정
│   ├── scripts/memit.yml      # memit conda 환경 파일
│   └── memit_requirements.txt # memit 환경 pip 패키지 목록
└── mergekit/                  # 모델 병합 라이브러리 (mergekit 환경에 pip install -e로 설치)
    ├── mergekit/merge_methods/  # task_arithmetic, ties, dare 등 병합 구현
    └── pyproject.toml
```

### 실험을 진행하면 생기는 폴더 (모두 `memit/` 아래)

저장소에는 포함되어 있지 않으며, 아래 실험 순서를 따라 실행하면 만들어집니다.

| 경로 | 내용 |
|---|---|
| `data/` | MultiCounterFact(`multi_counterfact.json`), `data/stats/` 레이어 통계(MEMIT 실행 시 자동 다운로드), `data/preprocessing/normal/mcf_sampled_{N}.json` 샘플링 데이터 |
| `original_models/gpt2-xl/` | 원본 GPT2-XL (병합 상대) |
| `edited_models/` | 편집 모델 `edited_1_{N}_1/` |
| `merged/` | 병합 모델 `edited1{N}1_gpt2xl_{병합방법}_l{λ}[_d{density}]/` |
| `logs/hclt/2_{N÷2}/` | 평가 로그 (`memit_1_{N}.txt` baseline, `{병합모델명}.txt`)와 분석 결과 |

---

## ⚙️ 환경 설정 — 가상환경 2개가 필요합니다

**편집·평가는 `memit` 환경**, **병합은 `mergekit` 환경**에서 실행합니다. 두 라이브러리는 하나의 환경에 함께 설치할 수 없습니다.

| | memit 환경 | mergekit 환경 |
|---|---|---|
| Python | 3.9 | **≥ 3.10** (mergekit 요구사항) |
| torch | 1.12.1 | ≥ 2.0 |
| transformers | 4.23.1 | ≥ 4.45.2, **< 5** |
| pydantic | 1.9.2 | 2.10.x |
| 실행하는 코드 | `tester.py`, `sample_json.py`, `summary_result.py` … | `merge_runner.py` |

### 1) memit 환경

```bash
conda create -n memit python=3.9 -y
conda activate memit
pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r memit/memit_requirements.txt
```

### 2) mergekit 환경

```bash
conda create -n mergekit python=3.10 -y
conda activate mergekit
pip install -e ./mergekit "transformers<5"
```

> `transformers<5`를 반드시 지정하세요. 버전을 지정하지 않으면 transformers 5.x가 설치되는데, 이 경우 mergekit에서 `` PydanticUserError: `ConfiguredModuleArchitecture` is not fully defined `` 오류가 납니다.

**모든 명령은 `memit/` 디렉토리에서 실행합니다.** `edited_models/`, `merged/`, `logs/`, `data/`는 이 디렉토리를 기준으로 한 상대경로입니다.

---

## 🚀 실험 진행 순서

```
[1. 데이터 샘플링] → [2. MEMIT 편집·저장] → [3. 원본 모델과 병합] → [4. 평가] → [5. 결과 분석]
     memit 환경        memit 환경 (GPU)       mergekit 환경        memit 환경 (GPU)   memit 환경
```

아래 예시는 빠르게 테스트할 수 있도록 **지식 20개를 편집하는 실험**을 기준으로 합니다. 편집 개수를 바꾸려면 `--size`와 `--num_edits_list`의 숫자만 바꾸면 됩니다.

편집·평가는 모두 `python -u tester.py --mode <모드> --num_edits_list <N ...>`로 실행합니다. `--num_edits_list`에 값을 여러 개 주면 차례대로 실행합니다(예: `--num_edits_list 20 50 100`).

| `--mode` | 실행 함수 | 하는 일 | 단계 |
|---|---|---|---|
| `edit` | `edit_and_save()` | MEMIT 편집 → `edited_models/`에 저장 | 2 |
| `merge_test` (기본값) | `merge_test()` | MEMIT baseline 평가 + `merged/` 전체 모델 평가 | 4 |
| `org_test` | `org_test()` | 편집하지 않은 원본 모델 평가 → `logs/hclt/2_{N÷2}/origin_1_{N}_1.txt` | 참고용 |

### 1. 데이터 준비 (memit 환경)

```bash
cd memit
mkdir -p data/preprocessing/normal
wget -P data https://memit.baulab.info/data/dsets/multi_counterfact.json   # 20,877개

python sample_json.py --input data/multi_counterfact.json \
                      --output data/preprocessing/normal/mcf_sampled_20.json \
                      --size 20 --seed 42
```

`tester.py`는 `data/preprocessing/normal/mcf_sampled_{N}.json`을 읽습니다.

### 2. MEMIT 편집 후 저장 (memit 환경)

원본 GPT2-XL에 20개 지식을 MEMIT으로 한 번에 편집하고, 편집 모델을 `edited_models/`에 저장합니다.

```bash
python -u tester.py --mode edit --num_edits_list 20
# → edited_models/edited_1_20_1/
```

병합 상대인 원본 GPT2-XL을 `original_models/gpt2-xl/`에 저장해 둡니다.

```bash
python -c "from transformers import AutoModelForCausalLM as M, AutoTokenizer as T; \
M.from_pretrained('gpt2-xl').save_pretrained('original_models/gpt2-xl'); \
T.from_pretrained('gpt2-xl').save_pretrained('original_models/gpt2-xl')"
```

### 3. 원본 모델과 병합 (mergekit 환경)

`merge_runner.py`는 `--edited_models`로 지정한 편집 모델(`edited_models/` 안의 폴더 이름)을 `original_models/gpt2-xl`과 병합합니다. base model은 `gpt2-xl`(Task Vector의 기준)이며, 주어진 하이퍼파라미터의 **모든 조합(grid)** 에 대해 병합을 수행합니다.

```bash
conda activate mergekit

# 편집 모델 + 원본 모델
python merge_runner.py --edited_models edited_1_20_1 \
                       --merge_methods ties --lambdas 1.6 --densities 0.5
# → merged/edited1201_gpt2xl_ties_l1.6_d0.5/

# 하이퍼파라미터 grid 탐색 (병합 방법 4개 × λ 2개 → 병합 모델 8개)
python merge_runner.py --edited_models edited_1_20_1 \
    --merge_methods task_arithmetic ties dare_ties dare_linear \
    --lambdas 1.1 1.5 --densities 0.5
```

| 병합 방법 | 사용 파라미터 |
|---|---|
| `task_arithmetic` | `--lambdas` |
| `ties` | `--lambdas`, `--densities` |
| `dare_ties` | `--lambdas`, `--densities` |
| `dare_linear` | `--lambdas` |

- `--lambdas`(λ)는 병합할 때 파라미터 변화의 크기를 조절하고, `--densities`는 task vector에서 절댓값 기준 상위 몇 %를 남길지 정합니다(예: `0.3` → 상위 30%).
- 출력 폴더가 이미 있으면 그 조합은 건너뜁니다.
- 병합 모델 1개는 약 3GB(float16)입니다. grid를 넓히면 조합 수만큼 디스크를 사용합니다.
- GPT2-XL(약 1.5B) 병합은 CPU에서도 가능하며, GPU가 있으면 자동으로 사용합니다.

### 4. 성능 평가 (memit 환경)

`--mode merge_test`(기본값)는 다음 두 가지를 실행합니다.

1. **Baseline**: 같은 데이터 20개를 MEMIT으로 편집해 평가 → `logs/hclt/2_10/memit_1_20.txt`
2. **병합 모델**: `merged/`의 **모든** 모델을 불러와 같은 데이터로 평가 → `logs/hclt/2_10/{병합모델명}.txt`

로그가 이미 있는 모델은 건너뜁니다. 따라서 병합 모델을 추가한 뒤 다시 실행하면 새로 추가된 모델만 평가됩니다.

```bash
conda activate memit
python -u tester.py --mode merge_test --num_edits_list 20
# → logs/hclt/2_10/
```

> 로그 폴더명은 `2_{N÷2}` 규칙을 따릅니다. 따라서 20개 평가 결과는 `logs/hclt/2_10/`에 저장됩니다.

### 5. 결과 분석 (memit 환경)

```bash
# 병합 로그 집계 → baseline(memit_*.txt)보다 좋은 설정 추출 → 하이퍼파라미터 분석
python summary_result.py --logs_dir ./logs/hclt/2_10
# → logs/hclt/2_10/combined_results.txt, high_performance_methods.txt, hyperparameter_analysis_result.txt

# run별 메트릭 요약 (all_summary.py에 입력된 run별 수치를 표로 출력)
python all_summary.py --detailed --analysis
```

### 편집 개수를 늘려 가며 탐색하기

병합 방법과 하이퍼파라미터는 **단계적 확장 탐색**으로 고릅니다.

1. 편집 개수를 **20 → 50 → 100 → 250 → 500 → 1,000**으로 늘려 가며 1~5단계를 반복합니다.
2. 각 단계에서 병합 방법(Task Arithmetic, TIES, DARE-TIES, DARE-Linear)과 하이퍼파라미터 조합을 grid로 탐색합니다.
3. MEMIT보다 성능이 좋은 비율(성공률)이 기준 이상인 조합만 다음 단계로 넘깁니다. `summary_result.py`의 `hyperparameter_analysis_result.txt`가 이 성공률을 집계합니다.
4. 최종적으로 **TIES**를 선택해 **10,000개 편집**에서 MEMIT과 비교합니다.

---

## 📊 평가 지표

| 지표 | 의미 |
|---|---|
| Score | Efficacy, Generalization, Specificity의 조화평균 |
| Efficacy | 편집한 프롬프트에서 새 target을 맞히는 비율 |
| Generalization (Paraphrase) | 바꿔 쓴 프롬프트에서도 편집이 반영되는 비율 |
| Specificity (Neighborhood) | 편집과 무관한 이웃 지식이 보존되는 비율 |
| Fluency / Consistency | 생성 문장의 유창성(n-gram entropy) / 일관성 |

---

## ✅ 실행 검증 범위

이 README의 명령은 Windows CPU 환경에서 아래 범위까지 확인했습니다.

| 단계 | 검증 내용 | 결과 |
|---|---|---|
| 환경 | `memit_requirements.txt` 설치 / `pip install -e ./mergekit "transformers<5"` 설치 | ✅ |
| 환경 | 하나의 환경에 mergekit 설치 (Python 3.9) | ❌ 불가능함을 확인 (mergekit은 Python 3.10 이상 필요) |
| 1 | MultiCounterFact 다운로드 + `sample_json.py` | ✅ |
| 3 | 소형 GPT-2로 편집 모델 + 원본 모델 병합, 병합 방법 4개 × λ 2개 grid 탐색 | ✅ |
| 3→4 | mergekit 환경에서 만든 병합 모델을 memit 환경(transformers 4.23.1)에서 로드하고 forward 실행 | ✅ |
| 5 | `summary_result.py`, `all_summary.py` | ✅ (형식을 맞춘 가짜 로그로 확인) |
| 2, 4 | GPT2-XL 실제 MEMIT 편집 및 평가 | ⚠️ 미검증 (CUDA GPU 필요, `ModelEditor`가 `.cuda()`를 직접 호출) |

> `dare_ties`는 task vector의 일부를 무작위로 drop하므로 같은 설정으로 병합해도 실행할 때마다 결과가 조금씩 다릅니다.

---

## 🙏 Acknowledgements

- [MEMIT: Mass-Editing Memory in a Transformer](https://github.com/kmeng01/memit) (Meng et al., ICLR 2023)
- [mergekit](https://github.com/arcee-ai/mergekit) (Arcee AI)

각 하위 폴더의 `LICENSE`를 따릅니다.
