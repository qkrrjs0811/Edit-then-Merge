# Edit-then-Merge

**지식 편집(MEMIT) 모델과 원본/편집 모델을 모델 병합(Model Merging)으로 결합해 대규모 지식 편집 성능을 개선하는 연구**의 실험 코드입니다.

- HCLT 2025 (한글 및 한국어 정보처리 학술대회)
- 한국정보과학회 2026 저널

기본 아이디어는 다음과 같습니다.

1. 편집할 지식(CounterFact)을 여러 batch로 나누고, 각 batch를 **원본 GPT2-XL에 독립적으로 MEMIT 편집**하여 모델을 따로 저장합니다.
2. 저장된 편집 모델들(또는 편집 모델 + 원본 모델)을 **mergekit의 Task Vector 기반 병합 방법**(TIES, DARE, DELLA 등)으로 하나로 합칩니다.
3. 병합 모델의 편집 성능을 **전체 데이터를 한 번에 MEMIT으로 편집한 모델(baseline)** 과 비교합니다.

---

## 📁 전체 폴더 구조

```
Edit-then-Merge/
├── README.md                  # (이 문서)
├── memit/                     # 지식 편집 + 평가 + 분석 (memit 환경, 이 폴더에서 실행)
│   ├── falcon/                # ★ 연구용 편집/평가 파이프라인
│   │   ├── tester.py          #   실행 진입점 (--mode edit | merge_test | eval_edited | org_test)
│   │   ├── model_editor.py    #   ModelEditor: 편집·예측·평가·로그·모델 저장
│   │   └── falcon_util.py     #   파일 I/O 등 유틸
│   ├── merge_runner.py        # ★ mergekit으로 편집 모델 병합 (mergekit 환경, 하이퍼파라미터 grid 탐색)
│   ├── sample_json.py         # ★ CounterFact에서 N개 랜덤 샘플링
│   ├── summary_result.py      # ★ logs/hclt/* 병합 결과 집계 + 하이퍼파라미터 분석
│   ├── all_summary.py         # ★ 논문 표 수치(run별 메트릭) 요약 출력 (추가 실험)
│   ├── pruning.py             # ★ 편집 모델 파라미터 top-k% 희소화 (추가 실험)
│   ├── part_edit_test.py      # ★ 로그의 특정 index 구간 정확도 계산 (추가 실험)
│   ├── memit/                 # MEMIT 편집 알고리즘 (원본)
│   ├── rome/                  # ROME (원본, MEMIT이 layer_stats·repr_tools를 사용)
│   ├── dsets/                 # CounterFact 등 데이터셋 로더 (원본)
│   ├── experiments/           # 편집 성능 평가 함수 eval_utils_*.py, summarize.py (원본)
│   ├── util/                  # nethook, generate 등 유틸 (원본)
│   ├── hparams/MEMIT/gpt2-xl.json  # MEMIT 하이퍼파라미터 (편집 레이어 13~17)
│   ├── globals.yml            # 결과/데이터/통계 경로 설정
│   ├── scripts/memit.yml      # memit conda 환경 파일
│   └── memit_requirements.txt # memit 환경 pip 패키지 목록
└── mergekit/                  # 모델 병합 라이브러리 (mergekit 환경에 pip install -e로 설치)
    ├── mergekit/merge_methods/  # task_arithmetic, ties, dare, della, sce 등 병합 구현
    └── pyproject.toml
```

★ 표시는 이 연구를 위해 새로 작성한 코드입니다. 나머지는 [MEMIT](https://github.com/kmeng01/memit), [mergekit](https://github.com/arcee-ai/mergekit) 원본 코드입니다.

> **mergekit 버전**: 번들된 `mergekit/`은 upstream 커밋 [`8440b18`](https://github.com/arcee-ai/mergekit/commit/8440b18ec8594085a1b57e1858e8e6f2e74e4fb4)(2025-04-20)에, 이후 upstream에서 나온 DELLA ε 전달 버그 수정([#573](https://github.com/arcee-ai/mergekit/pull/573), `generalized_task_arithmetic.py`)을 반영한 상태입니다. 이와 정확히 같은 공식 릴리스가 없기 때문에, 논문 결과를 재현하려면 PyPI 버전 대신 **번들된 코드를 사용하세요.**

### 실험을 진행하면 생기는 폴더 (모두 `memit/` 아래)

저장소에는 포함되어 있지 않으며, 아래 실험 순서를 따라 실행하면 만들어집니다.

| 경로 | 내용 |
|---|---|
| `data/` | CounterFact(`multi_counterfact.json`), `data/stats/` 레이어 통계(MEMIT 실행 시 자동 다운로드), `data/preprocessing/normal/mcf_sampled_{N}.json` 샘플링 데이터 |
| `original_models/gpt2-xl/` | 원본 GPT2-XL (편집 모델 + 원본 모델 병합 시 사용) |
| `edited_models/` | batch별 독립 편집 모델 `edited_{batch 수}_{batch당 편집 수}_{batch 번호}/` |
| `merged/` | 병합 모델 `{모델명들}_{병합방법}_l{λ}_d{density}[_e{ε}]/` |
| `logs/hclt/{2}_{batch당 편집 수}/` | 평가 로그 (`memit_*.txt` baseline, `{병합모델명}.txt`)와 분석 결과 |

---

## ⚙️ 환경 설정 — 가상환경 2개가 필요합니다

**편집·평가는 `memit` 환경**, **병합은 `mergekit` 환경**에서 실행합니다. 두 라이브러리는 하나의 환경에 함께 설치할 수 없습니다.

| | memit 환경 | mergekit 환경 |
|---|---|---|
| Python | 3.9 | **≥ 3.10** (mergekit 요구사항) |
| torch | 1.12.1 | ≥ 2.0 |
| transformers | 4.23.1 | ≥ 4.45.2, **< 5** |
| pydantic | 1.9.2 | 2.10.x |
| 실행하는 코드 | `falcon/`, `sample_json.py`, `summary_result.py`, `pruning.py` … | `merge_runner.py` |

### 1) memit 환경 (편집·평가, **NVIDIA GPU 필요**)

```bash
conda create -n memit python=3.9 -y
conda activate memit
pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r memit/memit_requirements.txt
```

> `memit_requirements.txt`에는 `safetensors==0.4.5`가 포함되어 있습니다. mergekit은 병합 결과를 `model.safetensors`로 저장하기 때문에, 이 패키지가 없으면 transformers 4.23.1이 병합 모델을 읽지 못합니다(`no file named pytorch_model.bin` 오류). conda 환경 파일(`memit/scripts/memit.yml`)을 쓰는 경우에도 같은 패키지가 들어 있습니다.

### 2) mergekit 환경 (병합, CPU로도 실행 가능)

```bash
conda create -n mergekit python=3.10 -y
conda activate mergekit
pip install -e ./mergekit "transformers<5"
```

> `transformers<5`를 반드시 지정하세요. 버전을 지정하지 않으면 transformers 5.x가 설치되는데, 이 경우 mergekit 0.1.2에서 `` PydanticUserError: `ConfiguredModuleArchitecture` is not fully defined `` 오류가 납니다.

**모든 명령은 `memit/` 디렉토리에서 실행합니다.** `edited_models/`, `merged/`, `logs/`, `data/`는 이 디렉토리를 기준으로 한 상대경로입니다.

---

## 🚀 실험 진행 순서

```
[1. 데이터 샘플링] → [2. batch별 독립 편집·저장] → [3. 모델 병합] → [4. 평가] → [5. 결과 분석]
     memit 환경          memit 환경 (GPU)        mergekit 환경   memit 환경 (GPU)   memit 환경
```

아래 예시는 **3000개 지식을 1500개씩 2 batch로 나눈 실험**을 기준으로 합니다.

편집·평가는 모두 `python -u -m falcon.tester --mode <모드>`로 실행합니다.

| `--mode` | 실행 함수 | 하는 일 | 단계 |
|---|---|---|---|
| `edit` | `edit_and_save()` | batch별 독립 MEMIT 편집 → `edited_models/`에 저장 | 2 |
| `merge_test` (기본값) | `hclt_merge_test()` | MEMIT baseline 평가 + `merged/` 전체 모델 평가 | 4 |
| `eval_edited` | `evaluate_edited_model()` | `edited_models/*_sparse*` 희소화 모델 평가 | 4 (추가 실험) |
| `org_test` | `hclt_edit_and_save()` | 편집하지 않은 원본 모델의 예측 로그 기록 | 참고용 |

### 1. 데이터 준비 (memit 환경)

```bash
cd memit
mkdir -p data/preprocessing/normal
wget -P data https://memit.baulab.info/data/dsets/multi_counterfact.json   # 20,877개

python sample_json.py --input data/multi_counterfact.json \
                      --output data/preprocessing/normal/mcf_sampled_3000.json \
                      --size 3000 --seed 42
```

`falcon/tester.py`는 `data/preprocessing/normal/mcf_sampled_{batch 수 × batch당 편집 수}.json`을 읽습니다.

### 2. batch별 독립 편집 후 저장 (memit 환경)

데이터를 `num_edits`개씩 batch로 나누고, **각 batch를 원본 GPT2-XL에 독립적으로 MEMIT 편집**합니다. batch마다 가중치를 원래대로 복원하므로 각 편집은 서로 영향을 주지 않으며, 결과는 `edited_models/`에 저장됩니다.

```bash
python -u -m falcon.tester --mode edit --identical_nums 2 --num_edits_list 1500
# → edited_models/edited_2_1500_1/, edited_models/edited_2_1500_2/
```

- `--identical_nums`: batch 수
- `--num_edits_list`: batch당 편집 수 (값을 여러 쌍 주면 차례대로 실행)

편집 모델과 원본 모델을 병합(e2o)하려면 원본 GPT2-XL을 `original_models/gpt2-xl/`에 저장해 둡니다.

```bash
python -c "from transformers import AutoModelForCausalLM as M, AutoTokenizer as T; \
M.from_pretrained('gpt2-xl').save_pretrained('original_models/gpt2-xl'); \
T.from_pretrained('gpt2-xl').save_pretrained('original_models/gpt2-xl')"
```

### 3. 모델 병합 (mergekit 환경)

`merge_runner.py`는 `edited_models/`와 `merged/`에서 `--model_dirs`와 **부분 일치**하는 모델을 찾아 병합합니다. base model은 `gpt2-xl`(Task Vector의 기준)이며, 주어진 하이퍼파라미터의 **모든 조합(grid)** 에 대해 병합을 수행합니다.

```bash
conda activate mergekit

# 편집 모델 + 편집 모델 (e2e)
python merge_runner.py --model_dirs edited_2_1500_1 edited_2_1500_2 \
                       --merge_methods ties --lambdas 1.6 --densities 0.5
# → merged/edited215001_edited215002_ties_l1.6_d0.5/

# 편집 모델 + 원본 모델 (e2o)
python merge_runner.py --model_dirs edited_2_1500_2 gpt2-xl \
                       --merge_methods ties --lambdas 1.6 --densities 0.5
# → merged/edited215002_gpt2xl_ties_l1.6_d0.5/

# 하이퍼파라미터 grid 탐색
python merge_runner.py --model_dirs edited_2_1500_1 edited_2_1500_2 \
    --merge_methods task_arithmetic ties dare_ties dare_linear della della_linear \
    --lambdas 1.1 1.3 1.5 --densities 0.1 0.3 0.5 0.7 0.9 --epsilons 0.1 0.2 0.3 0.4
```

| 병합 방법 | 사용 파라미터 |
|---|---|
| `task_arithmetic`, `dare_linear` | `--lambdas` |
| `ties`, `dare_ties` | `--lambdas`, `--densities` |
| `breadcrumbs`, `breadcrumbs_ties` | `--lambdas`, `--densities`, `--gammas` |
| `della`, `della_linear` | `--lambdas`, `--densities`, `--epsilons` (density ± ε가 (0, 1)을 벗어나는 조합은 자동으로 건너뜀) |
| `sce` | `--lambdas`, `--densities`, `--select_topk` |

- 출력 폴더가 이미 있으면 그 조합은 건너뜁니다.
- GPT2-XL(약 1.5B) 병합은 CPU에서도 가능하며, GPU가 있으면 자동으로 사용합니다.

### 4. 성능 평가 (memit 환경)

`--mode merge_test`(기본값)는 다음 두 가지를 실행합니다.

1. **Baseline**: 같은 데이터 전체를 MEMIT으로 한 번에 편집해 평가 → `logs/hclt/2_1500/memit_1_3000.txt`
2. **병합 모델**: `merged/`의 **모든** 모델을 불러와 같은 데이터로 평가 → `logs/hclt/2_1500/{병합모델명}.txt`

로그가 이미 있는 모델은 건너뜁니다. 따라서 병합 모델을 추가한 뒤 다시 실행하면 새로 추가된 모델만 평가됩니다.

```bash
conda activate memit
# 3000개 전체로 평가 → logs/hclt/2_1500/
python -u -m falcon.tester --mode merge_test --identical_nums 1 --num_edits_list 3000 --test_type all

# 한 batch만 평가: --test_type each --test_batch_size 0(앞 절반) / 1(뒤 절반)
python -u -m falcon.tester --identical_nums 1 --num_edits_list 3000 --test_type each --test_batch_size 1
```

> 로그 폴더명은 `{identical_num×2}_{num_edits÷2}` 규칙을 따릅니다. 따라서 3000개 평가 결과는 "1500개씩 2 batch" 실험 폴더인 `logs/hclt/2_1500/`에 저장됩니다.

**희소화(pruning) 추가 실험**

```bash
python pruning.py --k_percent 10                       # → edited_models/edited_2_1500_{1,2}_sparse_k10.0/
python -u -m falcon.tester --mode eval_edited --identical_nums 2 --num_edits_list 1500
# (평가 대상: edited_2_1500_1_sparse* — 1번 batch 모델)
```

### 5. 결과 분석 (memit 환경)

```bash
# 병합 로그 집계 → baseline(memit_*.txt)보다 좋은 설정 추출 → 하이퍼파라미터 분석
python summary_result.py --logs_dir ./logs/hclt/2_1500 --version all   # all | e2o | e2e
# → logs/hclt/2_1500/combined_results.txt, high_performance_methods.txt, hyperparameter_analysis_result.txt
#   (--version을 바꿔 실행하면 같은 파일을 덮어씁니다)

# 논문 표 수치 요약 (all_summary.py에 입력된 run별 수치를 표로 출력)
python all_summary.py --detailed --analysis

# 특정 index 구간 정확도 (스크립트 안의 log_file_path를 수정한 뒤 실행)
python part_edit_test.py --start_idx 0 --end_idx 1499
```

---

## 📊 평가 지표

| 지표 | 의미 |
|---|---|
| Efficacy | 편집한 프롬프트에서 새 target을 맞히는 비율 |
| Generalization (Paraphrase) | 바꿔 쓴 프롬프트에서도 편집이 반영되는 비율 |
| Specificity (Neighborhood) | 편집과 무관한 이웃 지식이 보존되는 비율 |
| Fluency / Consistency | 생성 문장의 유창성(n-gram entropy) / 일관성 |

---

## ✅ 실행 검증 범위

이 README의 명령은 Windows CPU 환경에서 새로 만든 가상환경 2개(Python 3.9 / 3.10)로 아래 범위까지 확인했습니다.

| 단계 | 검증 내용 | 결과 |
|---|---|---|
| 환경 | `memit_requirements.txt` 설치 / `pip install -e ./mergekit "transformers<5"` 설치 | ✅ |
| 환경 | 하나의 환경에 mergekit 설치 (Python 3.9) | ❌ 불가능함을 확인 (mergekit은 Python 3.10 이상 필요) |
| 1 | CounterFact 다운로드 + `sample_json.py` | ✅ |
| 2 | `--mode edit` 분기, 데이터 경로, 저장 이름(`edited_2_1500_*`) | ✅ (`ModelEditor`를 mock으로 대체해 확인) |
| 3 | 소형 GPT-2로 e2e·e2o 병합 및 위 grid 명령 실행 (9개 병합 방법 모두) | ✅ |
| 3→4 | mergekit 환경에서 만든 병합 모델을 memit 환경(transformers 4.23.1)에서 로드하고 forward 실행 | ✅ |
| 4 | `pruning.py` 희소화 모델 생성 | ✅ |
| 5 | `summary_result.py`(all/e2o/e2e), `all_summary.py` | ✅ (형식을 맞춘 가짜 로그로 확인) |
| 2, 4 | GPT2-XL 실제 MEMIT 편집 및 평가 | ⚠️ 미검증 (CUDA GPU 필요, `ModelEditor`가 `.cuda()`를 직접 호출) |

정리 후 `memit/`에 남은 Python 파일은 모두 위 실험 흐름에서 실제로 import되는 것을 확인했습니다.

### 논문 실험 당시 코드와 달라진 점

공개용으로 정리하면서 **실험 로직(편집·병합·평가 방식)은 그대로 두고**, 다른 환경에서도 실행되도록 아래 부분만 수정했습니다.

| 파일 | 변경 내용 |
|---|---|
| `memit/falcon/tester.py` | `--mode` 인자 추가 (이전에는 `__main__`의 주석을 바꿔 실행할 함수를 골랐음). `edit_and_save()`가 `--identical_nums`/`--num_edits_list`를 받고 `identical_num`을 `ModelEditor`에 전달하도록 수정. `home_dir` 절대 경로 → `memit/` 기준 상대 경로 |
| `memit/falcon/model_editor.py` | 확장 평가 로그 경로를 절대 경로에서 `./logs/logs_*`로 변경. 사용하지 않는 FT/MEND 알고리즘 import 제거 (`ALG_DICT`에는 ROME, MEMIT만 남음) |
| `memit/merge_runner.py` | `edited_models/`, `merged/` 폴더 자동 생성. `--model_dirs`를 생략했을 때 나던 오류 수정 |
| `memit/pruning.py` | `--edited_models_dir` 기본값을 `memit/edited_models`로 변경 |
| `memit/memit_requirements.txt` | conda 로컬 경로로 되어 있던 `typing_extensions` 줄 수정. `safetensors`, `tabulate` 추가 |
| `memit/scripts/memit.yml` | `safetensors` 추가 |

**실험에 쓰이지 않아 제외한 코드**

- 이전 실험용 코드: `falcon/data_preprocessor.py`, `find_with_wiki.py`, `log_processor.py`, `model_edit_runner.py`, `gpt2_model.py`
- 실행 스크립트: `memit/*.sh`, `scripts/` 안의 셸 스크립트(`memit.yml` 제외)
- MEMIT 원본 중 이 연구에서 쓰지 않는 부분: `baselines/`(FT, MEND), `notebooks/`, `experiments/evaluate·sweep·causal_trace·demo`, GPT2-XL MEMIT 외의 `hparams/`, `CITATION.cff`
- 그 밖: `compare_embeddings_in_3d.ipynb`, 논문 그림 생성 스크립트 `visualization/`

---

## 🙏 Acknowledgements

- [MEMIT: Mass-Editing Memory in a Transformer](https://github.com/kmeng01/memit) (Meng et al., ICLR 2023)
- [mergekit](https://github.com/arcee-ai/mergekit) (Arcee AI)

각 하위 폴더의 `LICENSE`를 따릅니다.
