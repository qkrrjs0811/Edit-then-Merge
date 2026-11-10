# MEMIT with Model Merging Framework

다중 지식 편집에서 모델 병합으로 성능을 개선하는 연구의 편집·평가 코드입니다.

> 📖 **환경 설정, 전체 폴더 구조, 단계별 실행 방법은 [루트 README](../README.md)를 참고하세요.**
> 이 폴더는 [MEMIT](https://github.com/kmeng01/memit) 원본 코드에 연구용 코드(`falcon/`, `merge_runner.py`, `summary_result.py` 등)를 추가한 것입니다.

## Quick Start

모든 명령은 이 `memit/` 디렉토리에서 실행합니다. 가상환경은 **memit**(편집·평가)과 **mergekit**(병합) 두 개가 필요합니다.

```bash
# [memit]    1. 데이터 샘플링 (20개)
python sample_json.py --input data/multi_counterfact.json \
                      --output data/preprocessing/normal/mcf_sampled_20.json --size 20

# [memit]    2. 20개를 MEMIT으로 한 번에 편집 → edited_models/edited_1_20_1/
python -u -m falcon.tester --mode edit --identical_nums 1 --num_edits_list 20

# [mergekit] 3. 편집 모델 + 원본 모델 병합 → merged/edited1201_gpt2xl_ties_l1.6_d0.5/
python merge_runner.py --model_dirs edited_1_20_1 gpt2-xl \
                       --merge_methods ties --lambdas 1.6 --densities 0.5

# [memit]    4. MEMIT baseline + 병합 모델 평가 → logs/hclt/2_10/
python -u -m falcon.tester --mode merge_test --identical_nums 1 --num_edits_list 20 --test_type all

# [memit]    5. 결과 분석
python summary_result.py --logs_dir ./logs/hclt/2_10 --version e2o
```

## 지원하는 병합 방법

`task_arithmetic`, `ties`, `dare_ties`, `dare_linear`, `breadcrumbs`, `breadcrumbs_ties`, `della`, `della_linear`, `sce`
(방법별 하이퍼파라미터는 루트 README의 "3. 모델 병합" 참고)

## 원본 MEMIT

이 폴더에는 MEMIT 원본 코드 중 이 연구에 필요한 부분(`memit/`, `rome/`, `dsets/`, `experiments/`의 평가 함수, `util/`)만 남겨 두었습니다. causal tracing, FT/MEND 베이스라인 등 원본의 전체 기능은 [kmeng01/memit](https://github.com/kmeng01/memit)을 참고하세요.
