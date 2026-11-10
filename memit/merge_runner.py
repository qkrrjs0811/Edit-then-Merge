import torch
import argparse
from pathlib import Path

from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions, run_merge

# Merge 옵션
LORA_MERGE_CACHE = "/tmp"
COPY_TOKENIZER = True
LAZY_UNPICKLE = False
LOW_CPU_MEMORY = False

BASE_MODEL = "gpt2-xl"  # Task Vector의 기준
ORIGINAL_MODEL_DIR = "original_models/gpt2-xl"  # 병합 상대인 원본 모델

# 각 방법론에 필요한 파라미터 정의
merge_method_params = {
    "task_arithmetic": ["lambda"],
    "ties": ["lambda", "density"],
    "dare_ties": ["lambda", "density"],
    "dare_linear": ["lambda"],
}


def create_merge_conf(merge_method, params, edited_model_dir):
    # 편집 모델을 원본 모델(base_model)과 병합
    config = {
        "base_model": BASE_MODEL,
        "dtype": "float16",
        "merge_method": merge_method,
        "models": [
            {"model": edited_model_dir, "parameters": {"weight": 1.0}},
            {"model": ORIGINAL_MODEL_DIR, "parameters": {"weight": 1.0}},
        ],
        "parameters": params
    }
    return MergeConfiguration.model_validate(config)


def main():
    parser = argparse.ArgumentParser(description='편집 모델을 원본 모델과 병합 (병합 방법 × 하이퍼파라미터 grid)')
    parser.add_argument('--edited_models', type=str, nargs='+', required=True,
                        help='edited_models/ 안의 편집 모델 이름 (예: edited_1_20_1)')
    parser.add_argument('--merge_methods', type=str, nargs='+', required=True,
                        choices=list(merge_method_params.keys()), help='병합 방법')
    parser.add_argument('--lambdas', type=float, nargs='+', default=[0.1], help='lambda 값 목록')
    parser.add_argument('--densities', type=float, nargs='+', default=[0.1], help='density 값 목록')

    args = parser.parse_args()

    Path('./merged').mkdir(exist_ok=True)

    for edited_name in args.edited_models:
        edited_model_dir = Path('./edited_models') / edited_name
        if not edited_model_dir.is_dir():
            print(f"[Error] 편집 모델이 없습니다: {edited_model_dir}")
            continue

        # edited_1_20_1 -> edited1201
        combined_name = f"{edited_name.replace('_', '')}_gpt2xl"

        for merge_method in args.merge_methods:
            params_list = merge_method_params[merge_method]

            # 파라미터 조합 생성
            param_combinations = [{"lambda": lambda_val} for lambda_val in args.lambdas]
            if "density" in params_list:
                param_combinations = [
                    {**comb, "density": density} for comb in param_combinations for density in args.densities
                ]

            for params in param_combinations:
                param_str = f"_l{params['lambda']}"
                if "density" in params:
                    param_str += f"_d{params['density']}"

                OUTPUT_PATH = f"./merged/{combined_name}_{merge_method}{param_str}"

                if Path(OUTPUT_PATH).exists():
                    print(f"\n[Warning] 병합된 모델이 이미 존재합니다: {OUTPUT_PATH}")
                    print("삭제 후 다시 실행하거나 다른 이름을 입력하세요.")
                    continue

                print(f"\n[MergeRunner] {edited_model_dir} + {ORIGINAL_MODEL_DIR}")
                print(f"[MergeRunner] merge_method: {merge_method}, params: {params}")
                print(f"[MergeRunner] 병합 결과 경로: {OUTPUT_PATH}")

                run_merge(
                    create_merge_conf(merge_method, params, str(edited_model_dir)),
                    out_path=OUTPUT_PATH,
                    options=MergeOptions(
                        lora_merge_cache=LORA_MERGE_CACHE,
                        cuda=torch.cuda.is_available(),
                        copy_tokenizer=COPY_TOKENIZER,
                        lazy_unpickle=LAZY_UNPICKLE,
                        low_cpu_memory=LOW_CPU_MEMORY,
                    ),
                )
                print(f"\n[MergeRunner] 병합 완료! 모델이 {OUTPUT_PATH}에 저장되었습니다.")


if __name__ == "__main__":
    main()
