import torch
import yaml
import argparse
import sys
from pathlib import Path

# mergekit이 설치된 경로 추가 (import 오류 방지)
# sys.path.append(str(Path(__file__).resolve().parent / "mergekit_local"))  # 필요시 수정

from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions, run_merge

# Merge 옵션
LORA_MERGE_CACHE = "/tmp"
COPY_TOKENIZER = True
LAZY_UNPICKLE = False
LOW_CPU_MEMORY = False

# 각 방법론에 필요한 파라미터 정의
merge_method_params = {
    "task_arithmetic": ["lambda"],
    "ties": ["lambda", "density"],
    "dare_ties": ["lambda", "density"],
    "dare_linear": ["lambda"],
    "breadcrumbs": ["lambda", "density", "gamma"],
    "breadcrumbs_ties": ["lambda", "density", "gamma"],
    "della": ["lambda", "density", "epsilon"],
    "della_linear": ["lambda", "density", "epsilon"],
    "sce": ["lambda", "density", "select_topk"]
}

def create_merge_conf(base_model, merge_method, params, model_dirs):
    # 이미 병합된 모델들을 재병합할 때는 첫 번째 모델을 base_model로 사용
    # merged_models = [d for d in model_dirs if 'merged' in str(d)]
    
    # 원본 모델들만 있는 경우 (기존 방식)
    actual_base_model = base_model
    models_to_merge = model_dirs
    print(f"[Info] 일반 병합 모드: base_model={actual_base_model}, 병합할 모델들={models_to_merge}")
    
    config = {
        "base_model": actual_base_model,
        "dtype": "float16", 
        "merge_method": merge_method,
        "models": [{"model": model_dir, "parameters": {"weight": 1.0}} for model_dir in models_to_merge],
        "parameters": params
    }
    with open("merge_conf.yml", "w", encoding="utf-8") as f:
        yaml.dump(config, f)

def main():
    parser = argparse.ArgumentParser(description='Merge models with different methods and hyperparameters.')
    parser.add_argument('--model_dirs', type=str, nargs='+', required=False, help='List of model directories to merge', default=None)
    parser.add_argument('--merge_methods', type=str, nargs='+', required=True, help='List of merge methods')
    parser.add_argument('--lambdas', type=float, nargs='+', required=False, help='List of lambda values', default=[0.1])
    parser.add_argument('--densities', type=float, nargs='+', required=False, help='List of density values', default=[0.1])
    parser.add_argument('--epsilons', type=float, nargs='+', required=False, help='List of epsilon values', default=[0.1])
    parser.add_argument('--gammas', type=float, nargs='+', required=False, help='List of gamma values', default=[0.1])
    parser.add_argument('--select_topk', type=int, nargs='+', required=False, help='List of select_topk values', default=[1])

    args = parser.parse_args()

    # 1. './edited_models'와 './merged' 폴더 내의 모든 디렉토리 경로를 리스트로 만듦
    Path('./edited_models').mkdir(exist_ok=True)
    Path('./merged').mkdir(exist_ok=True)
    all_model_dirs = [str(p) for p in Path('./edited_models').iterdir() if p.is_dir()]
    all_model_dirs += [str(p) for p in Path('./merged').iterdir() if p.is_dir()]

    # 2. 만약 'gpt2-xl'이 args.model_dirs에 포함되어 있으면, 'original_models/gpt2-xl' 경로도 추가
    if args.model_dirs and 'gpt2-xl' in args.model_dirs:
        all_model_dirs.append('original_models/gpt2-xl')
    
    # 3. 사용자가 --model_dirs 옵션을 명시적으로 줬다면,
    if args.model_dirs:
        # all_model_dirs 중에서 이름이 args.model_dirs에 포함된 것만 필터링 (부분 매칭 지원)
        # 사용자가 지정한 순서 유지
        model_dirs = []
        for user_name in args.model_dirs:
            matched = [d for d in all_model_dirs if user_name in Path(d).name]
            for match in matched:
                if match not in model_dirs:  # 중복 제거하면서 순서 유지
                    model_dirs.append(match)
    else:
        # 사용자가 --model_dirs 옵션을 주지 않았다면, 모든 모델 경로를 사용
        model_dirs = all_model_dirs

    print("================================================")
    print(f"all_model_dirs: {all_model_dirs}")
    print(f"model_dirs: {model_dirs}")
    print("================================================")


    # 각 방법론에 대한 파라미터 조합 생성
    for merge_method in args.merge_methods:
        params_list = merge_method_params[merge_method]
        
        # Initialize the parameter combinations
        param_combinations = [{}]
        
        # Iterate over each parameter type and update combinations
        if "lambda" in params_list:
            param_combinations = [
                {**comb, "lambda": lambda_val} for comb in param_combinations for lambda_val in args.lambdas
            ]
        if "density" in params_list:
            param_combinations = [
                {**comb, "density": density} for comb in param_combinations for density in args.densities
            ]
        if "epsilon" in params_list:
            param_combinations = [
                {**comb, "epsilon": epsilon} for comb in param_combinations for epsilon in args.epsilons
            ]
        if "gamma" in params_list:
            param_combinations = [
                {**comb, "gamma": gamma} for comb in param_combinations for gamma in args.gammas
            ]
        if "select_topk" in params_list:
            param_combinations = [
                {**comb, "select_topk": select_topk} for comb in param_combinations for select_topk in args.select_topk
            ]

        # Iterate over the generated parameter combinations
        for params in param_combinations:
            # Check for valid parameter combinations
            if merge_method in ['della', 'della_linear'] and ("density" in params and "epsilon" in params) and (params["density"] + params["epsilon"] >= 1 or params["density"] - params["epsilon"] <= 0):
                print(f"Error: Invalid combination for {merge_method} with density {params['density']} and epsilon {params['epsilon']}")
                continue

            create_merge_conf("gpt2-xl", merge_method, params, model_dirs)
            print("########################################################")
            print("================================================")
            print(f"병합할 모델들: {model_dirs}")
            print("================================================")
            
            # 병합할 모델들이 이미 병합된 모델들인지 확인
            merged_models = [d for d in model_dirs if 'merged' in str(d)]
            original_models = [d for d in model_dirs if 'original_models' in str(d)]
            edited_models = [d for d in model_dirs if 'edited_models' in str(d)]
            
            print(f"병합된 모델들: {merged_models}")
            print(f"원본 모델들: {original_models}")
            print(f"편집된 모델들: {edited_models}")
            
            # 출력 경로 생성 - 사용자가 지정한 순서대로 모델 이름 조합
            model_names = []
            if args.model_dirs:
                # 사용자가 지정한 순서 사용
                for user_name in args.model_dirs:
                    # 해당 user_name에 매칭되는 실제 모델 찾기
                    matched_models = [d for d in model_dirs if user_name in Path(d).name]
                    for model_dir in matched_models:
                        model_name = Path(model_dir).name
                        if model_name == 'gpt2-xl':
                            model_names.append('gpt2xl')
                        elif model_name.startswith('remerged_'):
                            # 이미 재병합된 모델의 경우 중요한 부분만 취함
                            parts = model_name.split('_')
                            if len(parts) >= 2:
                                # remerged_edited2101_... 에서 edited2101 추출
                                model_names.append(parts[1])
                            else:
                                model_names.append(model_name.replace('remerged_', ''))
                        elif model_name.startswith('merged_'):
                            # 이미 병합된 모델의 경우 중요한 부분만 취함
                            parts = model_name.split('_')
                            if len(parts) >= 3:
                                model_names.append(f"{parts[1]}{parts[2]}")
                            else:
                                model_names.append(model_name.replace('merged_', ''))
                        else:
                            # edited_2_10_1 -> edited2101 형태로 변환
                            clean_name = model_name.replace('_', '')
                            model_names.append(clean_name)
            else:
                # 기존 방식 (모든 모델 사용)
                for model_dir in model_dirs:
                    model_name = Path(model_dir).name
                    if model_name == 'gpt2-xl':
                        model_names.append('gpt2xl')
                    elif model_name.startswith('merged_'):
                        # 이미 병합된 모델의 경우 중요한 부분만 취함
                        parts = model_name.split('_')
                        if len(parts) >= 3:
                            model_names.append(f"{parts[1]}{parts[2]}")
                        else:
                            model_names.append(model_name.replace('merged_', ''))
                    else:
                        model_names.append(model_name.replace('_', ''))
            
            combined_name = '_'.join(model_names[:3])  # 너무 길어지지 않도록 최대 3개까지만
            
            # 파라미터를 간결한 형태로 변환
            param_str = ""
            for key, value in params.items():
                # 파라미터 이름을 약어로 변환
                if key == "lambda":
                    param_str += f"_l{value}"
                elif key == "density":
                    param_str += f"_d{value}"
                elif key == "epsilon":
                    param_str += f"_e{value}"
                elif key == "gamma":
                    param_str += f"_g{value}"
                elif key == "select_topk":
                    param_str += f"_k{value}"
                else:
                    param_str += f"_{key}{value}"
            
            OUTPUT_PATH = f"./merged/{combined_name}_{merge_method}{param_str}"
            
            if Path(OUTPUT_PATH).exists():
                print(f"\n[Warning] 병합된 모델이 이미 존재합니다: {OUTPUT_PATH}")
                print("삭제 후 다시 실행하거나 다른 이름을 입력하세요.")
                continue
            
            print(f"\n[MergeRunner] merge_method: {merge_method}, params: {params}")
            print(f"[MergeRunner] 병합 결과 경로: {OUTPUT_PATH}")
            
            with open("merge_conf.yml", "r", encoding="utf-8") as f:
                merge_config = MergeConfiguration.model_validate(yaml.safe_load(f))
            
            run_merge(
                merge_config,
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
