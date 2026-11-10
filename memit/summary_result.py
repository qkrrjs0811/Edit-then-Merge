import argparse
import os
import re
from collections import Counter
from typing import Dict, List, Tuple


MERGE_METHODS_ORDER = [
    "task_arithmetic",
    "ties",
    "dare_ties",
    "dare_linear",
    "breadcrumbs",
    "breadcrumbs_ties",
    "della",
    "della_linear",
    "sce",
]


def is_target_experiment_file(filename: str, version: str) -> bool:
    """
    version 기준으로 분석 대상 파일을 필터링.

    - all: 모든 병합/평가 로그 포함
    - e2o: 편집모델 + 원본모델(gpt2xl 포함 이름)
    - e2e: 편집모델 + 편집모델(gpt2xl 미포함 병합 이름)
    """
    if version == "all":
        return True

    lower_name = filename.lower()

    # memit 기준선 로그는 버전과 무관하게 항상 포함(Threshold 계산 및 비교용)
    if lower_name.startswith("memit_"):
        return True

    has_gpt2xl = "gpt2xl" in lower_name
    edited_token_count = len(re.findall(r"edited\d+", lower_name))
    is_merged_style = lower_name.startswith("merged_")

    if version == "e2o":
        # e2o를 "단일 편집모델 + 원본모델"로 엄격히 정의:
        # - gpt2xl 포함
        # - edited 토큰은 최대 1개(예: edited1201_gpt2xl_...)
        # - legacy merged_* 로그명 제외
        return has_gpt2xl and edited_token_count <= 1 and not is_merged_style
    if version == "e2e":
        # e2e를 "편집모델 + 편집모델"로 정의:
        # - gpt2xl 미포함 OR edited 토큰이 2개 이상(재병합 케이스 포함)
        return (not has_gpt2xl) or (edited_token_count >= 2)

    return True


def extract_parameters(filename: str) -> List[float]:
    """파일명에서 숫자형 하이퍼파라미터를 추출하여 리스트로 반환"""
    numbers = re.findall(r"\d+\.\d+|\d+", filename)
    return [float(num) for num in numbers]


def get_merge_method_index(filename: str, merge_methods_order: List[str]) -> int:
    """
    병합 방식이 정확하게 포함되어 있는 경우만 인덱스를 반환.
    길이가 긴 병합 방식부터 먼저 탐색해서 겹침 문제 해결.
    """
    sorted_methods = sorted(merge_methods_order, key=len, reverse=True)
    for method in sorted_methods:
        pattern = f"_{method}_"
        if pattern in filename:
            return merge_methods_order.index(method)
    return len(merge_methods_order)


def process_log_files(logs_dir: str, version: str = "all") -> str:
    """logs_dir의 로그를 합쳐 combined_results.txt 생성"""
    files = [f for f in os.listdir(logs_dir) if f.endswith(".txt")]
    files = [f for f in files if f not in ("combined_results.txt", "high_performance_methods.txt", "hyperparameter_analysis_result.txt")]
    files = [f for f in files if is_target_experiment_file(f, version)]

    sorted_files = sorted(
        files,
        key=lambda x: (
            get_merge_method_index(x, MERGE_METHODS_ORDER),
            extract_parameters(x),
        ),
    )

    combined_content = ""
    i = 1
    for filename in sorted_files:
        file_path = os.path.join(logs_dir, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        if "edit finish" in content:
            post_edit_content = content.split("edit finish", 1)[1]
            combined_content += "=" * 50 + f"\n{i}번째 파일\n"
            combined_content += f"File: {filename}\n"
            combined_content += post_edit_content
            combined_content += "\n" + "=" * 50 + "\n"
            i += 1

    combined_file_path = os.path.join(logs_dir, "combined_results.txt")
    if os.path.exists(combined_file_path):
        os.remove(combined_file_path)
    with open(combined_file_path, "w", encoding="utf-8") as combined_file:
        combined_file.write(combined_content)

    print(f"[1/3] Combined results saved to: {combined_file_path} (version={version}, files={len(sorted_files)})")
    return combined_file_path


def extract_high_performance_methods(file_path: str, threshold: float) -> List[Tuple[str, float]]:
    """combined_results에서 edited(avg) >= threshold인 파일 추출"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = content.split("==================================================")
    results: List[Tuple[str, float]] = []

    for block in blocks:
        file_match = re.search(r"File:\s*(.+\.txt)", block)
        edited_avg_match = re.search(r"edited \(avg\)\s*:\s*\[?([0-9.]+)", block)

        if file_match and edited_avg_match:
            file_name = file_match.group(1).strip()
            edited_avg = float(edited_avg_match.group(1))
            if edited_avg >= threshold:
                results.append((file_name, edited_avg))

    return results


def save_high_performance_methods(logs_dir: str, high_perf_files: List[Tuple[str, float]]) -> str:
    output_file_path = os.path.join(logs_dir, "high_performance_methods.txt")
    if os.path.exists(output_file_path):
        os.remove(output_file_path)
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        for file_name, score in high_perf_files:
            output_file.write(f"{file_name} - edited avg: {score}\n")

    print(f"[2/3] High-performance methods saved to: {output_file_path} (count={len(high_perf_files)})")
    return output_file_path


def extract_hyperparameters_from_lines(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    lambda_counter = Counter()
    density_counter = Counter()
    epsilon_counter = Counter()

    for line in lines:
        lambda_val, density_val, epsilon_val = parse_hyperparams_from_text(line)
        if lambda_val is not None:
            lambda_counter[lambda_val] += 1
        if density_val is not None:
            density_counter[density_val] += 1
        if epsilon_val is not None:
            epsilon_counter[epsilon_val] += 1

    return lambda_counter, density_counter, epsilon_counter


def parse_hyperparams_from_text(text: str):
    """
    하이퍼파라미터 파싱 유틸.
    지원 포맷:
    - 구버전: \"lambda': 1.6\", \"density': 0.5\", \"epsilon': 0.1\"
    - 신버전: \"_l1.6\", \"_d0.5\", \"_e0.1\"
    """
    # 숫자 파싱은 소수점 끝 문자(예: "1.0.")가 섞이지 않게 엄격히 제한
    number_pattern = r"([0-9]+(?:\.[0-9]+)?)"
    lambda_match = re.search(rf"lambda':\s*{number_pattern}", text) or re.search(rf"_l{number_pattern}(?:_|\.|$)", text)
    density_match = re.search(rf"density':\s*{number_pattern}", text) or re.search(rf"_d{number_pattern}(?:_|\.|$)", text)
    epsilon_match = re.search(rf"epsilon':\s*{number_pattern}", text) or re.search(rf"_e{number_pattern}(?:_|\.|$)", text)

    lambda_val = float(lambda_match.group(1)) if lambda_match else None
    density_val = float(density_match.group(1)) if density_match else None
    epsilon_val = float(epsilon_match.group(1)) if epsilon_match else None
    return lambda_val, density_val, epsilon_val


def analyze_hyperparameters(high_perf_file: str, combined_file: str):
    success_lambda, success_density, success_epsilon = extract_hyperparameters_from_lines(high_perf_file)
    total_lambda, total_density, total_epsilon = extract_hyperparameters_from_lines(combined_file)

    lambda_ratios = {k: (success_lambda[k], total_lambda[k]) for k in total_lambda.keys()}
    density_ratios = {k: (success_density[k], total_density[k]) for k in total_density.keys()}
    epsilon_ratios = {k: (success_epsilon[k], total_epsilon[k]) for k in total_epsilon.keys()}

    return lambda_ratios, density_ratios, epsilon_ratios


def count_methods_in_combined_file(file_path: str) -> Dict[str, int]:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    method_counts = {
        "task_arithmetic": 0,
        "ties": 0,
        "dare_ties": 0,
        "dare_linear": 0,
        "기타": 0,
    }

    file_names = re.findall(r"File:\s*(.+\.txt)", content)
    for file_name in file_names:
        lower_name = file_name.lower()
        if "task_arithmetic" in lower_name:
            method_counts["task_arithmetic"] += 1
        elif "dare_ties" in lower_name:
            method_counts["dare_ties"] += 1
        elif "dare_linear" in lower_name:
            method_counts["dare_linear"] += 1
        elif "ties" in lower_name and "dare" not in lower_name:
            method_counts["ties"] += 1
        else:
            method_counts["기타"] += 1

    return method_counts


def count_methods_in_high_perf_file(file_path: str) -> Dict[str, int]:
    """high_performance_methods.txt에서 방법론별 성공 개수를 집계"""
    method_counts = {
        "task_arithmetic": 0,
        "ties": 0,
        "dare_ties": 0,
        "dare_linear": 0,
        "기타": 0,
    }

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        file_name = line.strip().split(" - edited avg: ")[0].lower()
        if not file_name:
            continue

        if "task_arithmetic" in file_name:
            method_counts["task_arithmetic"] += 1
        elif "dare_ties" in file_name:
            method_counts["dare_ties"] += 1
        elif "dare_linear" in file_name:
            method_counts["dare_linear"] += 1
        elif "ties" in file_name and "dare" not in file_name:
            method_counts["ties"] += 1
        else:
            method_counts["기타"] += 1

    return method_counts


def extract_hyperparameter_combinations(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    combinations = []
    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split(" - edited avg: ")
        if len(parts) != 2:
            continue

        file_name = parts[0]
        performance = float(parts[1])

        lambda_val, density_val, epsilon_val = parse_hyperparams_from_text(file_name)
        if lambda_val is not None:

            if "task_arithmetic" in file_name:
                method = "task_arithmetic"
            elif "ties" in file_name and "dare" not in file_name:
                method = "ties"
            elif "dare_ties" in file_name:
                method = "dare_ties"
            elif "dare_linear" in file_name:
                method = "dare_linear"
            else:
                method = "기타"

            combinations.append(
                {
                    "lambda": lambda_val,
                    "density": density_val,
                    "epsilon": epsilon_val,
                    "performance": performance,
                    "file_name": file_name,
                    "method": method,
                }
            )

    return combinations


def analyze_sorted_combinations(high_perf_file: str, combined_file: str) -> List[str]:
    combinations = extract_hyperparameter_combinations(high_perf_file)
    if not combinations:
        return ["분석할 조합이 없습니다."]

    total_method_counts = count_methods_in_combined_file(combined_file)
    combinations.sort(key=lambda x: x["performance"], reverse=True)

    method_priority = {"dare_ties": 1, "ties": 2, "task_arithmetic": 3, "dare_linear": 4, "기타": 5}

    performance_groups = {}
    for combo in combinations:
        perf = combo["performance"]
        performance_groups.setdefault(perf, []).append(combo)

    result_lines = []
    result_lines.append("")
    result_lines.append("")
    result_lines.append("=" * 80)
    result_lines.append("성능별 하이퍼파라미터 조합 (성능 내림차순, 방법론별 정렬)")
    result_lines.append("=" * 80)

    for perf in sorted(performance_groups.keys(), reverse=True):
        group = performance_groups[perf]
        result_lines.append(f"\n성능 {perf:.3f} (총 {len(group)}개 조합):")

        method_counts = {}
        for combo in group:
            method = combo["method"]
            method_counts[method] = method_counts.get(method, 0) + 1

        result_lines.append("  [방법론별 통계 (전체 대비)]:")
        for method in sorted(method_counts.keys(), key=lambda x: method_priority.get(x, 999)):
            count = method_counts[method]
            total_count = total_method_counts.get(method, 0)
            percentage = (count / total_count) * 100 if total_count > 0 else 0
            result_lines.append(f"    {method:15}: {count:2d}/{total_count:2d} ({percentage:5.1f}%)")

        lambda_counts = {}
        density_counts = {}
        epsilon_counts = {}

        for combo in group:
            lambda_val = combo["lambda"]
            lambda_counts[lambda_val] = lambda_counts.get(lambda_val, 0) + 1

            if combo["density"] is not None:
                density_val = combo["density"]
                density_counts[density_val] = density_counts.get(density_val, 0) + 1

            if combo["epsilon"] is not None:
                epsilon_val = combo["epsilon"]
                epsilon_counts[epsilon_val] = epsilon_counts.get(epsilon_val, 0) + 1

        result_lines.append("  [하이퍼파라미터 통계 (값별 개수)]:")
        result_lines.append("    Lambda:")
        for lambda_val in sorted(lambda_counts.keys()):
            count = lambda_counts[lambda_val]
            result_lines.append(f"      {lambda_val:.1f}: {count:2d}개")

        if density_counts:
            result_lines.append("    Density:")
            for density_val in sorted(density_counts.keys()):
                count = density_counts[density_val]
                result_lines.append(f"      {density_val:.1f}: {count:2d}개")
        else:
            result_lines.append("    Density: 없음")

        if epsilon_counts:
            result_lines.append("    Epsilon:")
            for epsilon_val in sorted(epsilon_counts.keys()):
                count = epsilon_counts[epsilon_val]
                result_lines.append(f"      {epsilon_val:.1f}: {count:2d}개")
        else:
            result_lines.append("    Epsilon: 없음")

        group.sort(key=lambda x: (method_priority.get(x["method"], 999), x["lambda"], x["density"] or 0))
        result_lines.append("  [상세 조합 목록]:")
        for i, combo in enumerate(group, 1):
            density_str = f", density={combo['density']:.3f}" if combo["density"] is not None else ""
            epsilon_str = f", epsilon={combo['epsilon']:.3f}" if combo["epsilon"] is not None else ""
            result_lines.append(f"    {i:2d}. [{combo['method']:15}] lambda={combo['lambda']:.3f}{density_str}{epsilon_str}")

    return result_lines


def infer_memit_threshold_from_logs_dir(logs_dir: str) -> float:
    """
    logs_dir 내 memit_*.txt 파일에서 edited(avg)를 직접 읽어 임계값으로 사용.
    여러 개면 최대값 사용.
    """
    scores = []
    for filename in os.listdir(logs_dir):
        if not (filename.startswith("memit_") and filename.endswith(".txt")):
            continue

        file_path = os.path.join(logs_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        score_match = re.search(r"edited \(avg\)\s*:\s*([0-9.]+)", content)
        if score_match:
            scores.append(float(score_match.group(1)))

    if not scores:
        raise ValueError(
            f"{logs_dir}에서 memit_*.txt 기준 점수를 찾지 못했습니다. "
            "--threshold를 직접 지정하거나 memit 로그를 먼저 생성하세요."
        )

    return max(scores)


def run_all(logs_dir: str, threshold: float = None, version: str = "all"):
    combined_file = process_log_files(logs_dir, version=version)

    if threshold is None:
        threshold = infer_memit_threshold_from_logs_dir(logs_dir)
        print(f"[Info] Threshold not provided. Using MEMIT baseline from {logs_dir}: {threshold:.6f}")
    else:
        print(f"[Info] Using user threshold: {threshold:.6f}")

    high_perf = extract_high_performance_methods(combined_file, threshold)
    high_perf_file = save_high_performance_methods(logs_dir, high_perf)

    lambda_ratios, density_ratios, epsilon_ratios = analyze_hyperparameters(high_perf_file, combined_file)
    total_method_counts = count_methods_in_combined_file(combined_file)
    success_method_counts = count_methods_in_high_perf_file(high_perf_file)

    result_lines = []
    result_lines.append("=" * 80)
    result_lines.append("하이퍼파라미터 분석 결과")
    result_lines.append("=" * 80)
    result_lines.append(f"\n기준 Threshold (edited avg): {threshold:.6f}")

    result_lines.append("\n병합 방법 성공 비율:")
    for method in ["task_arithmetic", "ties", "dare_ties", "dare_linear", "기타"]:
        success = success_method_counts.get(method, 0)
        total = total_method_counts.get(method, 0)
        ratio = success / total if total > 0 else 0.0
        result_lines.append(f"{method}={ratio:.2%} ({success}/{total})")

    result_lines.append("\nLambda 성공 비율:")
    for lambda_val, (success, total) in sorted(lambda_ratios.items()):
        ratio = success / total if total > 0 else 0.0
        result_lines.append(f"lambda={lambda_val}: {ratio:.2%} ({success}/{total})")

    result_lines.append("\nDensity 성공 비율:")
    for density_val, (success, total) in sorted(density_ratios.items()):
        ratio = success / total if total > 0 else 0.0
        result_lines.append(f"density={density_val}: {ratio:.2%} ({success}/{total})")

    result_lines.append("\nEpsilon 성공 비율:")
    for epsilon_val, (success, total) in sorted(epsilon_ratios.items()):
        ratio = success / total if total > 0 else 0.0
        result_lines.append(f"epsilon={epsilon_val}: {ratio:.2%} ({success}/{total})")

    # 성능별 상세 조합 목록은 생략 (요청 사항)

    output_file = os.path.join(logs_dir, "hyperparameter_analysis_result.txt")
    if os.path.exists(output_file):
        os.remove(output_file)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(result_lines))

    print(f"[3/3] Hyperparameter analysis saved to: {output_file}")
    print("[Done] All steps completed with one command.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="통합 로그 분석: summary + high-perf 추출 + 하이퍼파라미터 분석")
    parser.add_argument("--logs_dir", type=str, default="./logs/hclt/2_10", help="분석할 로그 디렉토리")
    parser.add_argument(
        "--version",
        type=str,
        choices=["all", "e2o", "e2e"],
        default="all",
        help="분석 대상 버전: all(전체), e2o(편집모델+원본모델), e2e(편집모델+편집모델)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="고성능 기준 edited(avg). 미지정 시 combined_results에서 memit_* 값을 자동 사용",
    )
    args = parser.parse_args()

    run_all(args.logs_dir, args.threshold, args.version)
