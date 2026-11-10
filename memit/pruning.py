#!/usr/bin/env python3
"""
모델 파라미터 희소화 스크립트
edited_models 폴더 내의 모델들에 대해 파라미터의 절대값 크기를 기준으로
상위 k%의 파라미터만 남기고 나머지는 0으로 만드는 스크립트
"""

import os
import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from tqdm import tqdm
import json
import shutil


def get_top_k_mask(tensor, k_percent):
    """
    텐서에서 절대값 기준 상위 k% 위치의 마스크를 반환
    
    Args:
        tensor: 입력 텐서
        k_percent: 유지할 파라미터의 비율 (0-100)
    
    Returns:
        mask: 상위 k% 위치는 True, 나머지는 False인 마스크
    """
    flat_tensor = tensor.flatten()
    abs_values = torch.abs(flat_tensor)
    
    # 상위 k% 위치 계산
    k_count = int(len(abs_values) * k_percent / 100.0)
    if k_count == 0:
        k_count = 1  # 최소 1개는 유지
    
    # 상위 k개의 임계값 찾기
    threshold_value = torch.topk(abs_values, k_count)[0][-1]
    
    # 마스크 생성 (threshold_value 이상인 값들을 True로)
    mask = torch.abs(tensor) >= threshold_value
    
    return mask


def sparsify_model_parameters(model, k_percent):
    """
    모델의 파라미터를 희소화
    
    Args:
        model: PyTorch 모델
        k_percent: 유지할 파라미터의 비율 (0-100)
    
    Returns:
        sparsity_stats: 희소화 통계 정보
    """
    sparsity_stats = {}
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:  # 학습 가능한 파라미터만 처리
                original_param = param.data.clone()
                
                # 상위 k% 마스크 생성
                mask = get_top_k_mask(param.data, k_percent)
                
                # 마스크 적용 (상위 k%만 유지, 나머지는 0)
                param.data = param.data * mask.float()
                
                # 통계 계산
                total_params = param.numel()
                non_zero_params = torch.count_nonzero(param.data).item()
                actual_sparsity = (1 - non_zero_params / total_params) * 100
                
                sparsity_stats[name] = {
                    'total_params': total_params,
                    'non_zero_params': non_zero_params,
                    'sparsity_percent': actual_sparsity,
                    'original_norm': torch.norm(original_param).item(),
                    'sparsified_norm': torch.norm(param.data).item()
                }
                
                print(f"{name}: {total_params:,} -> {non_zero_params:,} params "
                      f"({actual_sparsity:.2f}% sparse)")
    
    return sparsity_stats


def sparsify_model_directory(model_dir, k_percent, output_suffix="_sparse"):
    """
    모델 디렉토리를 희소화하여 새로운 디렉토리에 저장
    
    Args:
        model_dir: 원본 모델 디렉토리 경로
        k_percent: 유지할 파라미터의 비율 (0-100)
        output_suffix: 출력 디렉토리 접미사
    """
    model_dir = Path(model_dir)
    output_dir = model_dir.parent / f"{model_dir.name}{output_suffix}_k{k_percent}"
    
    print(f"\n처리 중: {model_dir} -> {output_dir}")
    
    # 출력 디렉토리 생성
    output_dir.mkdir(exist_ok=True)
    
    try:
        # 모델 로드
        print("모델 로딩 중...")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=torch.float32,
        )
        
        # 토크나이저도 로드 (설정 파일들과 함께 복사하기 위해)
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        except:
            tokenizer = None
            print("토크나이저 로딩 실패 - 모델 파일만 처리합니다.")
        
        # 희소화 적용
        print(f"파라미터 희소화 중 (상위 {k_percent}% 유지)...")
        sparsity_stats = sparsify_model_parameters(model, k_percent)
        
        # 희소화된 모델 저장
        print("희소화된 모델 저장 중...")
        model.save_pretrained(str(output_dir))
        
        # 토크나이저 저장
        if tokenizer is not None:
            tokenizer.save_pretrained(str(output_dir))
        
        # 추가 설정 파일들 복사 (config.json, tokenizer 관련 파일들)
        for file_name in ["config.json", "merges.txt", "special_tokens_map.json", 
                         "tokenizer_config.json", "tokenizer.json", "vocab.json"]:
            src_file = model_dir / file_name
            if src_file.exists():
                shutil.copy2(src_file, output_dir / file_name)
        
        # 통계 정보 저장
        stats_file = output_dir / "sparsity_stats.json"
        with open(stats_file, 'w') as f:
            # numpy 타입을 JSON 직렬화 가능하도록 변환
            json_stats = {}
            for param_name, stats in sparsity_stats.items():
                json_stats[param_name] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                        for k, v in stats.items()}
            
            json.dump({
                'k_percent': k_percent,
                'total_model_params': sum(stats['total_params'] for stats in sparsity_stats.values()),
                'total_non_zero_params': sum(stats['non_zero_params'] for stats in sparsity_stats.values()),
                'overall_sparsity': (1 - sum(stats['non_zero_params'] for stats in sparsity_stats.values()) / 
                                   sum(stats['total_params'] for stats in sparsity_stats.values())) * 100,
                'per_parameter_stats': json_stats
            }, f, indent=2)
        
        print(f"희소화 완료! 결과 저장됨: {output_dir}")
        print(f"통계 정보: {stats_file}")
        
        # 메모리 정리
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f"오류 발생: {e}")
        # 실패한 경우 부분적으로 생성된 디렉토리 정리
        if output_dir.exists():
            shutil.rmtree(output_dir)
        raise


def main():
    parser = argparse.ArgumentParser(description='모델 파라미터 희소화 도구')
    parser.add_argument('--edited_models_dir', type=str, 
                       default=str(Path(__file__).resolve().parent / 'edited_models'),
                       help='edited_models 디렉토리 경로')
    parser.add_argument('--k_percent', type=float, required=True,
                       help='유지할 파라미터의 비율 (0-100)')
    parser.add_argument('--model_pattern', type=str, default='edited_*',
                       help='처리할 모델 디렉토리 패턴')
    parser.add_argument('--output_suffix', type=str, default='_sparse',
                       help='출력 디렉토리 접미사')
    parser.add_argument('--single_model', type=str, default=None,
                       help='단일 모델만 처리할 경우 모델 이름')
    
    args = parser.parse_args()
    
    if not (0 < args.k_percent <= 100):
        raise ValueError("k_percent는 0과 100 사이의 값이어야 합니다.")
    
    edited_models_dir = Path(args.edited_models_dir)
    if not edited_models_dir.exists():
        raise ValueError(f"디렉토리가 존재하지 않습니다: {edited_models_dir}")
    
    # 처리할 모델 디렉토리 찾기
    if args.single_model:
        model_dirs = [edited_models_dir / args.single_model]
        if not model_dirs[0].exists():
            raise ValueError(f"모델 디렉토리가 존재하지 않습니다: {model_dirs[0]}")
    else:
        model_dirs = list(edited_models_dir.glob(args.model_pattern))
        model_dirs = [d for d in model_dirs if d.is_dir()]
    
    if not model_dirs:
        print(f"처리할 모델이 없습니다. 패턴: {args.model_pattern}")
        return
    
    print(f"총 {len(model_dirs)}개의 모델을 처리합니다.")
    print(f"상위 {args.k_percent}% 파라미터를 유지합니다.")
    
    # 각 모델 디렉토리 처리
    failed_models = []
    for model_dir in tqdm(model_dirs, desc="모델 처리 중"):
        try:
            sparsify_model_directory(model_dir, args.k_percent, args.output_suffix)
        except Exception as e:
            print(f"모델 {model_dir.name} 처리 실패: {e}")
            failed_models.append(model_dir.name)
            continue
    
    print(f"\n처리 완료!")
    print(f"성공: {len(model_dirs) - len(failed_models)}/{len(model_dirs)} 모델")
    
    if failed_models:
        print(f"실패한 모델들: {failed_models}")


if __name__ == "__main__":
    main()
