#!/usr/bin/env python3
"""
모든 run 결과를 테이블 형태로 요약하여 출력하는 스크립트
Score, Efficacy, Generalization, Specificity, Fluency, Consistency를 정리
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from scipy.stats import hmean

# 터미널 테이블 출력을 위한 라이브러리
try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False
    print("Warning: tabulate 라이브러리가 없습니다. pip install tabulate로 설치하면 더 예쁜 테이블을 볼 수 있습니다.")

def extract_metrics_from_summarize_results(results_dir: str) -> List[Dict[str, Any]]:
    """
    experiments.summarize 결과에서 메트릭 추출
    """
    from experiments.summarize import main as summarize_main
    
    try:
        # summarize 실행해서 결과 가져오기
        summaries = summarize_main(results_dir, ['all'])
        return summaries
    except Exception as e:
        print(f"summarize 실행 중 오류: {e}")
        return []


def parse_manual_data() -> List[Dict[str, Any]]:
    """
    수동으로 입력된 데이터 파싱 (위 터미널 출력 기반)
    """
    # 위 터미널 출력에서 추출한 데이터
    data = [
        # run_000 (20 cases)
        {
            'run_dir': 'results/MEMIT/run_000',
            'num_cases': 20,
            'post_rewrite_success': (95.0, 21.79),
            'post_paraphrase_success': (82.5, 32.69),
            'post_neighborhood_success': (80.5, 25.78),
            'post_score': (85.54, np.nan),
            'post_ngram_entropy': (629.16, 12.99),
            'post_reference_score': (45.8, 10.16),
            'post_rewrite_acc': (85.0, 35.71),
            'post_paraphrase_acc': (37.5, 38.32),
            'post_neighborhood_acc': (12.0, 23.37),
        },
        # run_001 (50 cases)
        {
            'run_dir': 'results/MEMIT/run_001',
            'num_cases': 50,
            'post_rewrite_success': (98.0, 14.0),
            'post_paraphrase_success': (82.0, 34.29),
            'post_neighborhood_success': (82.6, 24.72),
            'post_score': (86.94, np.nan),
            'post_ngram_entropy': (628.44, 10.73),
            'post_reference_score': (43.6, 11.3),
            'post_rewrite_acc': (82.0, 38.42),
            'post_paraphrase_acc': (49.0, 41.82),
            'post_neighborhood_acc': (16.2, 26.68),
        },
        # run_002 (100 cases)
        {
            'run_dir': 'results/MEMIT/run_002',
            'num_cases': 100,
            'post_rewrite_success': (94.0, 23.75),
            'post_paraphrase_success': (81.5, 33.66),
            'post_neighborhood_success': (80.3, 27.62),
            'post_score': (84.84, np.nan),
            'post_ngram_entropy': (627.03, 12.87),
            'post_reference_score': (41.66, 10.91),
            'post_rewrite_acc': (82.0, 38.42),
            'post_paraphrase_acc': (50.5, 40.92),
            'post_neighborhood_acc': (13.0, 23.98),
        },
        # run_003 (250 cases)
        {
            'run_dir': 'results/MEMIT/run_003',
            'num_cases': 250,
            'post_rewrite_success': (96.0, 19.6),
            'post_paraphrase_success': (82.8, 32.31),
            'post_neighborhood_success': (80.12, 25.31),
            'post_score': (85.77, np.nan),
            'post_ngram_entropy': (626.45, 14.55),
            'post_reference_score': (40.26, 10.94),
            'post_rewrite_acc': (83.6, 37.03),
            'post_paraphrase_acc': (47.0, 42.2),
            'post_neighborhood_acc': (12.88, 20.87),
        },
        # run_004 (500 cases)
        {
            'run_dir': 'results/MEMIT/run_004',
            'num_cases': 500,
            'post_rewrite_success': (93.8, 24.12),
            'post_paraphrase_success': (81.3, 33.69),
            'post_neighborhood_success': (78.52, 25.57),
            'post_score': (84.04, np.nan),
            'post_ngram_entropy': (626.56, 13.23),
            'post_reference_score': (40.1, 12.16),
            'post_rewrite_acc': (80.4, 39.7),
            'post_paraphrase_acc': (46.9, 42.49),
            'post_neighborhood_acc': (12.82, 20.42),
        },
        # run_005 (1000 cases)
        {
            'run_dir': 'results/MEMIT/run_005',
            'num_cases': 1000,
            'post_rewrite_success': (93.6, 24.48),
            'post_paraphrase_success': (79.7, 34.75),
            'post_neighborhood_success': (76.55, 26.4),
            'post_score': (82.66, np.nan),
            'post_ngram_entropy': (626.0, 14.49),
            'post_reference_score': (39.37, 11.94),
            'post_rewrite_acc': (78.5, 41.08),
            'post_paraphrase_acc': (43.25, 42.27),
            'post_neighborhood_acc': (12.57, 19.65),
        },
        # run_006 (3000 cases)
        {
            'run_dir': 'results/MEMIT/run_006',
            'num_cases': 3000,
            'post_rewrite_success': (90.03, 29.96),
            'post_paraphrase_success': (76.18, 37.4),
            'post_neighborhood_success': (73.44, 27.98),
            'post_score': (79.26, np.nan),
            'post_ngram_entropy': (626.03, 13.62),
            'post_reference_score': (38.35, 11.82),
            'post_rewrite_acc': (69.07, 46.22),
            'post_paraphrase_acc': (37.45, 40.85),
            'post_neighborhood_acc': (11.67, 17.82),
        },
        # run_007 (5000 cases)
        {
            'run_dir': 'results/MEMIT/run_007',
            'num_cases': 5000,
            'post_rewrite_success': (86.4, 34.28),
            'post_paraphrase_success': (72.71, 39.25),
            'post_neighborhood_success': (71.66, 28.39),
            'post_score': (76.37, np.nan),
            'post_ngram_entropy': (625.77, 13.61),
            'post_reference_score': (37.64, 11.48),
            'post_rewrite_acc': (60.66, 48.85),
            'post_paraphrase_acc': (32.65, 39.96),
            'post_neighborhood_acc': (11.26, 17.52),
            'post_essence_score': (19.76, 0.0),
        },
        # run_008 (10000 cases)
        {
            'run_dir': 'results/MEMIT/run_008',
            'num_cases': 10000,
            'post_rewrite_success': (79.74, 40.19),
            'post_paraphrase_success': (65.42, 42.08),
            'post_neighborhood_success': (70.03, 28.37),
            'post_score': (71.25, np.nan),
            'post_ngram_entropy': (625.3, 13.5),
            'post_reference_score': (36.39, 11.31),
            'post_rewrite_acc': (46.29, 49.86),
            'post_paraphrase_acc': (24.71, 36.86),
            'post_neighborhood_acc': (10.36, 15.61),
            'post_essence_score': (19.10, 3.03),
        },
        # run_010 (20 cases) - 다른 실험
        {
            'run_dir': 'results/MEMIT/run_010',
            'num_cases': 20,
            'post_rewrite_success': (100.0, 0.0),
            'post_paraphrase_success': (95.0, 21.79),
            'post_neighborhood_success': (77.5, 24.87),
            'post_score': (89.74, np.nan),
            'post_ngram_entropy': (627.77, 13.92),
            'post_reference_score': (48.36, 12.48),
            'post_rewrite_acc': (95.0, 21.79),
            'post_paraphrase_acc': (72.5, 40.23),
            'post_neighborhood_acc': (10.0, 17.89),
        },
        # run_012 (50 cases)
        {
            'run_dir': 'results/MEMIT/run_012',
            'num_cases': 50,
            'post_rewrite_success': (98.0, 14.0),
            'post_paraphrase_success': (93.0, 24.52),
            'post_neighborhood_success': (81.2, 25.35),
            'post_score': (90.17, np.nan),
            'post_ngram_entropy': (624.11, 17.77),
            'post_reference_score': (46.78, 13.01),
            'post_rewrite_acc': (98.0, 14.0),
            'post_paraphrase_acc': (76.0, 36.39),
            'post_neighborhood_acc': (15.8, 26.54),
        },
        # run_014 (100 cases)
        {
            'run_dir': 'results/MEMIT/run_014',
            'num_cases': 100,
            'post_rewrite_success': (99.0, 9.95),
            'post_paraphrase_success': (95.0, 19.36),
            'post_neighborhood_success': (78.3, 28.0),
            'post_score': (89.82, np.nan),
            'post_ngram_entropy': (619.78, 29.16),
            'post_reference_score': (44.51, 14.79),
            'post_rewrite_acc': (98.0, 14.0),
            'post_paraphrase_acc': (75.5, 36.4),
            'post_neighborhood_acc': (13.6, 23.9),
        },
        # run_016 (250 cases)
        {
            'run_dir': 'results/MEMIT/run_016',
            'num_cases': 250,
            'post_rewrite_success': (99.6, 6.31),
            'post_paraphrase_success': (95.2, 17.23),
            'post_neighborhood_success': (76.32, 26.53),
            'post_score': (89.16, np.nan),
            'post_ngram_entropy': (618.43, 24.04),
            'post_reference_score': (42.21, 13.02),
            'post_rewrite_acc': (98.8, 10.89),
            'post_paraphrase_acc': (69.2, 37.57),
            'post_neighborhood_acc': (13.08, 20.53),
        },
        # run_018 (500 cases)
        {
            'run_dir': 'results/MEMIT/run_018',
            'num_cases': 500,
            'post_rewrite_success': (98.8, 10.89),
            'post_paraphrase_success': (93.4, 20.16),
            'post_neighborhood_success': (73.8, 26.93),
            'post_score': (87.26, np.nan),
            'post_ngram_entropy': (619.64, 20.95),
            'post_reference_score': (41.27, 12.75),
            'post_rewrite_acc': (95.4, 20.95),
            'post_paraphrase_acc': (67.6, 38.99),
            'post_neighborhood_acc': (11.86, 18.4),
        },
        # run_020 (1000 cases)
        {
            'run_dir': 'results/MEMIT/run_020',
            'num_cases': 1000,
            'post_rewrite_success': (97.9, 14.34),
            'post_paraphrase_success': (91.85, 22.94),
            'post_neighborhood_success': (70.34, 28.96),
            'post_score': (84.94, np.nan),
            'post_ngram_entropy': (618.9, 23.01),
            'post_reference_score': (40.7, 12.89),
            'post_rewrite_acc': (92.0, 27.13),
            'post_paraphrase_acc': (63.5, 41.2),
            'post_neighborhood_acc': (10.15, 15.94),
        },
        # run_022 (3000 cases)
        {
            'run_dir': 'results/MEMIT/run_022',
            'num_cases': 3000,
            'post_rewrite_success': (96.0, 19.6),
            'post_paraphrase_success': (88.52, 27.58),
            'post_neighborhood_success': (66.91, 29.67),
            'post_score': (81.84, np.nan),
            'post_ngram_entropy': (617.94, 20.32),
            'post_reference_score': (39.47, 12.25),
            'post_rewrite_acc': (79.5, 40.37),
            'post_paraphrase_acc': (53.85, 41.98),
            'post_neighborhood_acc': (9.27, 14.58),
        },
        # run_024 (5000 cases)
        {
            'run_dir': 'results/MEMIT/run_024',
            'num_cases': 5000,
            'post_rewrite_success': (92.86, 25.75),
            'post_paraphrase_success': (84.48, 31.24),
            'post_neighborhood_success': (64.65, 29.82),
            'post_score': (78.79, np.nan),
            'post_ngram_entropy': (616.7, 19.36),
            'post_reference_score': (38.13, 11.72),
            'post_rewrite_acc': (68.66, 46.39),
            'post_paraphrase_acc': (45.36, 42.14),
            'post_neighborhood_acc': (8.66, 14.11),
            'post_essence_score': (27.21, 0.0),
        },
        # run_026 (10000 cases)
        {
            'run_dir': 'results/MEMIT/run_026',
            'num_cases': 10000,
            'post_rewrite_success': (87.41, 33.17),
            'post_paraphrase_success': (77.11, 36.65),
            'post_neighborhood_success': (62.59, 30.25),
            'post_score': (74.28, np.nan),
            'post_ngram_entropy': (615.8, 17.68),
            'post_reference_score': (36.59, 11.6),
            'post_rewrite_acc': (52.65, 49.93),
            'post_paraphrase_acc': (33.17, 40.12),
            'post_neighborhood_acc': (6.93, 12.14),
            'post_essence_score': (26.13, 4.91),
        },
        # run_027 (20 cases) - 새로운 실험 데이터
        {
            'run_dir': 'results/MEMIT/run_027',
            'num_cases': 20,
            'post_rewrite_success': (20.0, 40.0),
            'post_paraphrase_success': (15.0, 32.02),
            'post_neighborhood_success': (83.0, 25.12),
            'post_score': (23.31, np.nan),
            'post_ngram_entropy': (622.74, 16.37),
            'post_reference_score': (28.15, 8.98),
            'post_rewrite_acc': (5.0, 21.79),
            'post_paraphrase_acc': (0.0, 0.0),
            'post_neighborhood_acc': (12.0, 24.0),
        },
        # run_028 (50 cases)
        {
            'run_dir': 'results/MEMIT/run_028',
            'num_cases': 50,
            'post_rewrite_success': (12.0, 32.5),
            'post_paraphrase_success': (15.0, 30.41),
            'post_neighborhood_success': (83.4, 24.05),
            'post_score': (18.52, np.nan),
            'post_ngram_entropy': (626.45, 12.57),
            'post_reference_score': (28.84, 9.27),
            'post_rewrite_acc': (2.0, 14.0),
            'post_paraphrase_acc': (0.0, 0.0),
            'post_neighborhood_acc': (15.2, 25.16),
        },
        # run_029 (100 cases)
        {
            'run_dir': 'results/MEMIT/run_029',
            'num_cases': 100,
            'post_rewrite_success': (15.0, 35.71),
            'post_paraphrase_success': (16.0, 30.72),
            'post_neighborhood_success': (82.0, 27.2),
            'post_score': (21.22, np.nan),
            'post_ngram_entropy': (625.06, 14.75),
            'post_reference_score': (29.24, 9.98),
            'post_rewrite_acc': (1.0, 9.95),
            'post_paraphrase_acc': (0.0, 0.0),
            'post_neighborhood_acc': (12.1, 21.97),
        },
        # run_030 (250 cases)
        {
            'run_dir': 'results/MEMIT/run_030',
            'num_cases': 250,
            'post_rewrite_success': (15.6, 36.29),
            'post_paraphrase_success': (17.0, 30.68),
            'post_neighborhood_success': (81.76, 26.67),
            'post_score': (22.20, np.nan),
            'post_ngram_entropy': (626.18, 12.43),
            'post_reference_score': (31.11, 9.78),
            'post_rewrite_acc': (0.8, 8.91),
            'post_paraphrase_acc': (0.2, 3.16),
            'post_neighborhood_acc': (11.68, 20.07),
        },
        # run_031 (500 cases)
        {
            'run_dir': 'results/MEMIT/run_031',
            'num_cases': 500,
            'post_rewrite_success': (18.2, 38.58),
            'post_paraphrase_success': (19.7, 33.57),
            'post_neighborhood_success': (80.8, 26.9),
            'post_score': (25.41, np.nan),
            'post_ngram_entropy': (626.28, 11.54),
            'post_reference_score': (31.11, 10.37),
            'post_rewrite_acc': (0.6, 7.72),
            'post_paraphrase_acc': (0.2, 3.16),
            'post_neighborhood_acc': (11.9, 20.25),
        },
        # run_032 (1000 cases)
        {
            'run_dir': 'results/MEMIT/run_032',
            'num_cases': 1000,
            'post_rewrite_success': (19.8, 39.85),
            'post_paraphrase_success': (21.95, 35.22),
            'post_neighborhood_success': (80.34, 27.35),
            'post_score': (27.65, np.nan),
            'post_ngram_entropy': (626.49, 12.2),
            'post_reference_score': (31.03, 10.2),
            'post_rewrite_acc': (0.6, 7.72),
            'post_paraphrase_acc': (0.45, 5.22),
            'post_neighborhood_acc': (11.43, 20.14),
        },
        # run_033 (3000 cases)
        {
            'run_dir': 'results/MEMIT/run_033',
            'num_cases': 3000,
            'post_rewrite_success': (20.73, 40.54),
            'post_paraphrase_success': (23.63, 36.67),
            'post_neighborhood_success': (79.09, 27.94),
            'post_score': (29.07, np.nan),
            'post_ngram_entropy': (626.19, 12.16),
            'post_reference_score': (31.46, 10.27),
            'post_rewrite_acc': (0.3, 5.47),
            'post_paraphrase_acc': (0.57, 5.6),
            'post_neighborhood_acc': (10.84, 19.2),
        },
        # run_034 (5000 cases)
        {
            'run_dir': 'results/MEMIT/run_034',
            'num_cases': 5000,
            'post_rewrite_success': (21.48, 41.07),
            'post_paraphrase_success': (24.04, 36.72),
            'post_neighborhood_success': (78.68, 28.0),
            'post_score': (29.74, np.nan),
            'post_ngram_entropy': (626.86, 12.01),
            'post_reference_score': (31.5, 10.4),
            'post_rewrite_acc': (0.36, 5.99),
            'post_paraphrase_acc': (0.51, 5.22),
            'post_neighborhood_acc': (10.5, 18.64),
            'post_essence_score': (13.57, 0.0),
        },
        # run_035 (10000 cases)
        {
            'run_dir': 'results/MEMIT/run_035',
            'num_cases': 10000,
            'post_rewrite_success': (21.72, 41.23),
            'post_paraphrase_success': (24.22, 37.1),
            'post_neighborhood_success': (78.67, 27.99),
            'post_score': (29.99, np.nan),
            'post_ngram_entropy': (626.71, 11.71),
            'post_reference_score': (31.39, 10.17),
            'post_rewrite_acc': (0.26, 5.09),
            'post_paraphrase_acc': (0.42, 4.67),
            'post_neighborhood_acc': (10.62, 18.82),
            'post_essence_score': (15.48, 3.13),
        },
    ]
    
    return data


def calculate_acc_score(item):
    """
    ACC 기반 Score 계산 (Rewrite, Paraphrase, Neighborhood ACC의 조화평균)
    post_score와 동일한 방식으로 계산
    """
    try:
        rewrite_acc = item.get('post_rewrite_acc', (0, 0))[0]
        paraphrase_acc = item.get('post_paraphrase_acc', (0, 0))[0]
        neighborhood_acc = item.get('post_neighborhood_acc', (0, 0))[0]
        
        # 0인 값이 있으면 조화평균 계산 불가
        if rewrite_acc <= 0 or paraphrase_acc <= 0 or neighborhood_acc <= 0:
            return 0.0
            
        # 조화평균 계산 (3개 값의 조화평균)
        acc_score = 3 / (1/rewrite_acc + 1/paraphrase_acc + 1/neighborhood_acc)
        return acc_score
        
    except (ZeroDivisionError, TypeError, KeyError):
        return 0.0


def format_metric(value_tuple, decimal_places=1):
    """
    (평균, 표준편차) 튜플을 포맷팅
    """
    if isinstance(value_tuple, tuple) and len(value_tuple) == 2:
        mean, std = value_tuple
        if np.isnan(std):
            return f"{mean:.{decimal_places}f}"
        else:
            return f"{mean:.{decimal_places}f}±{std:.{decimal_places}f}"
    else:
        return f"{value_tuple:.{decimal_places}f}"


def create_summary_table(data: List[Dict[str, Any]]) -> str:
    """
    주요 메트릭들을 테이블로 정리
    """
    headers = [
        "Run",
        "Cases", 
        "Score",
        "Efficacy",
        "Generalization", 
        "Specificity",
        "Fluency",
        "Consistency"
    ]
    
    rows = []
    for item in data:
        run_name = Path(item['run_dir']).name
        
        # 주요 메트릭 추출
        score = format_metric(item.get('post_score', (0, 0)), 1)
        efficacy = format_metric(item.get('post_rewrite_success', (0, 0)), 1)
        generalization = format_metric(item.get('post_paraphrase_success', (0, 0)), 1)
        specificity = format_metric(item.get('post_neighborhood_success', (0, 0)), 1)
        
        # Fluency (ngram_entropy)는 원래 스케일 유지
        fluency_raw = item.get('post_ngram_entropy', (0, 0))
        if isinstance(fluency_raw, tuple):
            fluency = f"{fluency_raw[0]:.0f}±{fluency_raw[1]:.0f}"
        else:
            fluency = f"{fluency_raw:.0f}"
            
        # Consistency (reference_score)
        consistency = format_metric(item.get('post_reference_score', (0, 0)), 1)
        
        rows.append([
            run_name,
            item['num_cases'],
            score,
            efficacy,
            generalization,
            specificity, 
            fluency,
            consistency
        ])
    
    if TABULATE_AVAILABLE:
        return tabulate(rows, headers=headers, tablefmt="grid", stralign="center")
    else:
        # 간단한 테이블 출력
        col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
        
        def format_row(row):
            return " | ".join(str(item).center(col_widths[i]) for i, item in enumerate(row))
        
        separator = "-" * (sum(col_widths) + 3 * (len(headers) - 1))
        
        table_lines = [
            separator,
            format_row(headers),
            separator
        ]
        
        for row in rows:
            table_lines.append(format_row(row))
            
        table_lines.append(separator)
        
        return "\n".join(table_lines)


def create_detailed_table(data: List[Dict[str, Any]]) -> str:
    """
    더 상세한 메트릭들을 포함한 테이블
    """
    headers = [
        "Run",
        "Cases",
        "Score (ACC)",
        "Rewrite Acc",
        "Paraphrase Acc", 
        "Neighborhood Acc",
        "Essence Score"
    ]
    
    rows = []
    for item in data:
        run_name = Path(item['run_dir']).name
        
        score_acc = f"{calculate_acc_score(item):.1f}"
        rewrite_acc = format_metric(item.get('post_rewrite_acc', (0, 0)), 1)
        paraphrase_acc = format_metric(item.get('post_paraphrase_acc', (0, 0)), 1)
        neighborhood_acc = format_metric(item.get('post_neighborhood_acc', (0, 0)), 1)
        
        # Essence Score (낮을수록 좋음, 없는 경우 N/A)
        essence_score = item.get('post_essence_score')
        if essence_score:
            essence_str = format_metric(essence_score, 1)
        else:
            essence_str = "N/A"
        
        rows.append([
            run_name,
            item['num_cases'],
            score_acc,
            rewrite_acc,
            paraphrase_acc,
            neighborhood_acc,
            essence_str
        ])
    
    if TABULATE_AVAILABLE:
        return tabulate(rows, headers=headers, tablefmt="grid", stralign="center")
    else:
        # 간단한 테이블 출력 (위와 동일한 로직)
        col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
        
        def format_row(row):
            return " | ".join(str(item).center(col_widths[i]) for i, item in enumerate(row))
        
        separator = "-" * (sum(col_widths) + 3 * (len(headers) - 1))
        
        table_lines = [
            separator,
            format_row(headers),
            separator
        ]
        
        for row in rows:
            table_lines.append(format_row(row))
            
        table_lines.append(separator)
        
        return "\n".join(table_lines)


def analyze_trends(data: List[Dict[str, Any]]) -> str:
    """
    케이스 수에 따른 성능 변화 분석
    """
    # 케이스 수별로 그룹화
    case_groups = {}
    for item in data:
        cases = item['num_cases']
        if cases not in case_groups:
            case_groups[cases] = []
        case_groups[cases].append(item)
    
    analysis = ["\n📈 케이스 수별 성능 분석:"]
    
    for cases in sorted(case_groups.keys()):
        items = case_groups[cases]
        if len(items) > 1:
            # 여러 run이 있는 경우 평균 계산
            scores_success = [item['post_score'][0] for item in items if 'post_score' in item]
            scores_acc = [calculate_acc_score(item) for item in items]
            efficacies = [item['post_rewrite_success'][0] for item in items if 'post_rewrite_success' in item]
            generalizations = [item['post_paraphrase_success'][0] for item in items if 'post_paraphrase_success' in item]
            
            if scores_success:
                avg_score_success = np.mean(scores_success)
                avg_score_acc = np.mean(scores_acc)
                avg_efficacy = np.mean(efficacies)
                avg_generalization = np.mean(generalizations)
                
                analysis.append(f"  {cases:5d} cases: Score={avg_score_success:.1f}, ACC-Score={avg_score_acc:.1f}, Efficacy={avg_efficacy:.1f}, Generalization={avg_generalization:.1f} (평균 {len(items)}개 run)")
        else:
            item = items[0]
            score_success = item.get('post_score', (0,))[0]
            score_acc = calculate_acc_score(item)
            efficacy = item.get('post_rewrite_success', (0,))[0]
            generalization = item.get('post_paraphrase_success', (0,))[0]
            analysis.append(f"  {cases:5d} cases: Score={score_success:.1f}, ACC-Score={score_acc:.1f}, Efficacy={efficacy:.1f}, Generalization={generalization:.1f}")
    
    return "\n".join(analysis)


def main():
    parser = argparse.ArgumentParser(description='모든 run 결과를 테이블로 요약')
    parser.add_argument('--results_dir', type=str, default='MEMIT',
                       help='결과 디렉토리 이름 (기본값: MEMIT)')
    parser.add_argument('--detailed', action='store_true',
                       help='상세한 accuracy 메트릭도 표시')
    parser.add_argument('--analysis', action='store_true',
                       help='케이스 수별 성능 분석 추가')

    args = parser.parse_args()
    
    print("🔍 실험 결과 요약 테이블")
    print("=" * 80)
    
    # 데이터 로드 (현재는 수동 데이터 사용)
    data = parse_manual_data()
    
    if not data:
        print("❌ 데이터를 찾을 수 없습니다.")
        return
    
    # 케이스 수 기준으로 정렬
    data.sort(key=lambda x: x['num_cases'])
    
    print("\n📊 주요 메트릭 요약:")
    print("- Score: 전체 성능 (높을수록 좋음)")
    print("- Efficacy: 편집 효과성 (높을수록 좋음)")  
    print("- Generalization: 일반화 성능 (높을수록 좋음)")
    print("- Specificity: 특이성/Locality (높을수록 좋음)")
    print("- Fluency: 생성 다양성 (높을수록 좋음)")
    print("- Consistency: 일관성 (높을수록 좋음)")
    print()
    
    # 주요 메트릭 테이블
    main_table = create_summary_table(data)
    print(main_table)
    
    # 상세 테이블 (옵션)
    if args.detailed:
        print("\n" + "=" * 80)
        print("📋 상세 Accuracy 메트릭:")
        print("- Score (ACC): ACC 기반 전체 성능 (높을수록 좋음)")
        print("- Rewrite Acc: 원본 편집 정확도")
        print("- Paraphrase Acc: 패러프레이즈 정확도") 
        print("- Neighborhood Acc: 관련없는 질문 정확도")
        print("- Essence Score: 핵심 이해도 (낮을수록 좋음)")
        print()
        
        detailed_table = create_detailed_table(data)
        print(detailed_table)
    
    # 성능 분석 (옵션)
    if args.analysis:
        print("\n" + "=" * 80)
        trend_analysis = analyze_trends(data)
        print(trend_analysis)

    print(f"\n✅ 총 {len(data)}개 run 결과 요약 완료")

    if not TABULATE_AVAILABLE:
        print("\n💡 Tip: 'pip install tabulate'로 더 예쁜 테이블을 볼 수 있습니다!")


if __name__ == "__main__":
    main()
