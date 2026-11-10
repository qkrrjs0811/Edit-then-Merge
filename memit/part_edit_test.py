import argparse
import re

def calculate_accuracy(log_file_path, start_idx, end_idx):
    correct_count = 0
    total_count = 0
    skipped_count = 0
    processed_indices = set()
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
        # 공통적인 형식으로 섹션 찾기
        pattern = r'#################### \[(\d+)\] \w+ test_batch_prediction\(\)\._print\(\) ####################(.*?)(?=#################### \[\d+\] \w+ test_batch_prediction\(\)\._print\(\)|$)'
        matches = re.finditer(pattern, content, re.DOTALL)
        
        print(f"\n디버깅 정보:")
        print(f"파일 경로: {log_file_path}")
        print(f"검색 범위: {start_idx} ~ {end_idx}")
        
        for match in matches:
            idx = int(match.group(1))
            processed_indices.add(idx)
            
            if start_idx <= idx <= end_idx:
                section = match.group(2)
                
                # predict와 target 비교 - 수정된 패턴
                predict_match = re.search(r'\[0\] predict\s*:\s*([^,]+)', section)
                target_match = re.search(r'target\s*:\s*(\w+)', section)
                
                if predict_match and target_match:
                    predict = predict_match.group(1).strip()
                    target = target_match.group(1)
                    
                    # 빈 값이나 괄호만 있는 경우 처리
                    if predict == '(' or predict == '':
                        predict = 'EMPTY'
                    
                    if predict == target:
                        correct_count += 1
                    total_count += 1
                else:
                    skipped_count += 1
                    print(f"\n인덱스 {idx}에서 predict/target 매칭 실패:")
                    if not predict_match:
                        print("predict를 찾을 수 없음")
                    if not target_match:
                        print("target을 찾을 수 없음")
    
    print(f"\n상세 통계:")
    print(f"처리된 총 인덱스 수: {len(processed_indices)}")
    print(f"범위 내 인덱스 수: {len([i for i in processed_indices if start_idx <= i <= end_idx])}")
    print(f"매칭 실패로 건너뛴 케이스 수: {skipped_count}")
    
    if total_count == 0:
        return 0.0, 0, 0
    
    accuracy = correct_count / total_count
    return accuracy, correct_count, total_count

def main():
    # 로그 파일 경로 직접 지정
    log_file_path = "logs/hclt/2_1500/edited215002_gpt2xl_ties_l1.6_d0.5.txt"
    
    parser = argparse.ArgumentParser(description='Calculate accuracy for a specific range of edit results')
    parser.add_argument('--start_idx', type=int, required=True, help='Start index of the range')
    parser.add_argument('--end_idx', type=int, required=True, help='End index of the range')
    
    args = parser.parse_args()
    
    accuracy, correct, total = calculate_accuracy(log_file_path, args.start_idx, args.end_idx)
    
    print(f"\n정확도 계산 결과:")
    print(f"범위: [{args.start_idx}] ~ [{args.end_idx}]")
    print(f"맞은 개수: {correct}")
    print(f"전체 개수: {total}")
    print(f"정확도: {accuracy:.4f}")

if __name__ == "__main__":
    main()
