import argparse
import random
from copy import deepcopy
from tqdm import tqdm

from .falcon_util import *
from .model_editor import *


SEED = 7
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# memit/ 디렉토리 (이 파일 기준 상대 경로)
HOME_DIR = str(Path(__file__).resolve().parent.parent)


def load_datas(in_file_path: str):
    in_file = open_file(in_file_path, mode='r')
    datas = json.load(in_file)

    print(f'# data_preprocessor.load_datas() datas size : {len(datas)}, in_file_path : {in_file_path}')
    return datas


def get_model_editor(num_edits=100, hparams_fname_suffix='', hparams_mod=None, identical_num=1, model_path=None, pruning=False, alg_name='MEMIT'):
    # alg_name = 'MEMIT'
    model_name = 'gpt2-xl'
    hparams_fname = model_name + '{}.json'.format(hparams_fname_suffix)
    ds_name = 'mcf'
    # num_edits = 100

    dataset_size_limit = None
    continue_from_run = None
    skip_generation_tests = False
    generation_test_interval = 1
    conserve_memory = False
    dir_name = alg_name
    use_cache = False
    output_hidden_states = False

    model_editor = ModelEditor(
        alg_name, model_name, hparams_fname, ds_name,
        dataset_size_limit, continue_from_run, skip_generation_tests,
        generation_test_interval, conserve_memory, dir_name, num_edits, use_cache, output_hidden_states,
        hparams_mod, identical_num, model_path, pruning
    )

    return model_editor


def edit_and_save(identical_nums=(2,), num_edits_list=(250,)):
    # 현재 목적: 각 batch를 독립적으로 편집하고 모델만 저장 (평가는 이후 별도 실행)

    home_dir = HOME_DIR
    data_dir = f'{home_dir}/data/preprocessing'

    for identical_num, num_edits in zip(identical_nums, num_edits_list):
        # identical_num: 총 batch의 개수
        # num_edits: 하나의 batch에 들어간 데이터 개수

        model_editor = get_model_editor(num_edits, identical_num=identical_num)

        model_editor._do_eval_org_model = False
        model_editor._do_eval_new_model = False
        
        # === Normal Edit ===
        size = identical_num * num_edits
        in_file_path = f'{data_dir}/normal/mcf_sampled_{size}.json'
        datas_subject = load_datas(in_file_path)

        # # === Sequential Edit ===
        # in_file_path = f'{data_dir}/mcf_sequential_identical{identical_num}_subjects_all.json'
        # datas_subject = load_datas(in_file_path)


        # # === Multiple Edit ===
        # size = identical_num * num_edits
        # if identical_num == 2:
        #     in_file_path = f'{data_dir}/multi_counterfact_identical{identical_num}_ext_n_{size}.json'
        # else:
        #     in_file_path = f'{data_dir}/multi_counterfact_identical{identical_num}_all_{size}.json'
        # datas_subject = load_datas(in_file_path)


        print("\n[SAVE] 각 batch별로 편집한 모델 독립적으로 저장중...")
        model_editor.edit_ext_datas(
            datas_subject,
            do_org_test=False,
            do_edit=True,
            do_edit_test=False,
            do_extend_test=False,
            do_restore=True,
            do_restore_test=False,
            do_print=True,
            do_save=True  # 저장 목적
        )
        
        # Origin Edit method 평가 수행
        model_editor_org = get_model_editor(num_edits, identical_num=identical_num)
        model_editor_org._do_eval_org_model = False
        model_editor_org._do_eval_new_model = False

        print("\n[Test] 기존 Editing 방법에 대해 평가 수행 중...")
        model_editor_org.edit_ext_datas(datas_subject, False, True, False, True, False, False)


def merge_test(model_name):

    # 경로 설정
    home_dir = HOME_DIR
    data_dir = f'{home_dir}/data/preprocessing'
    merged_model_path = Path(f"{home_dir}/merged/{model_name}")

    # 병합된 모델 디렉토리 존재 여부 확인
    if not merged_model_path.exists():
        print(f"[Error] 병합된 모델 디렉토리가 존재하지 않습니다: {merged_model_path}")
        return

    print(f"[Test] 병합된 모델 로드 중... from {merged_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        str(merged_model_path),
        torch_dtype=torch.float32
    ).cuda()
    tok = AutoTokenizer.from_pretrained(str(merged_model_path))
    tok.pad_token = tok.eos_token


    # === Normal Edit ===
    # 실험용 데이터 로드 및 평가 실행
    for identical_num, num_edits in zip([2], [250]):
        size = identical_num * num_edits
        in_file_path = f'{data_dir}/normal/mcf_sampled_{size}.json'
        datas_subject = load_datas(in_file_path)


        # Merge한 모델 평가 수행
        model_editor_merging = get_model_editor(num_edits)
        model_editor_merging._model = model
        model_editor_merging._tok = tok
        model_editor_merging._do_eval_org_model = False
        model_editor_merging._do_eval_new_model = True

        print("[Test] 병합된 모델에 대해 평가 수행 중...")
        model_editor_merging.edit_ext_datas(
            datas_subject,
            do_org_test=False,
            do_edit=False,
            do_edit_test=False,
            do_extend_test=True,  # 🔹 평가 목적
            do_restore=False,
            do_restore_test=False,
            do_print=False,
            do_save=False
        )
    

    # # === Sequential Edit ===
    # # 실험용 데이터 로드 및 평가 실행
    # for identical_num, num_edits in zip([4], [5]):
    #     in_file_path = f'{data_dir}/mcf_sequential_identical{identical_num}_subjects_all.json'
    #     datas_subject = load_datas(in_file_path)


    #     # Merge한 모델 평가 수행
    #     model_editor_merging = get_model_editor(num_edits)
    #     model_editor_merging._model = model
    #     model_editor_merging._tok = tok
    #     model_editor_merging._do_eval_org_model = False
    #     model_editor_merging._do_eval_new_model = True

    #     print("[Test] 병합된 모델에 대해 평가 수행 중...")
    #     model_editor_merging.edit_ext_datas(
    #         datas_subject,
    #         do_org_test=False,
    #         do_edit=False,
    #         do_edit_test=False,
    #         do_extend_test=True,  # 🔹 평가 목적
    #         do_restore=False,
    #         do_restore_test=False,
    #         do_print=False,
    #         do_save=False
    #     )


    # # === Multiple Edit ===
    # # 실험용 데이터 로드 및 평가 실행
    # for identical_num, num_edits in zip([2], [500]):
    #     size = identical_num * num_edits
    #     if identical_num == 2:
    #         in_file_path = f'{data_dir}/multi_counterfact_identical{identical_num}_ext_n_{size}.json'
    #     else:
    #         in_file_path = f'{data_dir}/multi_counterfact_identical{identical_num}_all_{size}.json'
    #     datas_subject = load_datas(in_file_path)


    #     # Merged된 모델 평가 수행
    #     model_editor = get_model_editor(num_edits)
    #     model_editor._model = model
    #     model_editor._tok = tok
    #     model_editor._do_eval_org_model = False
    #     model_editor._do_eval_new_model = True

    #     print("[Test] 병합된 모델에 대해 평가 수행 중...")
    #     model_editor.edit_ext_datas(
    #         datas_subject,
    #         do_org_test=False,
    #         do_edit=False,
    #         do_edit_test=False,
    #         do_extend_test=True,  # 🔹 평가 목적
    #         do_restore=False,
    #         do_restore_test=False,
    #         do_print=False,
    #         do_save=False
    #     )


def hclt_edit_and_save(identical_nums, num_edits_list):
    home_dir = HOME_DIR
    data_dir = f'{home_dir}/data/preprocessing'

    for identical_num, num_edits in zip(identical_nums, num_edits_list):
        model_editor = get_model_editor(num_edits, identical_num=identical_num)

        model_editor._do_eval_org_model = True
        model_editor._do_eval_new_model = False
        
        size = identical_num * num_edits
        in_file_path = f'{data_dir}/normal/mcf_sampled_{size}.json'
        datas_subject = load_datas(in_file_path)

        print("\n[SAVE] 각 batch별로 모델 편집 및 독립적으로 저장중...")
        model_editor.edit_ext_datas(
            datas_subject,
            do_org_test=True,
            do_edit=False,
            do_edit_test=False,
            do_extend_test=False,
            do_restore=False,
            do_restore_test=False,
            do_print=True,
            do_save=False,
            log_org=True,
            log_independent=False,
            log_merged=False,
            log_memit=False
        )


def hclt_merge_test(identical_nums, num_edits_list, test_type='all', test_batch_size=0):
    # 경로 설정
    home_dir = HOME_DIR
    data_dir = f'{home_dir}/data/preprocessing'
    merged_dir = Path(f"{home_dir}/merged")

    for identical_num, num_edits in zip(identical_nums, num_edits_list):
        # Origin Edit method 평가 수행
        model_editor_org = get_model_editor(num_edits, identical_num=identical_num)
        model_editor_org._do_eval_org_model = False
        model_editor_org._do_eval_new_model = False

        size = identical_num * num_edits
        in_file_path = f'{data_dir}/normal/mcf_sampled_{size}.json'
        datas_subject = load_datas(in_file_path)

        # /logs 디렉토리에 memit_{identical_num}_{num_edits}.txt와 동일한 파일이 존재하는지 확인
        log_file_path = f"./logs/hclt/{int(identical_num*2)}_{int(num_edits/2)}/memit_{identical_num}_{num_edits}.txt"
        print("#########################")
        print(f"log_file_path: {log_file_path}")
        if os.path.exists(log_file_path):
            print(f"[Warning] 동일한 이름의 로그 파일이 이미 존재합니다: {log_file_path}")
            print("기존 Editing 방법에 대한 평가를 건너뛰고 병합된 모델 디렉토리 탐색으로 넘어갑니다.")
        else:
            print("\n[Test] 기존 Editing 방법에 대해 평가 수행 중...")
            model_editor_org.edit_ext_datas(
                datas_subject,
                do_org_test=False,
                do_edit=True,
                do_edit_test=True,
                do_extend_test=False,
                do_restore=False,
                do_restore_test=False,
                do_print=True,
                do_save=False,
                log_org=False,
                log_independent=False,
                log_merged=False,
                log_memit=True
            )

        # 병합된 모델 디렉토리 탐색
        for merged_model_path in merged_dir.iterdir():
            if not merged_model_path.is_dir():
                continue

            # /logs 디렉토리에 동일한 이름의 txt 파일이 존재하는지 확인
            log_file_path = f"./logs/hclt/{int(identical_num*2)}_{int(num_edits/2)}/{merged_model_path.name}.txt"
            print("#########################")
            print(f"log_file_path: {log_file_path}")
            if os.path.exists(log_file_path):
                print(f"[Warn] 동일한 이름의 로그 파일이 이미 존재합니다: {log_file_path}")
                print("다음 경로로 넘어갑니다.")
                continue

            print(f"[Test] 병합된 모델 로드 중... from {merged_model_path}")

            model = AutoModelForCausalLM.from_pretrained(
                str(merged_model_path),
                torch_dtype=torch.float32
            ).cuda()
            tok = AutoTokenizer.from_pretrained(str(merged_model_path))
            tok.pad_token = tok.eos_token

            # test_type에 따른 평가 데이터 설정
            if test_type == 'all':
                eval_datas = datas_subject
                print(f"[Info] test_type=all: 전체 데이터 사용 (총 {len(eval_datas)}개)")
            elif test_type == 'each':
                batch_size = len(datas_subject) // 2
                if test_batch_size == 0:
                    eval_datas = datas_subject[:batch_size]  # 첫 번째 절반
                    print(f"[Info] test_type=each, test_batch_size=0: 첫 번째 절반 데이터 사용 (0~{batch_size-1})")
                elif test_batch_size == 1:
                    eval_datas = datas_subject[batch_size:]  # 두 번째 절반
                    print(f"[Info] test_type=each, test_batch_size=1: 두 번째 절반 데이터 사용 ({batch_size}~{len(eval_datas)+batch_size-1})")
                else:
                    eval_datas = datas_subject
                    print(f"[Warning] test_batch_size={test_batch_size}는 지원되지 않습니다. 전체 데이터를 사용합니다.")
            else:
                eval_datas = datas_subject
                print(f"[Warning] test_type={test_type}는 지원되지 않습니다. 전체 데이터를 사용합니다.")

            model_editor_merging = get_model_editor(num_edits, identical_num=identical_num, model_path=merged_model_path)
            model_editor_merging._model = model
            model_editor_merging._tok = tok
            model_editor_merging._do_eval_org_model = False
            model_editor_merging._do_eval_new_model = False

            print("[Test] 병합된 모델에 대해 평가 수행 중...")
            model_editor_merging.edit_ext_datas(
                eval_datas,
                do_org_test=False,
                do_edit=False,
                do_edit_test=True,
                do_extend_test=False,  # 🔹 평가 목적
                do_restore=False,
                do_restore_test=False,
                do_print=True,
                do_save=False,
                log_org=False,
                log_independent=False,
                log_merged=True,
                log_memit=False
            )


def evaluate_edited_model(identical_nums, num_edits_list, test_type, test_batch_size):
    """
    edited_models 폴더에 있는 모델을 평가하는 함수
    편집 없이 순수 평가만 수행
    
    Args:
        identical_nums: List of identical numbers for batches
        num_edits_list: List of number of edits per batch
        test_type: test_type ('all' or 'each')
        test_batch_size: test_batch_size
    """
    # 경로 설정
    home_dir = HOME_DIR
    data_dir = f'{home_dir}/data/preprocessing'
    edited_models_dir = Path(f"{home_dir}/edited_models")
    
    for identical_num, num_edits in zip(identical_nums, num_edits_list):
        print(f"\n=== edited_models 평가 시작 (identical_num: {identical_num}, num_edits: {num_edits}) ===")
        
        # 평가 데이터 로드
        size = identical_num * num_edits
        in_file_path = f'{data_dir}/normal/mcf_sampled_{size}.json'
        print(f"데이터 파일: {in_file_path}")
        datas_subject = load_datas(in_file_path)
        
        # test_type에 따른 평가 데이터 설정
        if test_type == 'all':
            eval_datas = datas_subject
            print(f"[Info] test_type=all: 전체 데이터 사용 (총 {len(eval_datas)}개)")
        elif test_type == 'each':
            batch_size = len(datas_subject) // 2
            if test_batch_size == 0:
                eval_datas = datas_subject[:batch_size]  # 첫 번째 절반
                print(f"[Info] test_type=each, test_batch_size=0: 첫 번째 절반 데이터 사용 (0~{batch_size-1})")
            elif test_batch_size == 1:
                eval_datas = datas_subject[batch_size:]  # 두 번째 절반
                print(f"[Info] test_type=each, test_batch_size=1: 두 번째 절반 데이터 사용 ({batch_size}~{len(eval_datas)+batch_size-1})")
            else:
                eval_datas = datas_subject
                print(f"[Warning] test_batch_size={test_batch_size}는 지원되지 않습니다. 전체 데이터를 사용합니다.")
        else:
            eval_datas = datas_subject
            print(f"[Warning] test_type={test_type}는 지원되지 않습니다. 전체 데이터를 사용합니다.")
        
        # edited_models 디렉토리 탐색
        pattern = f"edited_{identical_num}_{num_edits}_1_sparse*"   # 패턴 추출
        model_dirs = list(edited_models_dir.glob(pattern))
        model_dirs = [d for d in model_dirs if d.is_dir()]
        
        if not model_dirs:
            print(f"[Warning] 패턴 '{pattern}'에 해당하는 모델이 없습니다.")
            continue
        
        print(f"발견된 모델들: {[d.name for d in model_dirs]}")
        
        for model_path in model_dirs:
            try:
                print(f"\n--- 모델 평가: {model_path.name} ---")
                
                # 로그 파일 경로 확인
                log_file_path = f"./logs/hclt/{int(identical_num*2)}_{int(num_edits/2)}/edited_{model_path.name}.txt"
                print(f"log_file_path: {log_file_path}")
                if os.path.exists(log_file_path):
                    print(f"[Warning] 동일한 이름의 로그 파일이 이미 존재합니다: {log_file_path}")
                    print("다음 모델로 넘어갑니다.")
                    continue
                
                # 모델 로드
                print("모델 로딩 중...")
                model = AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    torch_dtype=torch.float32
                ).cuda()
                
                # 토크나이저 로드
                print("토크나이저 로딩 중...")
                tok = AutoTokenizer.from_pretrained(str(model_path))
                tok.pad_token = tok.eos_token
                
                # ModelEditor로 평가 수행
                print("ModelEditor 초기화 중...")
                model_editor = get_model_editor(num_edits, identical_num=identical_num, model_path=model_path)
                model_editor._model = model
                model_editor._tok = tok
                model_editor._do_eval_org_model = False
                model_editor._do_eval_new_model = False
                
                # 평가 실행
                print("평가 수행 중...")
                model_editor.edit_ext_datas(
                    eval_datas,
                    do_org_test=True,      # 원본 모델 테스트 안함
                    do_edit=False,          # 편집 안함
                    do_edit_test=False,      # 편집 후 테스트 수행
                    do_extend_test=False,   # 확장 평가 안함
                    do_restore=False,       # 가중치 복원 안함
                    do_restore_test=False,  # 복원 후 테스트 안함
                    do_print=True,          # 결과 출력
                    do_save=False,          # 저장 안함
                    log_org=False,
                    log_independent=False,
                    log_merged=True,        # edited model 로그
                    log_memit=False
                )
                
                print(f"=== 모델 평가 완료: {model_path.name} ===")
                
            except Exception as e:
                print(f"모델 {model_path.name} 평가 실패: {e}")
                continue


            
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--identical_nums', type=int, nargs='+', help='List of identical numbers for batches')
    parser.add_argument('--num_edits_list', type=int, nargs='+', help='List of number of edits per batch')
    parser.add_argument('--test_type', type=str, default='all', help='test_type')
    parser.add_argument('--test_batch_size', type=int, default=0, help='test_batch_size')
    parser.add_argument('--mode', type=str, default='merge_test',
                        choices=['edit', 'merge_test', 'eval_edited', 'org_test'],
                        help='edit: batch별 독립 편집 후 edited_models/에 저장 | '
                             'merge_test: MEMIT baseline + merged/ 모델 평가 (기본값) | '
                             'eval_edited: edited_models/*_sparse 평가 (pruning 실험) | '
                             'org_test: 원본 모델 예측 로그')

    args = parser.parse_args()

    if args.mode == 'edit':
        # 독립적으로 편집 후 저장
        edit_and_save(args.identical_nums, args.num_edits_list)
    elif args.mode == 'merge_test':
        # memit 환경에서 실행
        hclt_merge_test(args.identical_nums, args.num_edits_list, args.test_type, args.test_batch_size)
    elif args.mode == 'eval_edited':
        # edited models 평가 (pruning한 모델 평가할 때)
        evaluate_edited_model(args.identical_nums, args.num_edits_list, args.test_type, args.test_batch_size)
    elif args.mode == 'org_test':
        hclt_edit_and_save(args.identical_nums, args.num_edits_list)
    
    
