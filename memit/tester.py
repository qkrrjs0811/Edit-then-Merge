import argparse

from common_util import *
from model_editor import *


SEED = 7
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# memit/ 디렉토리 (이 파일 기준 상대 경로)
HOME_DIR = str(Path(__file__).resolve().parent)


def load_datas(in_file_path: str):
    in_file = open_file(in_file_path, mode='r')
    datas = json.load(in_file)

    print(f'# data_preprocessor.load_datas() datas size : {len(datas)}, in_file_path : {in_file_path}')
    return datas


def get_model_editor(num_edits=100, hparams_fname_suffix='', hparams_mod=None, model_path=None, alg_name='MEMIT'):
    model_name = 'gpt2-xl'
    hparams_fname = model_name + '{}.json'.format(hparams_fname_suffix)
    ds_name = 'mcf'

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
        hparams_mod, merged_model_path=model_path
    )

    return model_editor


def edit_and_save(num_edits_list=(250,)):
    # 원본 모델에 num_edits개 지식을 MEMIT으로 한 번에 편집하고 모델만 저장 (평가는 이후 별도 실행)

    home_dir = HOME_DIR
    data_dir = f'{home_dir}/data/preprocessing'

    for num_edits in num_edits_list:
        model_editor = get_model_editor(num_edits)

        model_editor._do_eval_org_model = False
        model_editor._do_eval_new_model = False

        in_file_path = f'{data_dir}/normal/mcf_sampled_{num_edits}.json'
        datas_subject = load_datas(in_file_path)

        print("\n[SAVE] 편집한 모델 저장중...")
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
        model_editor_org = get_model_editor(num_edits)
        model_editor_org._do_eval_org_model = False
        model_editor_org._do_eval_new_model = False

        print("\n[Test] 기존 Editing 방법에 대해 평가 수행 중...")
        model_editor_org.edit_ext_datas(datas_subject, False, True, False, True, False, False)


def org_test(num_edits_list):
    # 편집하지 않은 원본 모델 평가
    home_dir = HOME_DIR
    data_dir = f'{home_dir}/data/preprocessing'

    for num_edits in num_edits_list:
        model_editor = get_model_editor(num_edits)

        model_editor._do_eval_org_model = True
        model_editor._do_eval_new_model = False

        in_file_path = f'{data_dir}/normal/mcf_sampled_{num_edits}.json'
        datas_subject = load_datas(in_file_path)

        print("\n[Test] 원본 모델 평가 수행 중...")
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
            log_merged=False,
            log_memit=False
        )


def merge_test(num_edits_list):
    # 경로 설정
    home_dir = HOME_DIR
    data_dir = f'{home_dir}/data/preprocessing'
    merged_dir = Path(f"{home_dir}/merged")

    for num_edits in num_edits_list:
        # Origin Edit method 평가 수행
        model_editor_org = get_model_editor(num_edits)
        model_editor_org._do_eval_org_model = False
        model_editor_org._do_eval_new_model = False

        in_file_path = f'{data_dir}/normal/mcf_sampled_{num_edits}.json'
        datas_subject = load_datas(in_file_path)

        # /logs 디렉토리에 memit_1_{num_edits}.txt와 동일한 파일이 존재하는지 확인
        log_dir = f"./logs/hclt/2_{int(num_edits/2)}"
        log_file_path = f"{log_dir}/memit_1_{num_edits}.txt"
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
                log_merged=False,
                log_memit=True
            )

        # 병합된 모델 디렉토리 탐색
        for merged_model_path in merged_dir.iterdir():
            if not merged_model_path.is_dir():
                continue

            # /logs 디렉토리에 동일한 이름의 txt 파일이 존재하는지 확인
            log_file_path = f"{log_dir}/{merged_model_path.name}.txt"
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

            model_editor_merging = get_model_editor(num_edits, model_path=merged_model_path)
            model_editor_merging._model = model
            model_editor_merging._tok = tok
            model_editor_merging._do_eval_org_model = False
            model_editor_merging._do_eval_new_model = False

            print("[Test] 병합된 모델에 대해 평가 수행 중...")
            model_editor_merging.edit_ext_datas(
                datas_subject,
                do_org_test=False,
                do_edit=False,
                do_edit_test=True,
                do_extend_test=False,
                do_restore=False,
                do_restore_test=False,
                do_print=True,
                do_save=False,
                log_org=False,
                log_merged=True,
                log_memit=False
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MEMIT 편집 / 병합 모델 평가')
    parser.add_argument('--num_edits_list', type=int, nargs='+', required=True,
                        help='편집할 지식 개수 (여러 개를 주면 차례대로 실행)')
    parser.add_argument('--mode', type=str, default='merge_test',
                        choices=['edit', 'merge_test', 'org_test'],
                        help='edit: MEMIT 편집 후 edited_models/에 저장 | '
                             'merge_test: MEMIT baseline + merged/ 모델 평가 (기본값) | '
                             'org_test: 원본 모델 평가')

    args = parser.parse_args()

    if args.mode == 'edit':
        edit_and_save(args.num_edits_list)
    elif args.mode == 'merge_test':
        merge_test(args.num_edits_list)
    elif args.mode == 'org_test':
        org_test(args.num_edits_list)
