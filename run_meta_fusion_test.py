# @Time   : 2020/7/20
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE
# @Time   : 2020/10/3, 2020/10/1
# @Author : Yupeng Hou, Zihan Lin
# @Email  : houyupeng@ruc.edu.cn, zhlin@ruc.edu.cn


import argparse

from recbole.quick_start import run_meta_fusion_test


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_list', '-m', type=str, default='BPR', help='list of name of models')
        parser.add_argument('--dataset', '-d', type=str, default='ml-100k', help='name of datasets')
        parser.add_argument('--config_files', type=str, default=None, help='config files')
        parser.add_argument('--saved', type=str, default='True', help='saved')
        parser.add_argument('--hint', type=str, default='', help='hint for run_recbole')

        args, _ = parser.parse_known_args()

        model_list = eval(args.model_list) if args.model_list else None
        config_file_lists = eval(args.config_files) if args.config_files else None
        saved = (args.saved.lower() == 'true')
        result = run_meta_fusion_test(
            model_name_list=model_list, dataset=args.dataset, config_file_lists=config_file_lists, saved=saved
        )
        try:
            from mtjupyter_utils import remind
            message = ' '.join([str(args.model), args.hint]) + '\n'
            message += ' '.join(map(str, result.values()))
            remind(message)
        except Exception as e:
            pass
    except Exception as err:
        try:
            from mtjupyter_utils import remind
            remind(str(err))
            raise err
        except Exception as e:
            raise err
