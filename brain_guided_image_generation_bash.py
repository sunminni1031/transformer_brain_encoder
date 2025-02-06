import os


def run_main(rois_str, detach_k, subj, num_cpu=1, num_gpu=0, priority=True, cluster=True):
    config_file = rois_str.replace('-', '_') + f'_detachk{detach_k}_subj{subj}'
    if cluster:
        with open(config_file + '.sh', 'w') as f:
            f.write('#!/bin/bash\n'
                    + '\n'
                    + '#SBATCH --nodes=1\n'
                    + f'#SBATCH --cpus-per-task={num_cpu}\n'
                    + f'#SBATCH --mem-per-cpu=20gb\n'
                    + f'#SBATCH --gres=gpu:{num_gpu}\n'  # --gres=gpu:gtx1080:1\n'
                    + f'#SBATCH --output={config_file}.out\n'
                    #+ f'#SBATCH --account=nklab\n'
                    + '\n'
                    + '\n'
                    + f'python3 brain_guided_image_generation.py --rois_str={rois_str} --detach_k={detach_k} --subj={subj}'
                    + '\n'
                    + 'exit 0;\n')
        if priority:
            test_cmd = f'sbatch --qos=high-priority {config_file}.sh'
        else:
            test_cmd = f'sbatch {config_file}.sh'
        os.system(test_cmd)
    else:
        # nohup mycommand > mycommand.out 2>&1 &
        test_cmd = f'python3 brain_guided_image_generation.py --rois_str={rois_str} --detach_k={detach_k} --subj={subj}'
        os.system(test_cmd)
    return


if __name__ == '__main__':
    for rois_list in [
                        # ['OFA', 'FFA-1', 'FFA-2'],
                        ['OPA', 'PPA', 'RSC'],
                        # ['EBA', 'FBA-1', 'FBA-2'],
                        # ['OWFA', 'VWFA-1', 'VWFA-2'],
                      ]:
        rois_str = '_'.join(rois_list)
        for detach_k in [1]: #[0, 1]:
            for subj in [2]:
                run_main(rois_str, detach_k, subj, num_cpu=3, num_gpu=1, priority=True, cluster=True)