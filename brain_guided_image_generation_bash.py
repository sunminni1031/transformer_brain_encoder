import os


def run_main(rois_str, num_cpu=1, num_gpu=0, priority=True, cluster=True):
    config_file = rois_str.replace('-', '_')
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
                    + f'python3 brain_guided_image_generation.py --rois_str={rois_str}'
                    + '\n'
                    + 'exit 0;\n')
        if priority:
            test_cmd = f'sbatch --qos=high-priority {config_file}.sh'
        else:
            test_cmd = f'sbatch {config_file}.sh'
        os.system(test_cmd)
    else:
        # nohup mycommand > mycommand.out 2>&1 &
        test_cmd = f'python3 brain_guided_image_generation.py --rois_str={rois_str}'
        os.system(test_cmd)
    return


if __name__ == '__main__':
    # roi_names = [#'V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4', 
    #              'EBA', 'FBA-1', 'FBA-2', 'mTL-bodies', 
    #              'OFA', 'FFA-1', 'FFA-2', 'mTL-faces', 'aTL-faces', 
    #              'OPA', 'PPA', 'RSC', 
    #              'OWFA', 'VWFA-1', 'VWFA-2', 'mfs-words', 'mTL-words']
    # for roi_name in roi_names:
        # run_main(roi_name, num_cpu=3, num_gpu=1, priority=True, cluster=True)
    
    for rois_list in [
                        # ['OFA', 'FFA-1', 'FFA-2'],
                        # ['OPA', 'PPA', 'RSC'],
                        ['EBA', 'FBA-1', 'FBA-2'],
                        # ['OWFA', 'VWFA-1', 'VWFA-2'],
                      ]:
        rois_str = '_'.join(rois_list)
        run_main(rois_str, num_cpu=3, num_gpu=1, priority=True, cluster=True)