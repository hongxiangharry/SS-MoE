from utils.build_patch_lib_utils import build_patch_lib

## evaluate_using_training_testing_split
def build_training_set(gen_conf, train_conf) :
    count = build_patch_lib(gen_conf, train_conf)
    train_conf['actual_num_patches'] = count
    return count