from utils.preprocessing_util import interp_input

def interpolate(gen_conf, test_conf) :
    if 'interp' in test_conf:
        interp = test_conf['interp']
        is_interp = interp['is_interp']
    else:
        is_interp = False

    ## interpolation
    if is_interp is True:
        interp_order = interp['interp_order']
        print("Interpolating test data...")
        interp_input(gen_conf, test_conf, interp_order=interp_order) # input pre-processing

    return True