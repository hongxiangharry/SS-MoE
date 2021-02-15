from utils.image_evaluation import image_evaluation, interp_evaluation

def evaluation(gen_conf, test_conf, case_name = 1):
    image_evaluation(gen_conf, test_conf, case_name)

def interp_eval(gen_conf, test_conf, case_name = 1):
    interp_evaluation(gen_conf, test_conf, case_name)
