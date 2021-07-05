from .AnisoUnetCustomLoss import generate_aniso_unet_model
from .IsoSRUnetCustomLoss import generate_iso_srunet_model, generate_iso_srunet_model_S3
from .MoEIsoSRUnet import generate_moe_iso_srunet_model
from .MoEAnisoUnet import generate_moe_aniso_unet_model

def generate_model(gen_conf, train_conf) :
    approach = train_conf['approach']

    if approach == 'AnisoUnet' :
        return generate_aniso_unet_model(gen_conf, train_conf)
    if approach == 'IsoSRUnet' :
        return generate_iso_srunet_model(gen_conf, train_conf)
    if approach == 'MoEIsoSRUnet':
        return generate_moe_iso_srunet_model(gen_conf, train_conf)
    if approach == 'MoEAnisoUnet':
        return generate_moe_aniso_unet_model(gen_conf, train_conf)
    return None


def generate_model_s3(gen_conf, train_conf) :

    return generate_iso_srunet_model_S3(gen_conf, train_conf)