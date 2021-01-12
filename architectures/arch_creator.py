from .Kamnitsas import generate_kamnitsas_model
from .Dolz import generate_dolz_multi_model
from .Cicek import generate_unet_model as generate_unet_cicek_model
from .Guerrero import generate_uresnet_model
from .IsoUnet import generate_iso_unet_model
from .AnisoUnetOld import generate_aniso_unet_old_model
from .SRUnet import generate_srunet_model
from .AnisoUnetCustomLoss import generate_aniso_unet_model

def generate_model(gen_conf, train_conf) :
    approach = train_conf['approach']

    if approach == 'Kamnitsas' :
        return generate_kamnitsas_model(gen_conf, train_conf)
    if approach == 'DolzMulti' :
        return generate_dolz_multi_model(gen_conf, train_conf)
    if approach == 'Cicek' :
        return generate_unet_cicek_model(gen_conf, train_conf)
    if approach == 'Guerrero' :
        return generate_uresnet_model(gen_conf, train_conf)
    if approach == 'IsoUnet' :
        return generate_iso_unet_model(gen_conf, train_conf)
    if approach == 'AnisoUnetOld' :
        return generate_aniso_unet_old_model(gen_conf, train_conf)
    if approach == 'SRUnet' :
        return generate_srunet_model(gen_conf, train_conf)
    if approach == 'AnisoUnet' :
        return generate_aniso_unet_model(gen_conf, train_conf)
    return None
