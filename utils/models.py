from models.spiraconv import *
from models.panns import *

def return_model(c, model_params=False):
    model_name = c.model_name
    if(model_name == 'spiraconv_v1'):
        model = SpiraConvV1(c)
    elif (model_name == 'spiraconv_v2'):
        model = SpiraConvV2(c)
    elif (model_name == 'spiraconv_v3'):
        if not model_params:
            model = SpiraConvV3(c)
        else:
            model = SpiraConvV3(**model_params)
    
    elif (model_name == 'spiraconv_v4'):
        if not model_params:
            model = SpiraConvV4(c)
        else:
            model = SpiraConvV4(**model_params)
    elif (model_name == 'spiraconvlstm_v1'):
        model = SpiraConvLSTMV1(c)
    elif (model_name == 'spiraconvattn_v1'):
            model = SpiraConvAttnV1(c)
    elif (model_name == 'vit_v1'):
        model = SpiraVITv1(c)
    elif (model_name == 'vit_v2'):
        model = SpiraVITv2(c)
    elif (model_name == 'spt_v1'):
        model = SpiraSpTv1(c)
    elif (model_name == 'spt_v2'):
        if not model_params:
            model = SpiraSpTv2(c)
        else:
            model = SpiraSpTv2(**model_params)
    elif(model_name == 'panns'):
        model = Transfer_Cnn14(c)

    elif(model_name == 'panns_resnet'):
        model = Transfer_ResNet38(c)
    elif(model_name == 'panns_mobile'):
        model = Transfer_MobileNetV1(c)
    else:
        raise Exception(" The model '"+model_name+"' is not suported")
    return model