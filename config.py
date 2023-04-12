import json
from argparse import Namespace

ModelArguments = Namespace(
    CLIP_MODEL = 'openai/clip-vit-large-patch14', #"openai/clip-vit-base-patch32" "openai/clip-vit-large-patch14"
    seed=42, # 변경함!! 42
    hidden_size=768, # large: 768, base: 512
    intermediate_size=2048,
    
    patience=5, # Early Stop parameters
    num_workers=4,
    
    SCHEDULER_USE = True,
    LR_SCHEDULER_T_MAX=100,
    
    ## CL
    DATAPARALLEL = False, # 병렬처리 할지
    TEXT_ENCODER_NO_FREEZE = False, #True이면 vision만 freeze하고 싶으면 아래 None
    TEXT_ENCODER_PART_FREEZE = None, #3, 6,9, None
    
    learning_rate=1e-4,
    epochs=15,
    batch_size=1024, # CL 1024
    iteration = 20, # 290039 => 20480
    
    Contrastive_Mode = 'imgtotxt_H', # or imgtotxt, imgtotxt_H, txttotxt_H, imgtotxt_txttotxt_H
    MLP_USE = True, # True MLP 추가됨
     
    ADAM = False, # False ADAMW
    LOSS_TEMPERATURE = 0.05,
    # # GAMMA = 1.0,
    
    # # CF
    # n_cls=1,
    # learning_rate=1e-4,
    # epochs=3,
    # batch_size=256,
    # iteration=80, #840116 코퍼스 => 20480
    # CF_ENCODER = 'model_pt/20221130/large/ver12/clip_pre_part6_8_best.pt',#'model_pt/20221130/large/ver12/clip_pre_best.pt', # None, or path 
    # CF_ENCODER_CONFIG = 'model_pt/20221130/large/ver12/clip_pre_part6_8_config.json',#'model_pt/20221130/large/ver12/clip_pre_config.json', #'model_pt/20221130/large/ver8/pre_config.json'
    
    MODEL_NAME= 'NewsCLIP',#'clip_pre_CF', #pre, pre_tnf, ours, ours_tnf
    MODEL_SAVE_PATH= 'model_pt/20230328',#'model_pt/20221130/large/ver17',
    TRAIN_MODEL = 'contrastive', # or classifier, contrastive
)



DataArguments = Namespace(
    TRAIN_DATA_PATH = './data/Final_filtering_process_221226/230102/renewal_data_division_20230327/train_CL.pkl', # CL CF2: 그대로 CF3: PERSON처리 CFs|HN_NON 221229/train_CFs, 221229/valid_CFs  train_CFs_v1
    VALID_DATA_PATH = './data/Final_filtering_process_221226/230102/renewal_data_division_20230327/valid_CL.pkl',
    ORG_TITLE = 'contraction_title', #masked_title, title T-title contraction_title	
    NEGATIVE_TITLE = 'masked_title_predict_bert_base', #'MASK_PREDICT', # #masked_title_MP25 # distilbert_t025_np_mask_pred// masked_person_npv_title_predict masked_npv_title_predict  masked_NPV_title_predict_bert_base
    IMAGE_PATH= '../../coling/news_image', 
    IMAGE_TENSOR = './data/clip-vit-large-patch14_NELA_PIXEL_VALUES',#clip-vit-base-patch32_NELA_PIXEL_VALUES',clip-vit-large-patch14_NELA_PIXEL_VALUES 
    TRAIN_DATA_LOADER_SHUFFLE=True, 
    TRAIN_DATA_LOADER_DROP_LAST=True,
    VALID_DATA_LOADER_SHUFFLE=False, 
    VALID_DATA_LOADER_DROP_LAST=False,
    # TEST_DATA_PATH = './data/test_CL.pkl',
    pin_memory=True, 
)

#title, masked_npv_title_predict  
#masked_title, masked_person_npv_title_predict

## masked_npv_title_bert_large_predict    contraction_title

def save_args(path, epoch, iteration):
    data_path = path+'.json'
    args = {}
    args['ModelArguments'] = vars(ModelArguments)
    args['DataArguments'] = vars(DataArguments)
    # args['loss_data'] = loss_data
    args['best_epoch'] = epoch
    args['best_iteration'] = iteration
    with open(data_path, 'w') as f:
        json.dump(args, f)
        
def save_loss(path='checkpoint_loss', loss_data=[]):
    data_path = path+'.json'
    args = {}
    args['loss_data'] = loss_data
    with open(data_path, 'w') as f:
        json.dump(args, f)