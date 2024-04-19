from argparse import Namespace

ModelArguments = Namespace(
    CLIP_MODEL = 'openai/clip-vit-large-patch14',
    
    seed=0, 
    
    patience=5, # Early Stop parameters
    num_workers=8,
    
    SCHEDULER_USE = True,
    LR_SCHEDULER_T_MAX=100,
    
    DATAPARALLEL = False,
    
    learning_rate=1e-4, 
    epochs=20,
    batch_size=128,
    iteration = 20, 
    
    TEXT_ENCODER_FREEZE = 11,
    VISION_ENCODER_FREEZE = 0,
    
    
    LOSS_TEMPERATURE = 0.05,
    
    MODEL_NAME= 'CFT-CLIP_vf0_tf11',
    MODEL_SAVE_PATH= '../model/CFT-CLIP',
)

DataArguments = Namespace(
    TRAIN_DATA_PATH = './data/train_ntt_cossim_CFT.pkl',
    VALID_DATA_PATH = "./data/valid_ntt_cossim_CFT.pkl",
    ORG_TITLE = 'summary',
    NEGATIVE_TITLE = 'predict_t2_0',
    IMAGE_TENSOR = './data/pixel_values',
    IMAGE_ID = 'image_id',
    TRAIN_DATA_LOADER_SHUFFLE=True, 
    TRAIN_DATA_LOADER_DROP_LAST=True,
    VALID_DATA_LOADER_SHUFFLE=False, 
    VALID_DATA_LOADER_DROP_LAST=False,
    pin_memory=False, 
)