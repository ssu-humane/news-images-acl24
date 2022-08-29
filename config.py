import json
from argparse import Namespace

ModelArguments = Namespace(
    seed=42,

    hidden_size=512,
    intermediate_size=2048,
    n_cls=1,

    learning_rate=1e-7,
    batch_size=256,
    num_workers=0,
    epochs=100,
    iteration = 8806400, # 176333기사 batch 256 기준 1 epoch 당 iteration 688 * 5- = 48200 * 256 => 12339200

    MODEL_NAME='clip-baseline3-contrastive',
    MODEL_SAVE_PATH='model_pt',


    SCHEDULER_USE = False,
    LR_SCHEDULER_T_MAX=100,


    TRAIN_MODEL = 'contrastive', # or classifier

    Contrastive_Mode = 'imgtotxt', # or imgtotxt_H, txttotxt_H, imgtoimg_H_txttotxt_H
    LOSS_TEMPERATURE = 0.5,
    GAMMA = 1.0,
)


DataArguments = Namespace(
    TRAIN_DATA_PATH = './data/NELA_train.json',
    VALID_DATA_PATH = './data/NELA_valid.json',
    NEGATIVE_TITLE = 'SHUFFLE', # or 'MASK_PREDICT' RANDOM NOUN(Pharse) MASK and MASK PREDICT TITLE by BERT Base uncased (huggingface)
    IMAGE_PATH= '../../coling/news_image', #'../../ACLW2022_data/general_meta_img_jpg',
    DATA_LOADER_SHUFFLE=True,
    DATA_LOADER_DROP_LAST=True,
)

def save_args(path, loss_data):
    data_path = path+'/'+ModelArguments.MODEL_NAME+'.json'
    args = {}
    args['ModelArguments'] = vars(ModelArguments)
    args['DataArguments'] = vars(DataArguments)
    args['loss_data'] = loss_data
    with open(data_path, 'w') as f:
        json.dump(args, f)