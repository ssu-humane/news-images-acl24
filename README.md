# Assessing News Thumbnail Representativeness

## Task: Understanding News Thumbnail Representativeness
Given a news thumbnail I and its news text T, our task is to predict a binary label L indicating whether news thumbnail I is representative of T or not.

## Method: CFT-CLIP
We present CFT-CLIP, a contrastive learning framework based on counterfactual text.

This figure illustrates the CFT-CLIP framework.

We proposed the CLIP-like vision-language transformer encoder to represent the semantics of news thumbnails and news text. In addition, we implemented ..

## Datasets 
### Pretraining corpus
We used the [BBC English dataset](https://aclanthology.org/2023.eacl-main.263/).

### Evaluation Data: NewsTT
Our dataset is available upon request. Please contact ""

We introduce a dataset of 1,000 news thumbnail and text for detecting news thumbnail representativeness.
* Label 1: The thumbnail represents the news text.
* Label 0: The thumbnail not represents the news text.


## Usage
### CFT-CLIP pretraining
When a pre-training corpus exists, execute it in the order below.

#### counterfeactual text generation
```
python utils/save_pixel_values.py # Extract pixel values ​​in advance for learning speed
python utils/get_ntt.py --data_path 'train.pkl' --save_path 'train.pkl' --target_text 'summary' # Extract ntt from news text
python utils/image_text_cossine_similarity.py --data_path 'train.pkl' --save_path 'train.pkl' --target_text 'summary' # Extract CLIP cossine similarity between image-text pairs
python utils/counterfactual.py --data_path 'train.pkl' --save_path 'train.pkl' # counterfactual text generation 
```

#### pretrainig
Set configure using config.py.
```
python train.py
```
## Dataset usage
This dataset is shared under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.en). According to this license, you are free to use the dataset as long as you provide appropriate attribution
