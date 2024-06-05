# Assessing News Thumbnail Representativeness: Counterfactual text can enhance the cross-modal matching ability

## Task: Assessing News Thumbnail Representativeness
Given a news thumbnail I and its news text T, our task is to predict a binary label L indicating whether news thumbnail I is representative of T or not.


## Method: CFT-CLIP
We present CFT-CLIP, a contrastive learning framework based on counterfactual text. In addition, we implemented this task of assessing thumbnail representativeness in a zero-shot setting no labeled data is available for training, we employ a thresholding classifier based on embedding similarity.

<p align="center"><img src="https://github.com/ssu-humane/fake-news-thumbnail_2/assets/76805677/634d2896-9b8d-428c-8c96-8f5c69a7afbd" width="600" height="400"></p>
CFT-CLIP is the CLIP-like vision-language transformer encoder to represent the semantics of news thumbnails and news text and it aims to improve the vision and language bi-encoder by contrastive updates involving the counterfactual text generated from an input text. This figure illustrates of the proposed method. Given a pair of a news thumbnail image and an article, the method generates counterfactual news text for being used as negative samples for contrastive learning.


## Datasets 
### Pretraining corpus
We used the [BBC English dataset](https://aclanthology.org/2023.eacl-main.263/).

### Evaluation Data: NewsTT
Our dataset is available upon request. Please contact "yeayen789@gmail.com"

We introduce a dataset of 1,000 news thumbnail and text for detecting news thumbnail representativeness.
* Label 1: The thumbnail represents the news text.
* Label 0: The thumbnail not represents the news text.


## Usage
### CFT-CLIP pretraining
When a pre-training corpus exists, execute it in the order below.

#### Counterfeactual text generation
```shell
python utils/save_pixel_values.py # Extract pixel values ​​in advance for learning speed
python utils/get_ntt.py --data_path 'train.pkl' --save_path 'train.pkl' --target_text 'summary' # Extract ntt from news text
python utils/image_text_cossine_similarity.py --data_path 'train.pkl' --save_path 'train.pkl' --target_text 'summary' # Extract CLIP cossine similarity between image-text pairs
python utils/counterfactual.py --data_path 'train.pkl' --save_path 'train.pkl' # counterfactual text generation 
```

#### Trainig
Set configure using config.py.
```shell
python train.py
```

### Pretrained CFT-CLIP
```python3
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

processor = AutoProcessor.from_pretrained("humane-lab/cft-clip")
model = AutoModel.from_pretrained("humane-lab/cft-clip")


image = "cat.jpg"
image = Image.open(image)
inputs = processor(text=["this is a cat"], images=image, return_tensors="pt")

outputs = model(**inputs)
text_embeds = outputs.text_embeds
image_embeds = outputs.image_embeds
```

## Dataset usage
This dataset is shared under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.en). According to this license, you are free to use the dataset as long as you provide appropriate attribution
