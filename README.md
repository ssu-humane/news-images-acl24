# Assessing News Thumbnail Representativeness: Counterfactual text can enhance the cross-modal matching ability

## Task: Assessing News Thumbnail Representativeness
Given a news thumbnail I and its news text T, our task is to predict a binary label L indicating whether news thumbnail I is representative of T or not.


## Method: CFT-CLIP
We present CFT-CLIP, a contrastive learning framework based on counterfactual text. In addition, we implemented this task of assessing thumbnail representativeness in a zero-shot setting no labeled data is available for training, we employ a thresholding classifier based on embedding similarity.

<p align="center"><img src="https://github.com/ssu-humane/fake-news-thumbnail_2/assets/76805677/634d2896-9b8d-428c-8c96-8f5c69a7afbd" width="600" height="400"></p>
This figure illustrates of the proposed method. Given a pair of a news thumbnail image and an article text, the method generates counterfactual news text for being used as negative samples for contrastive learning. CFT-CLIP is the CLIP-like vision-language transformer encoder to represent the semantics of news thumbnails and news text and it aims to improve the vision and language bi-encoder by contrastive updates involving the counterfactual text generated from an input text.


## NewsTT
This dataset is shared under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.en). According to this license, you are free to use the dataset as long as you provide appropriate attribution

Our dataset is available upon request. Please contact "yeayen789@gmail.com"

### Data collection and annotation
We introduce a dataset of 1,000 news thumbnail and text for detecting news thumbnail representativeness.
- We used 1,000 randomly sampled cases from the NELA-GT-2021 dataset out of 10,000 used as a test set in the paper.
- All datasets were annotated two male and one female student from Soongsil University.
- Labeling guidelines are available in paper Appendix F.

### Data description
* Title: Title of the news article
* Summary: Text that summarized the text of a news article using ChatGPT
* Label 1: The thumbnail represents the news text.
* Label 0: The thumbnail not represents the news text.

|Thumbnail|Title|Summary|Label|
|---|---|---|---|
|![image](https://github.com/ssu-humane/news-images-acl24/assets/76805677/7356b5e3-18b3-4e63-93d8-25aec29fc776)|In their first call, Biden presses Putin on Navalny arrest, cyberattacks, bounties on U.S. troops|President Biden confronted Russian leader Vladimir Putin about a range of issues, including the arrest of Alexei Navalny, the SolarWinds cyberattack, interference in the 2020 election, and the alleged plot to assassinate American soldiers, in their first call since Biden's inauguration.|1|
|![image](https://github.com/ssu-humane/news-images-acl24/assets/76805677/1bb52486-4474-4cc4-b158-d96d0c05cde6)|Pastor of “Progressive Church” in California works in Adult Entertainment Industry|A "Progressive Church" in California is co-pastored by a husband and his wife, who is an ordained minister and actively works in the Adult Entertainment Industry.|0|


### Data analysis
Mean counts of words, part-of-speech units, and named entities.
<p align="center"><img src=https://github.com/ssu-humane/news-images-acl24/assets/76805677/e1bf0734-2bc0-4f1e-bc52-24e8b285ddf6 width="650" height="300"></p>


## Usage
### CFT-CLIP pretraining
When a pre-training corpus exists, execute it in the order below.

### Counterfeactual text generation
```shell
python utils/save_pixel_values.py # Extract pixel values ​​in advance for learning speed
python utils/get_ntt.py --data_path 'train.pkl' --save_path 'train.pkl' --target_text 'summary' # Extract ntt from news text
python utils/image_text_cossine_similarity.py --data_path 'train.pkl' --save_path 'train.pkl' --target_text 'summary' # Extract CLIP cossine similarity between image-text pairs
python utils/counterfactual.py --data_path 'train.pkl' --save_path 'train.pkl' # counterfactual text generation 
```

### Trainig
Set configure using config.py.
```shell
python train.py
```

### Evaluation
```shell
python evaluation.py --pixel_path "data/pixel_values" ...
```

### Pretrained [CFT-CLIP](https://huggingface.co/humane-lab/cft-clip)
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


