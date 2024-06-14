# Assessing News Thumbnail Representativeness

This repository provides the dataset and code for our paper, "Assessing News Thumbnail Representativeness: Counterfactual text can enhance the cross-matching ability," to be published at ACL 2024.

## Task

Given a news thumbnail image *I* and its news text *T*, the task is to predict a binary label *L* indicating whether a news thumbnail image *I* portrays the actor of a news event, which can be identified from *T*.

## Dataset: NewsTT 

We introduce a dataset of 1,000 news thumbnail images and text for the task, along with high-quality labels.
This dataset is intended for the zero-shot evaluation of vision language models.

|Image|Title|Summary|Label|
|---|---|---|---|
|![image](https://github.com/ssu-humane/news-images-acl24/assets/76805677/7356b5e3-18b3-4e63-93d8-25aec29fc776)|In their first call, Biden presses Putin on Navalny arrest, cyberattacks, bounties on U.S. troops|President Biden confronted Russian leader Vladimir Putin about a range of issues, including the arrest of Alexei Navalny, the SolarWinds cyberattack, interference in the 2020 election, and the alleged plot to assassinate American soldiers, in their first call since Biden's inauguration.|1|
|![image](https://github.com/ssu-humane/news-images-acl24/assets/76805677/1bb52486-4474-4cc4-b158-d96d0c05cde6)|Pastor of “Progressive Church” in California works in Adult Entertainment Industry|A "Progressive Church" in California is co-pastored by a husband and his wife, who is an ordained minister and actively works in the Adult Entertainment Industry.|0|

* Image: news thumbnail image
* Title: news headline 
* Summary: summarized body text, done by ChatGPT
* Label
  - 1: the image portrays at least one actor of the news event
  - 0: the image does not present any actor of the news event

The dataset is available upon request: [[**LINK**]](https://forms.gle/reGnAXrY84XKLpvc7)

## Method: CFT-CLIP

We present CFT-CLIP, a contrastive learning framework that uses counterfactual text to update vision and language bi-encoders. 

<p align="center"><img src="https://github.com/ssu-humane/fake-news-thumbnail_2/assets/76805677/634d2896-9b8d-428c-8c96-8f5c69a7afbd" width="600" height="400"></p>
This figure illustrates the key idea of the proposed method. Given a pair of a news thumbnail image and an article, the method generates counterfactual news text and uses it as negative samples for contrastive learning. CFT-CLIP is a CLIP-like vision-language transformer encoder that represents the semantics of news thumbnails and news text. It aims to improve the vision and language bi-encoder by contrastive updates involving the counterfactual text generated from an input text.

### Model usage

You can use the pretrained checkpoint available at HuggingFace Hub.
[[**LINK**]](https://huggingface.co/humane-lab/cft-clip)

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

### Code

### Counterfactual text generation
```shell
python utils/save_pixel_values.py # Extract pixel values ​​in advance for learning speed
python utils/get_ntt.py --data_path 'train.pkl' --save_path 'train.pkl' --target_text 'summary' # Extract ntt from news text
python utils/image_text_cossine_similarity.py --data_path 'train.pkl' --save_path 'train.pkl' --target_text 'summary' # Extract CLIP cossine similarity between image-text pairs
python utils/counterfactual.py --data_path 'train.pkl' --save_path 'train.pkl' # counterfactual text generation 
```
### Training
Set configure using config.py.
```shell
python train.py
```
### Evaluation
```shell
python evaluation.py --pixel_path "data/pixel_values" ...
```

## Attribution

The code and dataset are shared under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.en). 
You are free to use the resources for the non-commercial purpose.

```
@article{yoon2024assessing,
  title={Assessing News Thumbnail Representativeness: Counterfactual text can enhance the cross-modal matching ability},
  author={Yoon, Yejun and Yoon, Seunghyun and Park, Kunwoo},
  journal={arXiv preprint arXiv:2402.11159},
  year={2024}
}
```

