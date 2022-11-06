# Aesthetic classifier training &amp; inference tools.

This repo contains pretrained models that are to be used for rating anime-styled images with a ``0`` to ``1`` score, where the lowest score means the image has a low aesthetic rating and a high score means that the image has a high aesthetic rating.

## ⚙️ Pretrained Models
| Name       | CLIP Model  | Performance |
|------------|-------------|-------------|
| aes-B32-v0 | OpenAI B-32 | 0.0266 |

## 🔑 Setup

```shell
git clone git@github.com:waifu-diffusion/aesthetic.git
cd aesthetic
python -m pip install -r requirements
```

## 🤖 Inference
```python
import torch
from transformers import CLIPModel, CLIPProcessor
from aesthetic import image_embeddings, Classifier

aesthetic_path = 'aes-B32-v0.pth'
clip_name = 'openai/clip-vit-base-patch32'
url = 'https://cdn.donmai.us/original/16/67/__klein_moretti_lord_of_the_mysteries_drawn_by_ji26725339__1667415282975e8f8c574ca26d83e3be.jpg'

clipprocessor = CLIPProcessor.from_pretrained(clip_name)
clipmodel = CLIPModel.from_pretrained(clip_name).to('cuda').eval()

aes_model = Classifier(512, 256, 1)
aes_model.load_state_dict(torch.load(aesthetic_path))

image_embeds = image_embeddings(url, clipmodel, clipprocessor)
prediction = aes_model(torch.from_numpy(image_embeds).float().to('cuda'))
print(f'Prediction: {prediction.item()}')
```

And this should be the output:

```shell
$ python3 test.py 
Prediction: 0.999903678894043
```
