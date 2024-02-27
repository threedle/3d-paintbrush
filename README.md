# 🎨🖌️ 3D Paintbrush 

*[Dale Decatur](https://ddecatur.github.io/), [Itai Lang](https://itailang.github.io/), [Kfir Aberman](https://kfiraberman.github.io/), [Rana Hanocka](https://people.cs.uchicago.edu/~ranahanocka/)*

<a href="https://threedle.github.io/3d-paintbrush/"><img src="https://img.shields.io/website?down_color=lightgrey&down_message=offline&label=Project%20Page&up_color=lightgreen&up_message=online&url=https%3A//threedle.github.io/3d-paintbrush" height=22></a>
<a href="https://arxiv.org/abs/2311.09571"><img src="https://img.shields.io/badge/arXiv-3DPaintbrush-b31b1b.svg" height=22></a>

<!-- [[Project Page](https://threedle.github.io/3d-paintbrush/)] [[ArXiv](https://arxiv.org/abs/2311.09571)] -->

![](https://github.com/threedle/3d-paintbrush/raw/docs/docs/spot/gold_chain_necklace.gif)![](https://github.com/threedle/3d-paintbrush/raw/docs/docs/spot/heart-shaped_sunglasses.gif)![](https://github.com/threedle/3d-paintbrush/raw/docs/docs/spot/colorful_crochet_hat.gif)

### Abstract
*In this work we develop 3D Paintbrush, a technique for automatically texturing local semantic regions on meshes via text descriptions.
Our method is designed to operate directly on meshes, producing texture maps which seamlessly integrate into standard graphics pipelines.
We opt to simultaneously produce a localization map (to specify the edit region) and a texture map which conforms to it.
This synergistic approach improves the quality of both the localization and the stylization.
To enhance the details and resolution of the textured area, we leverage multiple stages of a cascaded diffusion model to supervise our local editing technique with generative priors learned from images at different resolutions.
Our technique, referred to as Cascaded Score Distillation (CSD), simultaneously distills scores at multiple resolutions in a cascaded fashion, enabling control over both the granularity and global understanding of the supervision.
We demonstrate the effectiveness of 3D Paintbrush to locally texture a variety of shapes within different semantic regions.*

## CSD Demo Notebook
To see a demo of the Cascaded Score Distillation (CSD) loss, check out this [notebook](src/csd_demo.ipynb) applying CSD to image generation and image editing. CSD enables us to supervise our optimization with multiple cascaded stages of the diffusion model instead of just the base stage used in standard SDS. We distill scores across multiple stages of a cascaded diffusion model simultaneously in order to leverage both the global awareness of the first stage and the higher level of detail contained in later stages.

## Getting Started
### Requirements
- 48 GB GPU
- CUDA 11.3
- Python 3.10

If you have less than 48 GB of GPU memory, you can still run the code, see the section on [memory optimization](#memory-optimization) for more details.

### Setup environemnt
First create the conda environment:
```
conda create -n "3d-paintbrush" python=3.10
```
and activate it with:
```
conda activate 3d-paintbrush
```
Next install the required packages by running the install script. **Make sure to run this script with access to a GPU**.
```
bash ./install_environment.sh
```

### Login to Hugging Face (to use DeepFloyd IF w/ Diffusers)
Instructions from [DeepFloyd IF](https://github.com/deep-floyd/IF):
1) If you do not already have one, create a [Hugging Face account](https://huggingface.co/join)
2) Accept the license on the model card of [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0)
3) Log in to Hugging face locally. First install `huggingface_hub`
```
pip install huggingface_hub --upgrade
```
run the login function in a python shell
```
from huggingface_hub import login

login()
```
and enter your [Hugging Face Hub access token](https://huggingface.co/docs/hub/security-tokens#what-are-user-access-tokens).

## Reproduce paper results
### (Optional) From Pre-trained
To use our pre-trained models download both the `trained_models` and `inverse_map_cache` folders from [here](https://drive.google.com/drive/folders/1kEHMtwgIGzv94Xbb8J-UaAdQ-an7drkd?usp=sharing) and add them under the data folder to create the following directory structure:
```
├── data
│   ├── inverse_map_cache
│   ├── trained_models
│   ├── hand.obj
...
│   ├── spot.obj
```
To run the pre-trained models, use the commands below. Results will be saved at `results/[name-of-mesh]/[name-of-edit]/renders/infernce.png`.

Spot:
```
python src/main.py --config_path demo/spot/gold_chain_necklace.yaml --log.inference true --log.model_path ./data/trained_models/spot/gold_chain_necklace.pth
python src/main.py --config_path demo/spot/heart-shaped_sunglasses.yaml --log.inference true --log.model_path ./data/trained_models/spot/heart-shaped_sunglasses.pth
python src/main.py --config_path demo/spot/colorful_crochet_hat.yaml --log.inference true --log.model_path ./data/trained_models/spot/colorful_crochet_hat.pth
```
<p align="left">
  <img src="https://github.com/threedle/3d-paintbrush/raw/docs/docs/spot/gold_chain_necklace_prob.png" width="133px"/>
  <img src="https://github.com/threedle/3d-paintbrush/raw/docs/docs/spot/gold_chain_necklace.png" width="133px"/>
  <img src="https://github.com/threedle/3d-paintbrush/raw/docs/docs/spot/heart-shaped_sunglasses_prob.png" width="133px"/>
  <img src="https://github.com/threedle/3d-paintbrush/raw/docs/docs/spot/heart-shaped_sunglasses.png" width="133px"/>
  <img src="https://github.com/threedle/3d-paintbrush/raw/docs/docs/spot/colorful_crochet_hat_prob.png" width="133px"/>
  <img src="https://github.com/threedle/3d-paintbrush/raw/docs/docs/spot/colorful_crochet_hat.png" width="133px"/>
</p>

Person:
```
python src/main.py --config_path demo/person/tie-dye_apron.yaml --log.inference true --log.model_path ./data/trained_models/person/tie-dye_apron.pth
python src/main.py --config_path demo/person/colorful_polo_shirt.yaml --log.inference true --log.model_path ./data/trained_models/person/colorful_polo_shirt.pth
python src/main.py --config_path demo/person/superman_chest_emblem.yaml --log.inference true --log.model_path ./data/trained_models/person/superman_chest_emblem.pth
```
<p align="left">
  <img src="https://github.com/threedle/3d-paintbrush/raw/docs/docs/person/tie-dye_apron_prob.png" width="133px"/>
  <img src="https://github.com/threedle/3d-paintbrush/raw/docs/docs/person/tie-dye_apron.png" width="133px"/>
  <img src="https://github.com/threedle/3d-paintbrush/raw/docs/docs/person/colorful_polo_shirt_prob.png" width="133px"/>
  <img src="https://github.com/threedle/3d-paintbrush/raw/docs/docs/person/colorful_polo_shirt.png" width="133px"/>
  <img src="https://github.com/threedle/3d-paintbrush/raw/docs/docs/person/superman_emblem_prob.png" width="133px"/>
  <img src="https://github.com/threedle/3d-paintbrush/raw/docs/docs/person/superman_emblem.png" width="133px"/>
</p>

Lego Minfigure:
```
python src/main.py --config_path demo/lego_minifig/barcelona_jersey.yaml --log.inference true --log.model_path ./data/trained_models/lego_minifig/barcelona_jersey.pth
python src/main.py --config_path demo/lego_minifig/blue_denim_overalls.yaml --log.inference true --log.model_path ./data/trained_models/lego_minifig/blue_denim_overalls.pth
python src/main.py --config_path demo/lego_minifig/red_bow_tie.yaml --log.inference true --log.model_path ./data/trained_models/lego_minifig/red_bow_tie.pth
```
<p align="left">
  <img src="https://github.com/threedle/3d-paintbrush/raw/docs/docs/lego_minifig/barcelona_jersey_prob.png" width="133px"/>
  <img src="https://github.com/threedle/3d-paintbrush/raw/docs/docs/lego_minifig/barcelona_jersey.png" width="133px"/>
  <img src="https://github.com/threedle/3d-paintbrush/raw/docs/docs/lego_minifig/blue_denim_overalls_prob.png" width="133px"/>
  <img src="https://github.com/threedle/3d-paintbrush/raw/docs/docs/lego_minifig/blue_denim_overalls.png" width="133px"/>
  <img src="https://github.com/threedle/3d-paintbrush/raw/docs/docs/lego_minifig/red_bow_tie_prob.png" width="133px"/>
  <img src="https://github.com/threedle/3d-paintbrush/raw/docs/docs/lego_minifig/red_bow_tie.png" width="133px"/>
</p>

Hand:
```
python src/main.py --config_path demo/hand/fancy_gold_watch.yaml --log.inference true --log.model_path ./data/trained_models/hand/fancy_gold_watch.pth
```
<p align="left">
  <img src="https://github.com/threedle/3d-paintbrush/raw/docs/docs/hand/fancy_gold_watch_prob.png" width="133px"/>
  <img src="https://github.com/threedle/3d-paintbrush/raw/docs/docs/hand/fancy_gold_watch.png" width="133px"/>
</p>

### From Scratch
To reproduce results from the paper from scratch, just pass a predefined demo config file. Results will be saved at `results/[name-of-mesh]/[name-of-edit]`.

Spot:
```
python src/main.py --config_path demo/spot/gold_chain_necklace.yaml
python src/main.py --config_path demo/spot/heart-shaped_sunglasses.yaml
python src/main.py --config_path demo/spot/colorful_crochet_hat.yaml
```

Person:
```
python src/main.py --config_path demo/person/tie-dye_apron.yaml
python src/main.py --config_path demo/person/colorful_polo.yaml
python src/main.py --config_path demo/person/superman_chest_emblem.yaml
```

Lego Minifigure:
```
python src/main.py --config_path demo/lego_minifig/barcelona_jersey.yaml
python src/main.py --config_path demo/lego_minifig/blue_denim_overalls.yaml
python src/main.py --config_path demo/lego_minifig/red_bow_tie.yaml
```

Hand:
```
python src/main.py --config_path demo/hand/fancy_gold_watch.yaml
```

## Run your own examples
To run your own examples you can create your own config files and pass those as done in the prior section. Additionally, you can instead pass values for any of the fields in `src/configs/train_config` as command line arguments. For example, to run the hand example without passing a config file, you may call:
```
python src/main.py --log.exp_dir results/hand/fancy_gold_watch --mesh.path ./data/spot.obj --guidance.object_name "hand" --guidance.style "fancy gold" --guidance.edit "watch"
```

## Memory Optimization
If you do not have access to a 48 GB GPU, you can...
1) Enable CPU offloading by setting the flag `cpu_offload` to `True` in `src/configs/guidance_config.py`. This will significanly reduce memory usage, but comes at the cost of speed.
2) Use a smaller batch size by changing the `batch_size` parameter either with the command line argument `--optim.batch_size` or in a custom config file. While this reduces memory usage, it can affect the accuracy.
3) Sample a subset of surface points each iteration (set `sample_points` to `True` and adjust `mlp_batch_size`). While this reduces memory usage, it can affect the accuracy.
4) Turn off batched score distillation calls (set `batched_sd` to `False`). This will reduce memory usage, but comes at the cost of speed.

## Acknowledgements
Our codebase is based on [Latent-NeRF/Latent-Paint](https://github.com/eladrich/latent-nerf) and our CSD guidance code is structured in the format of [ThreeStudio](https://github.com/threestudio-project/threestudio)'s guidance modules. We thank these authors for their amazing work.

## Citation
If you find this code helpful for your research, please cite our paper [3D Paintbrush: Local Stylization of 3D Shapes with Cascaded Score Distillation](https://arxiv.org/abs/2311.09571).

```
@article{decatur2023paintbrush,
    author = {Decatur, Dale and Lang, Itai and Aberman, Kfir and Hanocka, Rana},
    title  = {3D Paintbrush: Local Stylization of 3D Shapes with Cascaded Score Distillation},
    journal = {arXiv},
    year = {2023}
}
```
