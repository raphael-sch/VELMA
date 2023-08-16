Code for the preprint:  [Schumann et al, "VELMA: Verbalization Embodiment of LLM Agents for Vision and Language Navigation in Street View"](https://arxiv.org/pdf/2307.06082.pdf)

![](navigation.gif)

Project page: https://map2seq.schumann.pub/vln/velma/


# Preparations

Download CLIP embeddings for images and landmarks from: [here](https://www.cl.uni-heidelberg.de/~rschuman/files/VELMA/CLIP-ViT-bigG-14-laion2B-39B-b160k.zip) or [here](https://map2seq-assets.schumann.pub/files/VELMA/CLIP-ViT-bigG-14-laion2B-39B-b160k.zip)<br>
Extract all files and move them into the 'features/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k' folder.

Tested with Python 3.10<br>
Install packages in requirements.txt<br><br>

In general all default arguments in the run_+ scripts should be correct.

# Inference
## Reproduce GPT-3 and GPT-4 Results in Paper:
```
python run_inference_gpt.py --num_instances -1 --dataset_name touchdown --split dev --exp_name github --image openclip --model_name openai/text-davinci-003
python run_inference_gpt.py --num_instances -1 --dataset_name map2seq --split dev --exp_name github --image openclip --model_name openai/text-davinci-003
python run_inference_gpt.py --num_instances -1 --dataset_name touchdown --split dev --exp_name github --image openclip --model_name openai/gpt-4-0314
python run_inference_gpt.py --num_instances -1 --dataset_name map2seq --split dev --exp_name github --image openclip --model_name openai/gpt-4-0314
```

All API calls to GPT-3 and GPT4 with using openclip features should be cached. Otherwise be careful with how many instances you run, they can get expensive quickly. There are up to 40 API calls per instance with ~2000 tokens each!

## Reproduce LLaMa-1, LLaMa-2, OPT  Results in Paper:
```
python run_inference_llama.py --num_instances -1 --dataset_name map2seq --split dev --model_name decapoda-research/llama-7b-hf
python run_inference_llama.py --num_instances -1 --dataset_name map2seq --split dev --model_name meta-llama/Llama-2-7b-hf --hf_auth_token {huggingface key}
python run_inference_llama.py --num_instances -1 --dataset_name map2seq --split dev --model_name facebook/opt-1.3b
```

# Finetuning
Regular finetuning:
```
python run_finetune.py --exp_name github_openclip_seed1 --dataset_name map2seq --image openclip
python run_finetune.py --exp_name github_openclip_seed1 --dataset_name touchdown --image openclip
```

Finetuning with response based learning:
```
python run_finetune_rl.py --exp_name github_openclip_seed1 --dataset_name map2seq --image openclip
python run_finetune_rl.py --exp_name github_openclip_seed1 --dataset_name touchdown --image openclip
```


# References
Code based on https://github.com/VegB/VLN-Transformer and https://github.com/raphael-sch/map2seq_vln <br>
Touchdown splits based on: https://github.com/lil-lab/touchdown  <br>
map2seq splits based on: https://map2seq.schumann.pub  <br>
Panorama images can be downloaded here: https://sites.google.com/view/streetlearn/dataset <br>

# Citation
Please cite the following paper if you use this code:

```
@article {schumann-2023-velma,
 title = "VELMA: Verbalization Embodiment of LLM Agents for Vision and Language Navigation in Street View",
 author = "Raphael Schumann and Wanrong Zhu and Weixi Feng and Tsu-Jui Fu and Stefan Riezler and William Yang Wang",
 year = "2023",
 publisher = "arXiv",
 eprint = "2307.06082" 
}
```