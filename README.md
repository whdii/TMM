# Transferable Multimodal Attack on Vision-Language Pre-training Models

This is the official PyTorch implementation of the paper "Transferable Multimodal Attack on Vision-Language Pre-training Models".

## Requirements
- pytorch 1.10.2
- transformers 4.8.1
- timm 0.4.9
- bert_score 0.3.11

## Download
- Dataset json files for downstream tasks [[ALBEF github]](https://github.com/salesforce/ALBEF)
- Finetuned checkpoint for ALBEF [[ALBEF github]](https://github.com/salesforce/ALBEF)
- Finetuned checkpoint for TCL [[TCL github]](https://github.com/uta-smile/TCL)
- Finetuned checkpoint for X-VLM [[X-VLM github]](https://github.com/zengyan-97/X-VLM)
- Finetuned checkpoint for ViLT [[ViLT github]](https://github.com/dandelin/ViLT)
- Finetuned checkpoint for METER [[METER github]](https://github.com/zdou0830/METER)

## Attack Multimodal Embedding
```
python EvalTransferAttack.py --adv 1 --gpu 0 \
--config ./configs/Retrieval_flickr.yaml \
--output_dir ./output/Retrieval_flickr \
--checkpoint [Finetuned checkpoint]
--log_name [log_name]
--save_json_name [save_json_name]
--config_name [config_name]
--save_dir [save_dir]
```
