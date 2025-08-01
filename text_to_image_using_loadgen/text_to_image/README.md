# MLPerf Inference - Text to Image - stageai backend
This implementation runs text to image model with stageai backend.

Currently it supports the following models:
- stable-diffusion-xl

### SingleStream (Accuracy)

#### Setup
```
source ~/venv/bin/activate
cd ~/axs2stg/text_to_image
```

#### Accuracy

```
python3 main.py \
  --dataset "coco-1024" \
  --dataset-path coco2014 \
  --profile stable-diffusion-xl-stageai \
  --model-path stabilityai/stable-diffusion-xl-base-1.0 \
  --device cuda \
  --scenario SingleStream \
  --axs_device_id 1 \
  --backend stageai \
  --dtype fp16 \
  --accuracy \
  --stageai-model-path /home/ubuntu/engines_10.11.0.33_NVIDIA-H100-80GB-HBM3/ \
  --output accuracy_results
```

#### Validation (generate 10 images for compliance)
```
python tools/accuracy_coco.py \
  --mlperf-accuracy-file accuracy_results/mlperf_log_accuracy.json \
  --caption-path coco2014/captions/captions_source.tsv \
  --device gpu \
  --compliance-images-path ./stageai_latents_tensor
``` 
