#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# datasets=('bird200' 'car198' 'food101' 'pet37' 'ImageNet')
# task='far'
# L=500

datasets=('ImageNet10' 'ImageNet20' 'ImageNet')
task='near'
L=3

# datasets=('cub100_ID' 'car98_ID', 'food50_ID' 'pet18_ID')
# task='fine_grained'
# L=500

for dataset in "${datasets[@]}"
do
  for i in {0..3}; do
    echo "Running experiment with dataset=${dataset}, iteration=${i}, model=CLIP"
    python eval_ood_detection.py \
      --llm_model 'gpt-3.5-turbo-16k' \
      --ood_task "${task}" \
      --score_ablation "EOE" \
      --L "${L}" \
      --in_dataset "${dataset}" \
      --score 'EOE' \
      --json_number ${i} \
      --model CLIP \
      --CLIP_ckpt ViT-B/16 \
      --beta 0.25 \
      # --generate_class # You can directly comment `generate_class` if you want to use the generated classes from JSON file
  done
done



for dataset in "${datasets[@]}"
do
    echo "Running experiment with dataset=${dataset}"
    python eval_ood_detection.py \
    --ood_task "${task}"  \
    --in_dataset "${dataset}" \
    --score 'MCM' 

    echo "Running experiment with dataset=${dataset}"
    python eval_ood_detection.py \
    --ood_task "${task}"  \
    --in_dataset "${dataset}" \
    --score 'max-logit' 

    echo "Running experiment with dataset=${dataset}"
    python eval_ood_detection.py \
    --ood_task "${task}"  \
    --in_dataset "${dataset}" \
    --score 'energy' \
    --T 0.01
done