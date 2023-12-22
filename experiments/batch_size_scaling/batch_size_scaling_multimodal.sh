for quantisation in 0 4 8; do
    for dtype in "torch.bfloat16" "torch.float16" "torch.float32"; do
        for model_name in $(cat ./models_multimodal.txt); do
            python batch_size_scaling_multimodal.py --model_name $model_name --dtype $dtype --quantisation $quantisation
        done
    done
done
