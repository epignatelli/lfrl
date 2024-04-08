export NUM_ANN=50
export SEQ_LEN=20

for prompts_dir in /home/uceeepi/repos/lfrl/experiments/dataset/results/dataset-2/prompts-*; do
    python annotate_prompts.py \
    --prompts_dir=$prompts_dir
done
