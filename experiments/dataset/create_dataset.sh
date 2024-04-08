export NUM_ANN=50
export SEQ_LEN=20

# ablate role
ABLATION="role=analyst"
python create_dataset.py \
    --dest_dir="/scratch/uceeepi/calf/dataset/dataset-2" \
    --source_path="/scratch/uceeepi/calf/demonstrations/demo_2.pkl" \
    --output="generic" \
    --seq_len=$SEQ_LEN \
    --num_annotations=$NUM_ANN \
    --ablation=$ABLATION \
    --role="analyst"

# tasks
ABLATION="task-win"
python create_dataset.py \
    --dest_dir="/scratch/uceeepi/calf/dataset/dataset-2" \
    --source_path="/scratch/uceeepi/calf/demonstrations/demo_2.pkl" \
    --output="generic" \
    --seq_len=$SEQ_LEN \
    --num_annotations=$NUM_ANN \
    --ablation=$ABLATION \
    --task="win"

# subgoals
ABLATION="subgoals-preset"
python create_dataset.py \
    --dest_dir="/scratch/uceeepi/calf/dataset/dataset-2" \
    --source_path="/scratch/uceeepi/calf/demonstrations/demo_2.pkl" \
    --output="generic" \
    --seq_len=$SEQ_LEN \
    --num_annotations=$NUM_ANN \
    --ablation=$ABLATION \
    --subgoals="preset"

# instructions
for instr in "identify" "progress" "optimality"; do
    ABLATION="instr-$instr"
    python create_dataset.py \
        --dest_dir="/scratch/uceeepi/calf/dataset/dataset-2" \
        --source_path="/scratch/uceeepi/calf/demonstrations/demo_2.pkl" \
        --output="generic" \
        --seq_len=$SEQ_LEN \
        --num_annotations=$NUM_ANN \
        --ablation=$ABLATION \
        --instruction=$instr
done

# remark
ABLATION="remark-message-action"
python create_dataset.py \
    --dest_dir="/scratch/uceeepi/calf/dataset/dataset-2" \
    --source_path="/scratch/uceeepi/calf/demonstrations/demo_2.pkl" \
    --output="generic" \
    --seq_len=$SEQ_LEN \
    --num_annotations=$NUM_ANN \
    --ablation=$ABLATION \
    --remark="message_action"

# ablate actions
ABLATION="ablate-action"
python create_dataset.py \
    --dest_dir="/scratch/uceeepi/calf/dataset/dataset-2" \
    --source_path="/scratch/uceeepi/calf/demonstrations/demo_2.pkl" \
    --output="generic" \
    --seq_len=$SEQ_LEN \
    --num_annotations=$NUM_ANN \
    --ablation=$ABLATION \
    --ablate_action

# ablate message
ABLATION="ablate-message"
python create_dataset.py \
    --dest_dir="/scratch/uceeepi/calf/dataset/dataset-2" \
    --source_path="/scratch/uceeepi/calf/demonstrations/demo_2.pkl" \
    --output="generic" \
    --seq_len=$SEQ_LEN \
    --num_annotations=$NUM_ANN \
    --ablation=$ABLATION \
    --ablate_message

# ablate action and message
ABLATION="ablate-message-action"
python create_dataset.py \
    --dest_dir="/scratch/uceeepi/calf/dataset/dataset-2" \
    --source_path="/scratch/uceeepi/calf/demonstrations/demo_2.pkl" \
    --output="generic" \
    --seq_len=$SEQ_LEN \
    --num_annotations=$NUM_ANN \
    --ablation=$ABLATION \
    --ablate_message \
    --ablate_action

# tokenisation
ABLATION="token-separator-none"
python create_dataset.py \
    --dest_dir="/scratch/uceeepi/calf/dataset/dataset-2" \
    --source_path="/scratch/uceeepi/calf/demonstrations/demo_2.pkl" \
    --output="generic" \
    --seq_len=$SEQ_LEN \
    --num_annotations=$NUM_ANN \
    --ablation=$ABLATION \
    --token_separator=""

# tty
ABLATION="tty"
python create_dataset.py \
    --dest_dir="/scratch/uceeepi/calf/dataset/dataset-2" \
    --source_path="/scratch/uceeepi/calf/demonstrations/demo_2.pkl" \
    --output="generic" \
    --seq_len=$SEQ_LEN \
    --num_annotations=$NUM_ANN \
    --ablation=$ABLATION \
    --tty

# cherry picked
for instr in "identify" "progress" "optimality"; do
    ABLATION="instr-$instr-task-win-tty-role-analyst"
    python create_dataset.py \
        --dest_dir="/scratch/uceeepi/calf/dataset/dataset-2" \
        --source_path="/scratch/uceeepi/calf/demonstrations/demo_2.pkl" \
        --output="generic" \
        --seq_len=$SEQ_LEN \
        --num_annotations=$NUM_ANN \
        --ablation=$ABLATION \
        --instruction=$instr \
        --task="win" \
        --role="analyst" \
        --tty
done

ABLATION="task-win-tty-role-analyst-sep-none"
python create_dataset.py \
    --dest_dir="/scratch/uceeepi/calf/dataset/dataset-2" \
    --source_path="/scratch/uceeepi/calf/demonstrations/demo_2.pkl" \
    --output="generic" \
    --seq_len=$SEQ_LEN \
    --num_annotations=$NUM_ANN \
    --ablation=$ABLATION \
    --task="win" \
    --role="analyst" \
    --tty \
    --token_separator=""
