FLASK_ENDPOINT="http://localhost:5000/ready"

function cleanup {
    echo "Process terminated."
    kill $SERVER_PID
    exit 1
}

function annotate {
    if [[ $1 == *"8x7B"* || $1 == *"70b"* || $1 == *"c4ai"* || $1 == *"70B"* ]]; then
        python ~/repos/lfrl/calm/lmaas/server.py --name="$1" --revision="$3" --load_in_4bit &
    else
        python ~/repos/lfrl/calm/lmaas/server.py --name="$1" --revision="$3" &
    fi
    SERVER_PID=$!
    trap cleanup SIGINT
    wait_for_server
    python annotate_prompts.py --prompts_dir="/scratch/uceeepi/calf/dataset/dataset-3/prompts-$2"
    kill $SERVER_PID
}

function is_server_ready {
    local response_code
    response_code=$(curl -s -o /dev/null -w "%{http_code}" $FLASK_ENDPOINT)
    if [[ $response_code == 200 ]]; then
        return 0
    else
        return 1
    fi
}

function wait_for_server {
    attempt=1
    while ! is_server_ready; do
        echo -ne "Waiting for server to be ready... ($attempt)\t\r"
        attempt=$((attempt + 1))
        sleep 1
    done
    echo -e "\nServer is ready!"
}

for ABLATION in "subgoals-preset-win"; do
    annotate "meta-llama/Meta-Llama-3-8B-Instruct" "$ABLATION" "refs/pr/4"
    annotate "meta-llama/Meta-Llama-3-70B-Instruct" "$ABLATION" "refs/pr/2"
    annotate "google/gemma-1.1-7b-it" "$ABLATION"
    annotate "google/gemma-7b-it" "$ABLATION"
    annotate "meta-llama/Llama-2-7b-chat-hf" "$ABLATION"
    annotate "meta-llama/Llama-2-13b-chat-hf" "$ABLATION"
    annotate "mistralai/Mistral-7B-Instruct-v0.2" "$ABLATION"
    annotate "CohereForAI/c4ai-command-r-v01" "$ABLATION"
    annotate "mistralai/Mixtral-8x7B-Instruct-v0.1" "$ABLATION"
    annotate "meta-llama/Llama-2-70b-chat-hf" "$ABLATION"
done

# ABLATION="subgoals-preset-win"

# # make dataset
# python create_dataset.py \
#     --source_path="/scratch/uceeepi/calf/demonstrations/demo_2.pkl" \
#     --dest_dir="/scratch/uceeepi/calf/dataset/dataset-3" \
#     --seq_len=2 \
#     --num_annotations=50 \
#     --ablation=$ABLATION \
#     --instruction="transition" \
#     --output="transition" \
#     --subgoals="preset" \
#     --task="win" \
#     --tty

# annotate "google/gemma-1.1-7b-it" "$ABLATION"
# annotate "google/gemma-7b-it" "$ABLATION"
# annotate "meta-llama/Meta-Llama-3-8B-Instruct" "$ABLATION" "refs/pr/4"
# annotate "meta-llama/Llama-2-7b-chat-hf" "$ABLATION"
# annotate "meta-llama/Llama-2-13b-chat-hf" "$ABLATION"
# annotate "mistralai/Mistral-7B-Instruct-v0.2" "$ABLATION"
# annotate "CohereForAI/c4ai-command-r-v01" "$ABLATION"
# annotate "mistralai/Mixtral-8x7B-Instruct-v0.1" "$ABLATION"
# annotate "meta-llama/Meta-Llama-3-70B-Instruct" "$ABLATION" "refs/pr/2"
# annotate "meta-llama/Llama-2-70b-chat-hf" "$ABLATION"

# python evaluate_dataset.py \
#     --sort="F1" \
#     --latex \
#     --evaluation="human" \
#     --source_dir="/scratch/uceeepi/calf/dataset/dataset-3/" \
#     --match=".*-$ABLATION" \
#     --highlight
# select ablation
