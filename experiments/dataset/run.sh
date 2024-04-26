FLASK_ENDPOINT="http://localhost:5000/ready"

function cleanup {
    echo "Process terminated."
    kill $SERVER_PID
    exit 1
}

function annotate {
    echo "Annotating prompts at $2 with $1"
    if [[ $1 == *"8x7B"* || $1 == *"70b"* || $1 == *"c4ai"* || $1 == *"70B"* ]]; then
        python ~/repos/lfrl/calm/lmaas/server.py --name="$1" --load_in_4bit &
    else
        python ~/repos/lfrl/calm/lmaas/server.py --name="$1" &
    fi
    SERVER_PID=$!
    trap cleanup SIGINT
    wait_for_server
    python annotate_prompts.py --prompts_dir="$2"
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

function annotate_all_models {
    annotate "google/gemma-1.1-7b-it" "$1"
    annotate "google/gemma-7b-it" "$1"
    annotate "meta-llama/Meta-Llama-3-8B-Instruct" "$1"
    annotate "meta-llama/Meta-Llama-3-70B-Instruct" "$1"
    annotate "meta-llama/Llama-2-7b-chat-hf" "$1"
    annotate "meta-llama/Llama-2-13b-chat-hf" "$1"
    annotate "mistralai/Mistral-7B-Instruct-v0.2" "$1"
    annotate "mistralai/Mixtral-8x7B-Instruct-v0.1" "$1"
    annotate "CohereForAI/c4ai-command-r-v01" "$1"
}

# select ablation
ABLATION="subgoals-minihack-role-keyroom-transition-action-tty"
PROMPTS_DIR="/scratch/uceeepi/calf/dataset/dataset-3"
# python create_dataset.py \
#     --source_path="/scratch/uceeepi/calf/demonstrations/demo_2.pkl" \
#     --dest_dir=$PROMPTS_DIR \
#     --seq_len=2 \
#     --num_annotations=50 \
#     --ablation=$ABLATION \
#     --role="generic_keyroom" \
#     --task="win" \
#     --subgoals="identify_minihack" \
#     --instruction="transition_action" \
#     --output="transition" \
#     --tty
annotate_all_models "$PROMPTS_DIR/prompts-$ABLATION"
