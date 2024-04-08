./create_dataset.sh

function annotate {
    python ~/repos/lfrl/calm/lmaas/server.py --name="$1" & 
    SERVER_PID=$!
    sleep 30s
    ./annotate_prompts.sh
    kill $SERVER_PID
}

# gemma
annotate "google/gemma-7b-it"
# command-r
annotate "CohereForAI/c4ai-command-r-v01"
# openchat gemma
annotate "openchat/openchat-3.5-0106-gemma"
# openchat
annotate "openchat/openchat-3.5-0106"
