# python demonstrate.py --n_actors=8 --budget=1000000 --env_name="MiniHack-KeyRoom-S5-v0" --lambda_=0.95 --discount=0.99 --iteration_size=2048 --batch_size=128 --seed=0

# python annotate.py \
# --batch_size=4 \
# --seq_len=20 \
# --experiment="experiment_2" \
# --out_name="ann_role.pkl" \
# --ablation="planned_act_with_role"

python run.py \
--n_actors=8 \
--budget=10000000 \
--env_name="MiniHack-KeyRoom-S5-v0" \
--lambda_=0.95 \
--discount=0.99 \
--iteration_size=2048 \
--batch_size=128 --seed=0 \
--annotations_path="/scratch/uceeepi/calf/experiment_2/ann_role.pkl" \
--ablation="planned_act_with_role" \
--beta=1.0

python run.py \
--n_actors=8 \
--budget=10000000 \
--env_name="MiniHack-KeyRoom-S5-v0" \
--lambda_=0.95 \
--discount=0.99 \
--iteration_size=2048 \
--batch_size=128 --seed=0 \
--annotations_path="/scratch/uceeepi/calf/experiment_2/ann_role.pkl" \
--ablation="planned_act_with_role" \
--beta=0.5

python run.py \
--n_actors=8 \
--budget=10000000 \
--env_name="MiniHack-KeyRoom-S5-v0" \
--lambda_=0.95 \
--discount=0.99 \
--iteration_size=2048 \
--batch_size=128 --seed=0 \
--annotations_path="/scratch/uceeepi/calf/experiment_2/ann_role.pkl" \
--ablation="planned_act_with_role" \
--beta=0.1