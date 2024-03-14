# python demonstrate.py --n_actors=8 --budget=1000000 --env_name="MiniHack-KeyRoom-S5-v0" --lambda_=0.95 --discount=0.99 --iteration_size=2048 --batch_size=128 --seed=0
python annotate.py \
--batch_size=4 \
--experiment="experiment_2" \
--out_name="ann_full.pkl"

python run.py \
--n_actors=8 \
--budget=10000000 \
--env_name="MiniHack-KeyRoom-S5-v0" \
--lambda_=0.95 \
--discount=0.99 \
--iteration_size=2048 \
--batch_size=128 --seed=0 \
--annotations_path="/scratch/uceeepi/calf/experiment_2/ann_full.pkl"
--ablation="planned_action"