python demonstrate.py --n_actors=8 --budget=1000000 --env_name="MiniHack-KeyRoom-S5-v0" --lambda_=0.95 --discount=0.99 --iteration_size=2048 --batch_size=128 --seed=0
python annotate.py --filepath="./demonstrations/demonstrations_1.pkl"
