declare -a envs=("seaquest-mixed-v0")
# "breakout-medium-v0" "qbert-medium-v0"
# "seaquest-medium-v0" "breakout-mixed-v0"
# "qbert-mixed-v0" "seaquest-mixed-v0"
for seed in 1 2 3; do
for env in "${envs[@]}"; do
    python iql.py --dataset "$env" --gpu 1 --ratio 0.05 --seed "$seed" &
done
done