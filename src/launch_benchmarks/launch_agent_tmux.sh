while getopts g:s: flag
do
    case "${flag}" in
        g) gpu=${OPTARG};;
        s) sweep=${OPTARG};;
        f) fullname=${OPTARG};;
    esac
done
tmux new-session -d "conda activate tabularbench; cd src; export CUDA_VISIBLE_DEVICES=$gpu; wandb agent felixatkaist/mlp_pwl_benchmark/$sweep"