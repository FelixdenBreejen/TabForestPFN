while getopts g:p:s: flag
do
    case "${flag}" in
        g) gpu=${OPTARG};;
        p) project=${OPTARG};;
        s) sweep=${OPTARG};;
    esac
done
tmux new-session -d "conda activate tabularbench; export WANDB_MODE="offline"; export CUDA_VISIBLE_DEVICES=$gpu; wandb agent $project/$sweep; tmux kill-session"