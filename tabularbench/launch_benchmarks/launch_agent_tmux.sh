while getopts g:p:s: flag
do
    case "${flag}" in
        g) gpu=${OPTARG};;
        p) path=${OPTARG};;
        s) seed=${OPTARG};;
    esac
done
tmux new-session -d "conda activate tabularbench; export CUDA_VISIBLE_DEVICES=$gpu; python tabularbench/run_sweeps.py --sweep_csv_path=$path --seed=$seed; tmux kill-session"