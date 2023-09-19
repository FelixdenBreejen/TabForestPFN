#!/usr/bin/env bash
python main.py hyperparams/tabpfn_finetune=tabpfn_zeroshot_1k.yaml 'model_plot_names=[TabPFN Zeroshot 1k]'
python main.py hyperparams/tabpfn_finetune=tabpfn_zeroshot_10k.yaml 'model_plot_names=[TabPFN Zeroshot 10k]'
python main.py hyperparams/tabpfn_finetune=tabpfn_finetune_1k.yaml 'model_plot_names=[TabPFN Finetune 1k]'
python main.py hyperparams/tabpfn_finetune=tabpfn_finetune_10k.yaml 'model_plot_names=[TabPFN Finetune 10k]'
python main.py hyperparams/tabpfn_finetune=tabpfn_scratch_10k.yaml 'model_plot_names=[TabPFN Scratch]'