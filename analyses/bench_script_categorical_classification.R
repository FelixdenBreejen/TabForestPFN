source("analyses/plot_utils.R")
source("analyses/bench_script_get_benchmark.R")

###################################
# Benchmark classif categorical medium

df <- benchmark %>% 
  filter(benchmark == "categorical_classification_medium")

# checks
checks(df)

# Dataset by dataset

plot_results_per_dataset(df, "accuracy", default_colscale = T)

benchmark_datasets_path <- file.path(output_dir, "benchmark_datasets.pdf")
ggsave(benchmark_datasets_path, width=15, height=10, bg="white")
print(paste("Saved in", benchmark_datasets_path))

# Aggregated
benchmark_poster_path <- file.path(output_dir, "benchmark_poster.pdf")
plot_aggregated_results(df, y_inf=0.5, y_sup=0.95, score="accuracy", quantile=0.1, truncate_scores = F, text_size=8, theme_size=25, max_iter=400)
ggsave(benchmark_poster_path, width=13.5, height=7, bg="white")
print(paste("Saved in", benchmark_poster_path))

benchmark_path <- file.path(output_dir, "benchmark.pdf")
plot_aggregated_results(df, y_inf=0.5, y_sup=0.95, score="accuracy", max_iter=400)
ggsave(benchmark_path, width=7, height=6, bg="white")
print(paste("Saved in", benchmark_path))

