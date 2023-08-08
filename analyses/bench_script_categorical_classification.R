source("analyses/plot_utils.R")

# args = commandArgs(trailingOnly=TRUE)
# benchmark <- read_csv(args[0])

benchmark <- read_csv("analyses/results/benchmark_total.csv")
new_runs <- read_csv("outputs/2023-08-08/20-13-22/categorical_classification_random_mlp_pwl/results_modified.csv")
benchmark <- bind_rows(benchmark, new_runs)

###################################
# Benchmark classif categorical medium

df <- benchmark %>% 
  filter(benchmark == "categorical_classification_medium")

# checks
checks(df)

# Dataset by dataset

plot_results_per_dataset(df, "accuracy", default_colscale = T)


ggsave("outputs/2023-08-08/20-13-22/categorical_classification_random_mlp_pwl/benchmark_datasets.pdf", width=15, height=10, bg="white")

# Aggregated
plot_aggregated_results(df, y_inf=0.5, y_sup=0.95, score="accuracy", quantile=0.1, truncate_scores = F, text_size=8, theme_size=25, max_iter=400)
ggsave("outputs/2023-08-08/20-13-22/categorical_classification_random_mlp_pwl/benchmark_poster.pdf", width=13.5, height=7, bg="white")

plot_aggregated_results(df, y_inf=0.5, y_sup=0.95, score="accuracy", max_iter=400)
ggsave("outputs/2023-08-08/20-13-22/categorical_classification_random_mlp_pwl/benchmark.pdf", width=7, height=6, bg="white")

