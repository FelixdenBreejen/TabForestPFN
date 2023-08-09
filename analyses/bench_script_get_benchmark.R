args = commandArgs(trailingOnly=TRUE)
output_dir <- args
print(paste("Output dir:", output_dir))

benchmark <- read_csv("analyses/results/benchmark_total.csv")
results_modified_path <- file.path(output_dir, "results_modified_for_plotting.csv")
print(paste("Reading", results_modified_path))
new_runs <- read_csv(results_modified_path)
benchmark <- bind_rows(benchmark, new_runs)