#!/bin/bash
# NOTE: Should be executed from the root of the repository.
# This script generates a screenshot of the environment with all points.

while getopts w:h:o: flag
do
    case "${flag}" in
        w) width=${OPTARG};;
        h) height=${OPTARG};;
        o) save_dir=${OPTARG};;
        ?)
            echo "Usage: $(basename \$0) [-w width] [-h height] [-o save_dir]"
            exit 1
            ;;
    esac
done

DIR="${save_dir}/${width}x${height}"

for mode in "val_dataset" "col_dataset"; do
    RESULTS_DIR="${DIR}/results/${mode}"
    mkdir -p "${RESULTS_DIR}"

    ENV_DIR="${DIR}/env/${mode}"

    poetry run python -m experiments.utils.reforestree.env_runner \
            --graph_file "${ENV_DIR}/obstacle_graph.pkl" \
            --graph_metadata_file "${ENV_DIR}/obstacle_graph_metadata.json" \
            --point_labels_file "${ENV_DIR}/point_labels.pkl" \
            --mode "screenshot" \
            --screenshot_output_file "${RESULTS_DIR}/screenshot.png"
done
