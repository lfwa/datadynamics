#!/bin/bash
# NOTE: Should be executed from the root of the repository.
# This script runs a greedy and premade simulation for a given environment.

while getopts w:h:d:o: flag
do
    case "${flag}" in
        w) width=${OPTARG};;
        h) height=${OPTARG};;
        d) filter_dates=${OPTARG};;
        o) save_dir=${OPTARG};;
        ?)
            echo "Usage: $(basename \$0) [-w width] [-h height] [-d filter_dates] [-o save_dir]"
            exit 1
            ;;
    esac
done

DIR="${save_dir}/${width}x${height}/${filter_dates}"

for mode in "greedy" "premade"; do
    echo "- Running simulation for mode: ${mode}"

    RESULTS_DIR="${DIR}/results/${mode}"
    mkdir -p $RESULTS_DIR

    poetry run python -m experiments.utils.forestcarbon.env_runner \
        --graph_file "${DIR}/env/obstacle_graph.pkl" \
        --graph_metadata_file "${DIR}/env/obstacle_graph_metadata.json" \
        --point_labels_file "${DIR}/env/point_labels.pkl" \
        --goal_dict_file "${DIR}/env/goal_dict.pkl" \
        --mode $mode \
        --video_output_file "${RESULTS_DIR}/video.mp4" \
        --collections_output_file "${RESULTS_DIR}/collections.pkl"
done
