#!/bin/bash
# NOTE: Should be executed from the root of the repository.
# This script runs a greedy and premade simulation for a given environment.

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

for reward_mode in "uniform" "knn" "lava"; do
    for cheating_cost_mode in "none" "mid" "high"; do
        echo "- Running simulation for reward_mode: ${reward_mode}, cheating_cost_mode: ${cheating_cost_mode}"

        RESULTS_DIR="${DIR}/results/greedy/${reward_mode}/${cheating_cost_mode}"
        mkdir -p $RESULTS_DIR

        poetry run python -m experiments.utils.reforestree.env_runner \
            --graph_file "${DIR}/env/obstacle_graph.pkl" \
            --graph_metadata_file "${DIR}/env/obstacle_graph_metadata.json" \
            --point_labels_file "${DIR}/env/point_labels.pkl" \
            --mode "greedy" \
            --video_output_file "${RESULTS_DIR}/video.mp4" \
            --collections_output_file "${RESULTS_DIR}/collections.pkl" \
            --cheating_cost_map_file "${DIR}/env/cheat_maps/${reward_mode}/${cheating_cost_mode}.pkl" \
            --collection_reward_map_file "${DIR}/env/reward_maps/${reward_mode}.pkl"
    done
done
