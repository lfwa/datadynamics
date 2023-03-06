#!/bin/bash
# NOTE: Should be executed from the root of the repository.
# This script runs a greedy and premade simulation for a given environment.

while getopts w:h:o:r: flag
do
    case "${flag}" in
        w) width=${OPTARG};;
        h) height=${OPTARG};;
        o) save_dir=${OPTARG};;
        r) repeats=${OPTARG};;
        ?)
            echo "Usage: $(basename \$0) [-w width] [-h height] [-o save_dir] [-r repeats]"
            exit 1
            ;;
    esac
done

DIR="${save_dir}/${width}x${height}"

for mode in "val_dataset" "col_dataset"; do
    for reward_mode in "uniform" "knn" "lava"; do
        for cheating_cost_mode in "none" "mid" "high"; do
            for repeat in $(seq 1 $repeats); do
                echo "- Running simulation ${repeat} for mode: ${mode}, reward_mode: ${reward_mode}, cheating_cost_mode: ${cheating_cost_mode}"

                RESULTS_DIR="${DIR}/results/${mode}/greedy/${reward_mode}/${cheating_cost_mode}/${repeat}"
                mkdir -p $RESULTS_DIR

                ENV_DIR="${DIR}/env/${mode}"

                poetry run python -m experiments.utils.reforestree.env_runner \
                    --graph_file "${ENV_DIR}/obstacle_graph.pkl" \
                    --graph_metadata_file "${ENV_DIR}/obstacle_graph_metadata.json" \
                    --point_labels_file "${ENV_DIR}/point_labels.pkl" \
                    --mode "greedy" \
                    --video_output_file "${RESULTS_DIR}/video.mp4" \
                    --collections_output_file "${RESULTS_DIR}/collections.pkl" \
                    --cheating_cost_map_file "${ENV_DIR}/cheat_maps/${reward_mode}/${cheating_cost_mode}.pkl" \
                    --collection_reward_map_file "${ENV_DIR}/reward_maps/${reward_mode}.pkl" \
                    --seed $repeat
            done
        done
    done
done
