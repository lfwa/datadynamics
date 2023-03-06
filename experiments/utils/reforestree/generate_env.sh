#!/bin/bash
# NOTE: Should be executed from the root of the repository.
# This script generates the simulation environment for a given set of params.

while getopts w:h:l:m:o:f:t:s: flag
do
    case "${flag}" in
        w) width=${OPTARG};;
        h) height=${OPTARG};;
        l) locality=${OPTARG};;
        m) mask_file=${OPTARG};;
        o) save_dir=${OPTARG};;
        f) final_dataset_file=${OPTARG};;
        t) train_limit=${OPTARG};;
        s) test_limit=${OPTARG};;
        ?)
            echo "Usage: $0 [-w width] [-h height] [-l locality] [-m mask_file] [-o save_dir] [-f final_dataset_file] [-t train_limit] [-s test_limit]"
            exit 1
            ;;
    esac
done

DIR="${save_dir}/${width}x${height}"

for mode in "val_dataset" "col_dataset"; do
    ENV_DIR="${DIR}/env/${mode}"
    mkdir -p $ENV_DIR

    poetry run python -m scripts.graph.generate_graph \
        --input_file $mask_file \
        --output_file "${ENV_DIR}/obstacle_graph.pkl" \
        --metadata "${ENV_DIR}/obstacle_graph_metadata.json" \
        --resize $width $height

    poetry run python -m experiments.utils.reforestree.generate_points_data \
        --final_dataset_file $final_dataset_file \
        --grid_width $width \
        --grid_height $height \
        --locality "${locality}" \
        --train_limit $train_limit \
        --test_limit $test_limit \
        --output_point_labels_file "${ENV_DIR}/point_labels.pkl" \
        --output_data_file "${ENV_DIR}/data.pkl" \
        --mode $mode

    mkdir -p "${ENV_DIR}/reward_maps"
    for reward_mode in "uniform" "knn" "lava"; do
        mkdir -p "${ENV_DIR}/cheat_maps/${reward_mode}"
    done

    poetry run python -m experiments.utils.reforestree.generate_reward_cheat_maps \
        --point_labels_file "${ENV_DIR}/point_labels.pkl" \
        --data_file "${ENV_DIR}/data.pkl" \
        --save_dir "${ENV_DIR}"
done
