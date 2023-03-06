#!/bin/bash
# NOTE: Should be executed from the root of the repository.
# This script generates the simulation environment for a given set of params.

while getopts w:h:l:d:s:t:m:o:b:c: flag
do
    case "${flag}" in
        w) width=${OPTARG};;
        h) height=${OPTARG};;
        l) locality=${OPTARG};;
        d) filter_dates=${OPTARG};;
        s) survey_details_file=${OPTARG};;
        t) tree_records_file=${OPTARG};;
        m) mask_file=${OPTARG};;
        o) save_dir=${OPTARG};;
        b) bbox_survey_details_file=${OPTARG};;
        c) bbox_tree_records_file=${OPTARG};;
        ?)
            echo "Usage: $(basename \$0) [-w width] [-h height] [-l locality] [-d filter_dates] [-s survey_details_file] [-t tree_records_file] [-m mask_file] [-o save_dir] [-b bbox_survey_details_file] [-c bbox_tree_records_file]"
            exit 1
            ;;
    esac
done

DIR="${save_dir}/${width}x${height}/${filter_dates}"

mkdir -p "${DIR}/env"

poetry run python -m scripts.graph.generate_graph \
    --input_file $mask_file \
    --output_file "${DIR}/env/obstacle_graph.pkl" \
    --metadata "${DIR}/env/obstacle_graph_metadata.json" \
    --resize $width $height

poetry run python -m experiments.utils.forestcarbon.generate_points_actions \
    --survey_details $survey_details_file \
    --tree_records $tree_records_file \
    --bbox_survey_details $bbox_survey_details_file \
    --bbox_tree_records $bbox_tree_records_file \
    --grid_width $width \
    --grid_height $height \
    --locality "${locality}" \
    --filter_dates "${filter_dates} ${filter_dates}" \
    --output_point_labels_file "${DIR}/env/point_labels.pkl" \
    --output_goal_dict_file "${DIR}/env/goal_dict.pkl"
