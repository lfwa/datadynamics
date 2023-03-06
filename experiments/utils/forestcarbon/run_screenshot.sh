#!/bin/bash
# NOTE: Should be executed from the root of the repository.
# This script generates a screenshot of the environment with all points.

while getopts w:h:l:s:t:m:o:b:c: flag
do
    case "${flag}" in
        w) width=${OPTARG};;
        h) height=${OPTARG};;
        l) locality=${OPTARG};;
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

filter_dates="all"
DIR="${save_dir}/${width}x${height}/${filter_dates}"
RESULTS_DIR="${DIR}/results"

mkdir -p "${RESULTS_DIR}"

./experiments/utils/forestcarbon/generate_env.sh -w $width -h $height -l "${locality}" -d "${filter_dates}" -s "${survey_details_file}" -t "${tree_records_file}" -m "${mask_file}" -o "${save_dir}" -b "${bbox_survey_details_file}" -c "${bbox_tree_records_file}"

poetry run python -m experiments.utils.forestcarbon.env_runner \
        --graph_file "${DIR}/env/obstacle_graph.pkl" \
        --graph_metadata_file "${DIR}/env/obstacle_graph_metadata.json" \
        --point_labels_file "${DIR}/env/point_labels.pkl" \
        --goal_dict_file "${DIR}/env/goal_dict.pkl" \
        --mode "screenshot" \
        --screenshot_output_file "${RESULTS_DIR}/screenshot.png"
