#!/bin/bash
# NOTE: Should be executed from the root of the repository.
WIDTH=355
HEIGHT=242
LOCALITY="Lagadlarin, Lobo, Batangas"
declare -a filter_dates=(
    "2022-10-06"
    "2022-10-07"
    "2022-10-08"
    "2022-10-09"
    "2022-10-10"
    "2022-10-11"
    "2022-10-12"
    "2022-10-14"
    )

SURVEY_DETAILS_FILE="experiments/lobo/data/survey_details.csv"
TREE_RECORDS_FILE="experiments/lobo/data/tree_records.csv"
BBOX_SURVEY_DETAILS_FILE="experiments/lobo/data/survey_details.csv"
BBOX_TREE_RECORDS_FILE="experiments/lobo/data/tree_records.csv"
MASK_FILE="experiments/lobo/simulations/custom_mask/obstacle_mask.png"
SAVE_DIR="experiments/lobo/simulations/custom_mask"

VISUALIZATIONS_INPUT_DIR="${SAVE_DIR}/${WIDTH}x${HEIGHT}"
VISUALIZATIONS_OUTPUT_DIR="${VISUALIZATIONS_INPUT_DIR}/output-visualizations"
VISUALIZATIONS_OUTPUT_FILE="${VISUALIZATIONS_OUTPUT_DIR}/otdd.png"
VISUALIZATIONS_NROWS=2
VISUALIZATIONS_NCOLS=4
VISUALIZATIONS_FIGSIZE="12 5"

for date in "${filter_dates[@]}"; do
    echo "Generating environment for date: ${date}"

    ./experiments/utils/forestcarbon/generate_env.sh -w $WIDTH -h $HEIGHT -l "${LOCALITY}" -d "${date}" -s "${SURVEY_DETAILS_FILE}" -t "${TREE_RECORDS_FILE}" -m "${MASK_FILE}" -o "${SAVE_DIR}" -b "${BBOX_SURVEY_DETAILS_FILE}" -c "${BBOX_TREE_RECORDS_FILE}"

    echo "Running simulation for date: ${date}"
    ./experiments/utils/forestcarbon/run_simulation.sh -w $WIDTH -h $HEIGHT -d "${date}" -o "${SAVE_DIR}"
done

echo "Generating visualizations..."
mkdir -p "${VISUALIZATIONS_OUTPUT_DIR}"
./experiments/utils/forestcarbon/run_visualization.sh -i "${VISUALIZATIONS_INPUT_DIR}" -o "${VISUALIZATIONS_OUTPUT_FILE}" -d "${filter_dates[*]}" -r $VISUALIZATIONS_NROWS -c $VISUALIZATIONS_NCOLS -f "${VISUALIZATIONS_FIGSIZE}"

echo "Generating screenshot..."
./experiments/utils/forestcarbon/run_screenshot.sh -w $WIDTH -h $HEIGHT -l "${LOCALITY}" -s "${SURVEY_DETAILS_FILE}" -t "${TREE_RECORDS_FILE}" -m "${MASK_FILE}" -o "${SAVE_DIR}" -b "${BBOX_SURVEY_DETAILS_FILE}" -c "${BBOX_TREE_RECORDS_FILE}"
