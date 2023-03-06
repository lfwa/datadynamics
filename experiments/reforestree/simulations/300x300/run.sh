#!/bin/bash
# NOTE: Should be executed from the root of the repository.
WIDTH=300
HEIGHT=300
LOCALITY="Carlos Vera Arteaga RGB"

FINAL_DATASET_FILE="experiments/reforestree/data/final_dataset.csv"
TRAIN_LIMIT=300
TEST_LIMIT=300
REPEATS=5
MASK_FILE="experiments/reforestree/simulations/obstacle_mask.png"
SAVE_DIR="experiments/reforestree/simulations"

VISUALIZATIONS_INPUT_DIR="${SAVE_DIR}/${WIDTH}x${HEIGHT}/results"
VISUALIZATIONS_OUTPUT_DIR="${VISUALIZATIONS_INPUT_DIR}/output-visualizations"
VISUALIZATIONS_OUTPUT_FILE="${VISUALIZATIONS_OUTPUT_DIR}/otdd.png"

echo "Generating environment..."
./experiments/utils/reforestree/generate_env.sh -w $WIDTH -h $HEIGHT -l "${LOCALITY}" -m "${MASK_FILE}" -o "${SAVE_DIR}" -f "${FINAL_DATASET_FILE}" -t $TRAIN_LIMIT -s $TEST_LIMIT

echo "Running simulations..."
./experiments/utils/reforestree/run_simulation.sh -w $WIDTH -h $HEIGHT -o "${SAVE_DIR}" -r $REPEATS

echo "Generating visualizations..."
mkdir -p "${VISUALIZATIONS_OUTPUT_DIR}"
./experiments/utils/reforestree/run_visualization.sh -i "${VISUALIZATIONS_INPUT_DIR}" -o "${VISUALIZATIONS_OUTPUT_FILE}" -r $REPEATS

echo "Generating screenshot..."
./experiments/utils/reforestree/run_screenshot.sh -w $WIDTH -h $HEIGHT -o "${SAVE_DIR}"
