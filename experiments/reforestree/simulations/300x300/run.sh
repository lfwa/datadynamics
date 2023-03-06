#!/bin/bash
# NOTE: Should be executed from the root of the repository.
WIDTH=300
HEIGHT=300
LOCALITY="Carlos Vera Arteaga RGB"

FINAL_DATASET_FILE="experiments/reforestree/data/final_dataset.csv"
TRAIN_LIMIT=300
TEST_LIMIT=300
MASK_FILE="experiments/reforestree/simulations/obstacle_mask.png"
SAVE_DIR="experiments/reforestree/simulations"

echo "Generating environment..."
./experiments/utils/reforestree/generate_env.sh -w $WIDTH -h $HEIGHT -l "${LOCALITY}" -m "${MASK_FILE}" -o "${SAVE_DIR}" -f "${FINAL_DATASET_FILE}" -t $TRAIN_LIMIT -s $TEST_LIMIT

echo "Running simulations..."
./experiments/utils/reforestree/run_simulation.sh -w $WIDTH -h $HEIGHT -o "${SAVE_DIR}"

echo "Generating screenshot..."
./experiments/utils/reforestree/run_screenshot.sh -w $WIDTH -h $HEIGHT -o "${SAVE_DIR}"
