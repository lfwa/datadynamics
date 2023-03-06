#!/bin/bash
# NOTE: Should be executed from the root of the repository.
# This script generates the visualizations for the simulation results.

while getopts i:o:r: flag
do
    case "${flag}" in
        i) input_dir=${OPTARG};;
        o) output_file=${OPTARG};;
        r) repeats=${OPTARG};;
        ?)
            echo "Usage: $(basename \$0) [-i input_dir] [-o output_file] [-r repeats]"
            exit 1
            ;;
    esac
done

poetry run python -m experiments.utils.reforestree.visualize \
    --input_dir "${input_dir}" \
    --output_file "${output_file}" \
    --repeats $repeats
