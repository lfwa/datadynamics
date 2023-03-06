#!/bin/bash
# NOTE: Should be executed from the root of the repository.
# This script generates the visualizations for the simulation results.

while getopts i:o:d:r:c:f: flag
do
    case "${flag}" in
        i) input_dir=${OPTARG};;
        o) output_file=${OPTARG};;
        d) filter_dates=${OPTARG};;
        r) nrows=${OPTARG};;
        c) ncols=${OPTARG};;
        f) figsize=${OPTARG};;
        ?)
            echo "Usage: $(basename \$0) [-i input_dir] [-o output_file] [-d filter_dates] [-r nrows] [-c ncols] [-f figsize_width figsize_height]"
            exit 1
            ;;
    esac
done

poetry run python -m experiments.utils.forestcarbon.visualize \
    --input_dir "${input_dir}" \
    --output_file "${output_file}" \
    --filter_dates "${filter_dates}" \
    --nrows $nrows \
    --ncols $ncols \
    --figsize $figsize
