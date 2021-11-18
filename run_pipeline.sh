#!/bin/bash

# Make sure the script aborts if any of the intermediate steps fail
set -euo pipefail  # https://vaneyckt.io/posts/safer_bash_scripts_with_set_euxo_pipefail/

# Setup command line flags to hydra
if [[ $1 == "minimal" ]]; then
  echo "Running pipeline in minimal mode";
  eval_args="dr_config@decision_referral=minimal"
  plot_args="operating_pairs=[NT@0.97+SN@0.98] n_bootstrap=10"
elif [[ $1 == "maximal" ]]; then
  echo "Running pipeline in maximal mode"
  eval_args=""
  plot_args=""
else
  echo "Please provide either 'minimal' or 'maximal'; got: $1; aborting."
  exit
fi

# Make sure we have the docker image
DOCKER_IMAGE=vara_dr
if ! [[ "$(docker inspect --type=image $DOCKER_IMAGE --format='available')" == "available" ]]; then
  echo "Docker image not found locally, downloading it"
  curl --output $DOCKER_IMAGE.tar.gz https://storage.googleapis.com/mx-healthcare-pub/$DOCKER_IMAGE.tar.gz
  echo "Unzipping image"
  docker image load -i $DOCKER_IMAGE.tar.gz
fi

# Make sure input data is available
for filename in internal_validation_set.h5 internal_test_set.h5 external_test_set.h5; do
  if ! [[ -e data/inputs/$filename ]]; then
      echo "$filename not found, downloading it"
      curl --output data/inputs/$filename https://storage.googleapis.com/mx-healthcare-pub/$filename
  fi
done

echo -n "Delete existing artifacts if necessary .. "
rm -rf data/results
echo "✓"

# Run the pipeline: evaluation and plotting for all datasets
for dataset in INTERNAL_TEST_SET EXTERNAL_TEST_SET; do
  echo -n "Evaluate decision referral on $dataset.. "
  docker run -it --rm -v "$(pwd):/vara_dr" $DOCKER_IMAGE \
    python evaluate.py test_dataset=$dataset $eval_args
  echo "✓"

  echo -n "Generate plots for $dataset.. "
  docker run -it --rm -v "$(pwd):/vara_dr" $DOCKER_IMAGE \
    python generate_plots.py test_dataset=$dataset $plot_args
  echo "✓"
done
