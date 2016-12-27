#!/bin/bash

source activate lsst
source eups-setups.sh
setup lsst_distrib

source setup.bash
jupyter notebook