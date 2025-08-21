#!/bin/bash

# format metrics:
# Total execution time of 2 iterations of sedov up to t = 0.000002: 0.315663s
in=$1
out="bencher_beverin_mi300_prgenv-gnu-25.07-6.3.3-v4.json"
# commit=`head -n1 $in |awk '{print $3}'`
sec_per_iter=`grep 'Total execution time' $in |awk '{print $15/$6}' |tr -d s`
echo "{\"sphexa-hip sedov\": {\"sec/iter\": {\"value\": $sec_per_iter }}}" > $out


# install bencher:
wget --quiet https://github.com/bencherdev/bencher/releases/download/v0.5.4/bencher-v0.5.4-linux-x86-64
ln -fs bencher-v0.5.4-linux-x86-64 bencher
chmod +x bencher
./bencher --version


# send data to bencher:
testbed="${out#*_}"        # remove ^bencher
testbed="${testbed%.json}" # remove .json
echo "testbed=$testbed"

./bencher run \
    --threshold-measure sec/iter \
    --threshold-test percentage \
    --threshold-lower-boundary _ \
    --threshold-upper-boundary 0.1 \
    --threshold-max-sample-size 64 \
    \
    --adapter json \
    --file bencher=$out \
    --testbed $testbed \
    --thresholds-reset \
    --branch develop \
    \
    --token $BENCHER_API_TOKEN \
    --project $BENCHER_PROJECT
