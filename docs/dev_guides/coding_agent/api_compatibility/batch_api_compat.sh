#!/bin/bash

# Usage: ./batch_api_compat.sh api1 api2 api3 ...
# Or pipe from file: ./batch_api_compat.sh $(cat api_list.txt)

set -e

APIS=("$@")
BATCH_SIZE=8
TOTAL=${#APIS[@]}

if [ "$TOTAL" -eq 0 ]; then
    echo "Usage: $0 <api1> <api2> ... <apiN>"
    echo "       $0 \$(cat api_list.txt)"
    exit 1
fi

echo "Total APIs: $TOTAL, batch size: $BATCH_SIZE"
echo "================================"

for (( i=0; i<TOTAL; i+=BATCH_SIZE )); do
    batch=("${APIS[@]:$i:$BATCH_SIZE}")
    batch_str="${batch[*]}"
    batch_num=$(( i / BATCH_SIZE + 1 ))
    total_batches=$(( (TOTAL + BATCH_SIZE - 1) / BATCH_SIZE ))

    echo ""
    echo ">>> Batch $batch_num / $total_batches: $batch_str"
    ducc -p "/api-compatibility $batch_str"
done

echo ""
echo "================================"
echo ">>> All batches done. Creating PR..."
ducc -p "/create-pr"

echo ""
echo "Done."
