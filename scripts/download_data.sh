#!/usr/bin/env bash
# Downloads UCI Default of Credit Card Clients (CC BY 4.0) into data/
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
[ -s data/UCI_Credit_Card.csv ] || curl -fsSL https://huggingface.co/datasets/scikit-learn/credit-card-clients/resolve/main/UCI_Credit_Card.csv -o data/UCI_Credit_Card.csv
echo done
