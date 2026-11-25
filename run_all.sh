#!/usr/bin/env bash
# macOS/Linux용 전체 파이프라인 (실제 실험에 사용한 설정). 태깅 단계에서 멈춘다.
set -e
DEVICE=cpu python -m src.track_b.train --ticker AAPL --start 2015-01-01
DEVICE=cpu python -m src.track_b.hparam_search --csv data/AAPL.csv --arch GRU
DEVICE=cpu python -m src.track_b.train --csv data/AAPL.csv --config outputs/track_b/hparam/best_config.json \
       --out outputs/track_b_tuned --reports reports/tuned

python -m src.track_a.prepare_data --source cifar10 --classes cat dog deer horse bird
python -m src.track_a.train --strategy all --img_size 128 --epochs 15
python -m src.track_a.error_analysis extract --run outputs/track_a/partial
echo ">>> reports/error_analysis_table.csv 의 tag 열을 채운 뒤 README '실행' 절의 나머지 명령을 실행하세요."
