#!/usr/bin/env bash
# 전체 파이프라인. 오차 분석 태깅(2단계)은 사람이 직접 해야 하므로 스크립트가 중간에 멈춘다.
set -e
# ---------------- Track B (시계열) : 베이스라인 → 딥러닝 → 진단/비교 (train.py 내부에서 순서 보장)
python -m src.track_b.train --ticker AAPL --start 2015-01-01
python -m src.track_b.hparam_search --csv data/AAPL.csv --arch GRU        # 보너스
python -m src.track_b.train --csv data/AAPL.csv --config outputs/track_b/hparam/best_config.json \
       --out outputs/track_b_tuned --reports reports/tuned

# ---------------- Track A (전이 학습)
python -m src.track_a.prepare_data --source cifar10 --classes cat dog deer horse bird
python -m src.track_a.train --strategy all
python -m src.track_a.error_analysis extract --run outputs/track_a/full
echo ">>> reports/error_analysis_table.csv 의 tag 열을 직접 채운 뒤 아래를 실행하세요:"
echo "python -m src.track_a.error_analysis summarize --run outputs/track_a/full"
echo "python -m src.track_a.train --strategy full --aug strong --tag aug_strong"
echo "python -m src.track_a.error_analysis compare --before outputs/track_a/full --after outputs/track_a/full_aug_strong"
