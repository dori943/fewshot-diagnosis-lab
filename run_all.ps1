# Windows PowerShell용 전체 파이프라인.  실행:  powershell -ExecutionPolicy Bypass -File .\run_all.ps1
# 오차 분석 태깅(사람 작업)이 필요해서 중간에 멈춘다.
$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
function Run { python @args; if ($LASTEXITCODE -ne 0) { throw "실패: python $args" } }

# ---------------- Track B (시계열)
Run -m src.track_b.train --ticker AAPL --start 2015-01-01
Run -m src.track_b.hparam_search --csv data/AAPL.csv --arch GRU
Run -m src.track_b.train --csv data/AAPL.csv --config outputs/track_b/hparam/best_config.json --out outputs/track_b_tuned --reports reports/tuned

# ---------------- Track A (전이 학습)
Run -m src.track_a.prepare_data --source cifar10 --classes cat dog deer horse bird
Run -m src.track_a.train --strategy all
Run -m src.track_a.error_analysis extract --run outputs/track_a/full

Write-Host ">>> reports/error_analysis_table.csv 의 tag 열을 직접 채운 뒤 아래를 실행하세요:"
Write-Host "python -m src.track_a.error_analysis summarize --run outputs/track_a/full"
Write-Host "python -m src.track_a.train --strategy full --aug strong --tag aug_strong"
Write-Host "python -m src.track_a.error_analysis compare --before outputs/track_a/full --after outputs/track_a/full_aug_strong"
