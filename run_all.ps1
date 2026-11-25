# Windows PowerShell용 전체 파이프라인 (실제 실험에 사용한 설정)
# 실행:  powershell -ExecutionPolicy Bypass -File .\run_all.ps1
# 오차 분석 태깅(사람 작업)이 필요해서 Track A는 extract 후 멈춘다 → 이후는 run_all_after_tagging 참고.
$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
function Run { python @args; if ($LASTEXITCODE -ne 0) { throw "실패: python $args" } }

# ---------------- Track B (시계열): 작은 모델이라 CPU 권장
$env:DEVICE = "cpu"
Run -m src.track_b.train --ticker AAPL --start 2015-01-01
Run -m src.track_b.hparam_search --csv data/AAPL.csv --arch GRU
Run -m src.track_b.train --csv data/AAPL.csv --config outputs/track_b/hparam/best_config.json --out outputs/track_b_tuned --reports reports/tuned
Remove-Item Env:DEVICE

# ---------------- Track A (전이 학습)
Run -m src.track_a.prepare_data --source cifar10 --classes cat dog deer horse bird
Run -m src.track_a.train --strategy all --img_size 128 --epochs 15 --workers 0
Run -m src.track_a.error_analysis extract --run outputs/track_a/partial

Write-Host ">>> reports/error_analysis_table.csv 의 tag 열을 직접 채운 뒤 아래를 순서대로 실행하세요:"
Write-Host "python -m src.track_a.error_analysis summarize --run outputs/track_a/partial"
Write-Host "python -m src.track_a.train --strategy partial --img_size 224 --tag res224_only --epochs 15 --workers 0 --batch_size 16"
Write-Host "python -m src.track_a.train --strategy partial --img_size 224 --label_smoothing 0.1 --tag res224 --epochs 15 --workers 0 --batch_size 16"
Write-Host "python -m src.track_a.train --arch efficientnet_b0 --strategy partial --img_size 224 --tag effb0_224 --epochs 15 --workers 0 --batch_size 16"
Write-Host "python -m src.track_a.train --strategy partial --aug strong --dropout 0.5 --weight_decay 1e-3 --tag aug_strong --img_size 128 --epochs 15 --workers 0 --batch_size 16"
Write-Host "python -m src.track_a.error_analysis compare --before outputs/track_a/partial --after outputs/track_a/partial_<tag>   # 처방마다 1회"
