#!/bin/bash
# run.sh — Robot Brain launcher
# Usage:
#   bash run.sh baseline    ← 論文實驗 Baseline
#   bash run.sh corruption  ← 論文實驗 Corruption
#   bash run.sh demo        ← Demo（直接用 baseline DB）
#   bash run.sh dev         ← 開發測試
#   bash run.sh reset       ← 清空某個 DB
#   bash run.sh results     ← 看實驗結果 + 畫圖

cd "$(dirname "$0")"

MODE=${1:-dev}

_load_charades() {
    DB=$1
    python3 -c "
import os; os.environ['DB_NAME']='$DB'
from pymongo import MongoClient
n = MongoClient()['$DB'].transition_matrix.count_documents({})
if n == 0:
    print('[run] Loading Charades prior...')
    import subprocess, sys
    subprocess.run([sys.executable,'tools/charades_pipeline.py'])
else:
    print(f'[run] Charades ready ({n} records)')
" 2>/dev/null
}

case $MODE in

  baseline)
    echo "[run] Baseline → robot_exp_baseline"
    _load_charades robot_exp_baseline
    DB_NAME=robot_exp_baseline python3 app.py
    ;;

  corruption)
    echo "[run] Corruption → robot_exp_corruption"
    echo "[run] Unity: pickupMissRate=0.15, putdownMissRate=0.10"
    _load_charades robot_exp_corruption
    DB_NAME=robot_exp_corruption python3 app.py
    ;;

  demo)
    BASELINE_N=$(python3 -c "
from pymongo import MongoClient
print(MongoClient()['robot_exp_baseline'].transition_counts.count_documents({}))
" 2>/dev/null || echo "0")
    if [ "$BASELINE_N" -gt "5" ] 2>/dev/null; then
      echo "[run] Demo using baseline data ($BASELINE_N transitions)"
      echo "[run] eval_logs and transition_counts are safe — demo only writes conversation_logs"
      DB_NAME=robot_exp_baseline python3 app.py
    else
      echo "[run] No baseline data — using synthetic demo data"
      DB_NAME=robot_demo python3 tools/seed_demo.py
      DB_NAME=robot_demo python3 app.py
    fi
    ;;

  dev)
    echo "[run] Dev → robot_rag_db"
    DB_NAME=robot_rag_db python3 app.py
    ;;

  reset)
    TARGET=${2:-""}
    case $TARGET in
      baseline)   RESET_DB=robot_exp_baseline ;;
      corruption) RESET_DB=robot_exp_corruption ;;
      demo)       RESET_DB=robot_demo ;;
      dev)        RESET_DB=robot_rag_db ;;
      *)
        echo "Usage: bash run.sh reset [baseline|corruption|demo|dev]"
        echo ""
        echo "  bash run.sh reset baseline    ← clear Baseline DB"
        echo "  bash run.sh reset corruption  ← clear Corruption DB"
        echo "  bash run.sh reset demo        ← clear Demo DB"
        exit 1
        ;;
    esac
    echo "[run] Resetting $RESET_DB..."
    DB_NAME=$RESET_DB python3 tools/resetall.py --keep-charades
    echo "[run] Done. Next: bash run.sh $TARGET"
    ;;

  results)
    echo "Which analysis?"
    echo "  1) Exp 1: HAR accuracy (baseline)"
    echo "  2) Exp 2: Corruption comparison (baseline vs corruption)"
    echo "  3) Exp 3: Habit learning (baseline)"
    echo "  4) All three"
    read -p "Choice [1-4]: " choice
    case $choice in
      1)
        DB_NAME=robot_exp_baseline python3 analysis/exp1_har_accuracy.py
        ;;
      2)
        python3 analysis/exp2_corruption.py
        ;;
      3)
        DB_NAME=robot_exp_baseline python3 analysis/exp3_habit_learning.py
        ;;
      4)
        echo "=== Exp 1: HAR accuracy ==="
        DB_NAME=robot_exp_baseline python3 analysis/exp1_har_accuracy.py
        echo "=== Exp 2: Corruption ==="
        python3 analysis/exp2_corruption.py
        echo "=== Exp 3: Habit learning ==="
        DB_NAME=robot_exp_baseline python3 analysis/exp3_habit_learning.py
        echo "Results saved to analysis/results/"
        ;;
      *) echo "Invalid"; exit 1 ;;
    esac
    ;;

  watch)
    echo "Watch which DB?"
    echo "  1) robot_exp_baseline"
    echo "  2) robot_exp_corruption"
    echo "  3) robot_rag_db"
    read -p "Choice [1-3]: " choice
    case $choice in
      1) WATCH_DB=robot_exp_baseline ;;
      2) WATCH_DB=robot_exp_corruption ;;
      3) WATCH_DB=robot_rag_db ;;
      *) echo "Invalid"; exit 1 ;;
    esac
    DB_NAME=$WATCH_DB python3 tools/check_system.py --watch
    ;;

  check)
    DB=${2:-robot_rag_db}
    DB_NAME=$DB python3 tools/check_system.py --quick
    ;;

  *)
    echo "Usage: bash run.sh [baseline|corruption|demo|dev|reset|results|watch|check]"
    exit 1
    ;;
esac