# "${1}" is the first argument passed to the script
# "${2}" is the second argument passed to the script
if [ ! -f intent_best.pt ]; then
  bash download.sh
fi
python3 test_intent.py --test_file "${1}" --load_ckpt_path intent_best.pt --pred_file "${2}"