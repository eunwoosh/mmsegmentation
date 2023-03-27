while getopts "c:m" flag
do
  case "${flag}" in
    c) cuda_idx=${OPTARG} ;;
    m) mixed_precision="true"
  esac
done

if [ -n "$cuda_idx" ]; then
  export CUDA_VISIBLE_DEVICES=$cuda_idx
fi

random_hash=$(echo $RANDOM | md5sum | head -c 20)


start_time=$(date)

if [ -z "$mixed_precision" ]; then
  PYTHONBREAKPOINT=pudb.forked.set_trace \
  python ../tools/train.py model.py --work-dir logs/test_seg_voc_full_fullp_${random_hash}
else
  python ../tools/train.py model_fp16.py --work-dir logs/test_seg_voc_full_mixedp_${random_hash}
fi

  # python ../tools/train.py model.py --work-dir logs/no_custom_seg_voc_full_fullp_${random_hash}
  # python ../tools/train.py model_fp16.py --work-dir logs/no_custom_seg_voc_full_mixedp_${random_hash}

echo "start time : ${start_time}"
echo "end time   : $(date)"
