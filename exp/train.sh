while getopts "c:" flag
do
  case "${flag}" in
    c) cuda_idx=${OPTARG} ;;
  esac
done

if [ -n "$cuda_idx" ]; then
  export CUDA_VISIBLE_DEVICES=$cuda_idx
fi

random_hash=$(echo $RANDOM | md5sum | head -c 20)

python ../tools/train.py model.py | tee logs/otxmodel_voc_full_${random_hash}.log