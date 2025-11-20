export HF_ENDPOINT=https://hf-mirror.com

model="8b_shadowkv"
path="/home/test/test01/hyx/Megatron-LM/hf_checkpoints/8b_full_sft_llama/900"

for task in summ_16k; do
# for task in cite_16k; do
  mkdir -p output/${model}
  mkdir -p output/${model}/${task}
  python eval.py --config configs/${task}.yaml \
    --model_name_or_path ${path} \
    --output_dir output/${model}/${task} \
    # --max_test_samples 50 \
    # --use_chat_template False # only if you are using non-instruction-tuned models, otherwise use the default.
done
