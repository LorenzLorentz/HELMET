export HF_ENDPOINT=https://hf-mirror.com

model="1b_infllmv2_sft_test_single_gpu"
path="/home/test/test01/wpj/Megatron-LM/hf_ckpts/1b_infllmv2_sft"

for task in recall_16k rag_16k rerank_16k cite_16k longqa_16k summ_16k icl_16k; do
# for task in cite_16k; do
  mkdir -p output/${model}
  mkdir -p output/${model}/${task}
  python eval.py --config configs/${task}.yaml \
    --model_name_or_path ${path} \
    --output_dir output/${model}/${task} \
    # --max_test_samples 50 \
    # --use_chat_template False # only if you are using non-instruction-tuned models, otherwise use the default.
done
