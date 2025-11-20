import os
import gc
import time
import json
import re
import random
import copy
import traceback
import numpy as np
import torch
import torch.multiprocessing as mp
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import DataLoader
import logging
import yaml

MODEL_CONFIGS = {
    # "8b_shadowkv_chunk_size=8": "/home/test/test01/wpj/Megatron-LM/hf_ckpts/8b_full_sft_llama/900",
    # "1b_shadowkv_chunk_size=8": "/home/test/test01/wpj/Megatron-LM/hf_ckpts/1b_fullattn_sft",
    # "3b_shadowkv_chunk_size=8": "/home/test/test01/wpj/Megatron-LM/hf_ckpts/3b_fullattn_sft",
    # "8b_shadowkv_chunk_size=64": "/home/test/test01/wpj/Megatron-LM/hf_ckpts/8b_full_sft_llama/900",
    # "1b_shadowkv_chunk_size=64": "/home/test/test01/wpj/Megatron-LM/hf_ckpts/1b_fullattn_sft",
    # "3b_shadowkv_chunk_size=64": "/home/test/test01/wpj/Megatron-LM/hf_ckpts/3b_fullattn_sft",
    # "1b_infllmv1_128": "/home/test/test01/wpj/Megatron-LM/hf_ckpts/1b_fullattn_sft",
    # "3b_infllmv1_128": "/home/test/test01/wpj/Megatron-LM/hf_ckpts/3b_fullattn_sft",
    # "8b_infllmv1_128": "/home/test/test01/wpj/Megatron-LM/hf_ckpts/8b_full_sft_llama/900",
    # "1b_infllmv1_64": "/home/test/test01/wpj/Megatron-LM/hf_ckpts/1b_fullattn_sft",
    # "3b_infllmv1_64": "/home/test/test01/wpj/Megatron-LM/hf_ckpts/3b_fullattn_sft",
    # "8b_infllmv1_64": "/home/test/test01/wpj/Megatron-LM/hf_ckpts/8b_full_sft_llama/900",
    "1b_nosa_sft": "/home/test/test01/wpj/Megatron-LM/hf_ckpts/1b_nosa_sft",
    "3b_nosa_sft": "/home/test/test01/wpj/Megatron-LM/hf_ckpts/3b_nosa_normal_init_fix_select_sft",
    "8b_nosa_sft": "/home/test/test01/wpj/Megatron-LM/hf_ckpts/8b_nosa_sft_llama/900",
    "1b_infllmv2_sft": "/home/test/test01/wpj/Megatron-LM/hf_ckpts/1b_infllmv2_sft",
    "3b_infllmv2_sft": "/home/test/test01/wpj/Megatron-LM/hf_ckpts/3b_infllmv2_sft",
    "8b_infllmv2_sft": "/home/test/test01/wpj/Megatron-LM/hf_ckpts/8b_infllmv2_sft_llama/900",
    "1b_fullattn_sft": "/home/test/test01/wpj/Megatron-LM/hf_ckpts/1b_fullattn_sft",
    "3b_fullattn_sft": "/home/test/test01/wpj/Megatron-LM/hf_ckpts/3b_fullattn_sft",
    "8b_fullattn_sft": "/home/test/test01/wpj/Megatron-LM/hf_ckpts/8b_full_sft_llama/900",
}

DATASET_NAMES = ["cite_16k"] # + ["recall_16k", "rag_16k", "rerank_16k", "cite_16k", "longqa_16k", "summ_16k", "icl_16k"]

DATASET_CONFIG_DIR = "configs"
SEEDS = [0, 42, 3407]
AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]

from arguments import parse_arguments
from model_utils import load_LLM, OpenAIModel, AnthropicModel, TgiVllmModel
from data import load_data, TestItemDataset

logging.basicConfig(
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def run_single_test(args, model, dataset_name):
    """
    运行单个 (Model, Dataset, Seed) 的评测任务。
    返回: (output_path, metrics)
    """
    model.max_length = args.input_max_length
    model.generation_max_length = args.generation_max_length
    model_short_name = args.model_alias

    dataset = args.datasets
    test_file = args.test_files
    demo_file = args.demo_files
    tag = args.tag

    if dataset == "popqa":
        tag += f"_pop{args.popularity_threshold}"
    test_name = os.path.splitext(os.path.basename(test_file))[0]
    
    # 基础文件名
    base_filename = f"{model_short_name}/{dataset_name}/{dataset}_{tag}_{test_name}_in{args.input_max_length}_size{args.max_test_samples}_shots{args.shots}_samp{args.do_sample}max{args.generation_max_length}min{args.generation_min_length}t{args.temperature}p{args.top_p}_chat{args.use_chat_template}.json"
    # 实际输出文件名
    output_filename = f"{base_filename}_seed{args.seed}.json"
    output_path = os.path.join(args.output_dir, output_filename)

    # 检查是否已存在
    if os.path.exists(output_path) and not args.overwrite:
        logger.info(f"Skipping {output_path} (exists)")
        # 尝试读取现有的 metrics 用于聚合
        try:
            with open(output_path, "r") as f:
                data = json.load(f)
                return output_path, data.get("averaged_metrics", {}), base_filename
        except:
            logger.warning(f"Failed to read existing file {output_path}, rerunning...")

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 加载数据
    data = load_data(args, dataset, test_file, demo_file)
    
    # 构造 DataLoader
    dataloader = DataLoader(
        TestItemDataset(data, model, model.tokenizer),
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x,
        num_workers=0,
    )

    # 准备输入
    all_inputs = []
    all_input_texts = []
    metrics = defaultdict(list)

    for idx, inputs in enumerate(dataloader):
        inputs, input_text = inputs[0]
        if args.count_tokens:
            metrics['input_len'].append(inputs.input_ids.shape[1])
            continue
        all_inputs.append(inputs)
        all_input_texts.append(input_text)

    if args.count_tokens:
        # Token counting mode specific return
        return output_path, {}, base_filename

    # 处理 Thinking mode
    if args.thinking:
        # 注意：这里直接修改 model 实例属性是安全的，因为进程独占
        model.max_length = args.input_max_length + 32768
        model.generation_max_length = args.generation_max_length + 32768
        
    # 推理
    start_time = time.time()
    if (isinstance(model, OpenAIModel) or isinstance(model, AnthropicModel)) and (not isinstance(model, TgiVllmModel)):
        all_outputs = model.generate_batch(all_inputs, batch_file=output_path+".batch")
    else:
        all_outputs = model.generate_batch(all_inputs)
    end_time = time.time()

    # 后处理
    results = []
    for idx, output in enumerate(all_outputs):
        test_item = data["data"][idx]
        input_text = all_input_texts[idx]
        
        if output is None: continue

        if not args.use_chat_template:
            prepend_text = data["system_template"].format(**test_item)
            output["output"] = prepend_text + output["output"]
            
        if args.thinking:
            matches = re.search(r"(.*</think>)(.*)", output['output'], flags=re.DOTALL)
            if matches:
                output["output"] = matches.group(2).strip()
                output["thoughts"] = matches.group(1).strip()

        mets, others = data['post_process'](output, test_item)
        output.update({**others, **mets})
        
        for k, v in mets.items():
            metrics[k].append(v)
            
        metrics["input_len"].append(output["input_len"])
        metrics["output_len"].append(output["output_len"])
        
        result = {**test_item, **output}
        result.pop("context", None); result.pop("input_ids", None)
        if input_text is None: input_text = result['input_text']
        results.append(result)

    # 计算平均指标
    averaged_metrics = {k: np.mean(v)*(100 if "_len" not in k else 1) for k, v in metrics.items()}

    # 构造输出对象
    output_data = {
        "args": args.__dict__,
        "data": results,
        "metrics": metrics,
        "averaged_metrics": averaged_metrics,
        "throughput": len(results) / (end_time - start_time),
    }

    # 写入文件 (JSON)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)
    
    # 写入文件 (.score) - 单次 Run 的分数
    if "alce" not in dataset:
        with open(output_path + ".score", "w") as f:
            json.dump(averaged_metrics, f, indent=4)

    return output_path, averaged_metrics, base_filename


def gpu_worker_process(gpu_id, task_queue, result_queue, base_args):
    """工作进程：独占一个 GPU, 不断从队列获取任务。"""
    # 1. 设置环境变量，锁定该进程只能看到分配的 GPU
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 设置进程名，方便调试
    current_process = mp.current_process()
    current_process.name = f"Worker-GPU{gpu_id}"
    
    logger.info(f"Worker started on physical GPU {gpu_id}")

    current_model_key = None
    model = None

    while True:
        try:
            # 获取任务 (timeout 防止死锁，虽然 None 也是结束信号)
            task = task_queue.get(timeout=5) 
        except:
            break # 队列空了或超时，退出

        if task is None:
            break # 收到结束信号

        (
            model_key,
            model_path,
            dataset,
            test_file,
            demo_file,
            seed,
            max_length,
            gen_length,
            dataset_name,
            use_chat_template,
            max_test_samples,
            shots,
            stop_new_line,
        ) = task

        logger.info(f"Received Task: Model={model_key}, Dataset={dataset}, Seed={seed}")

        # 2. 检查是否需要切换模型
        if model_key != current_model_key:
            if model is not None:
                logger.info(f"Unloading previous model {current_model_key}...")
                del model
                gc.collect()
                torch.cuda.empty_cache()
            
            logger.info(f"Loading new model: {model_key} from {model_path}")
            current_model_key = model_key
            
            # 构造加载模型的 args
            load_args = copy.deepcopy(base_args)
            load_args.model_key = model_key
            load_args.model_name_or_path = model_path

            model = load_LLM(load_args, _device="cuda:0")

        # 3. 构造运行时的 args
        task_args = copy.deepcopy(base_args)
        task_args.model_name_or_path = model_path
        task_args.model_alias = model_key

        task_args.datasets = dataset
        task_args.test_files = test_file
        task_args.demo_files = demo_file
        task_args.input_max_length = max_length
        task_args.generation_max_length = gen_length
        task_args.use_chat_template = use_chat_template
        task_args.max_test_samples = max_test_samples
        task_args.shots = shots
        task_args.stop_new_line = stop_new_line

        task_args.seed = seed
        task_args.tag = getattr(base_args, "tag", "")
        
        try:
            # 4. 执行评测
            out_path, metrics, base_fn = run_single_test(task_args, model, dataset_name)
            
            # 4.1 如果是 ALCE 相关数据集，用 eval_alce 生成/更新 .score，并用其中的结果覆盖 metrics
            if ("alce" in dataset) and (not base_args.count_tokens):
                try:
                    # 只有在 .score 不存在或指定 overwrite 时才重新跑 ALCE 评测
                    score_path = out_path + ".score"
                    if (not os.path.exists(score_path)) or getattr(base_args, "overwrite", False):
                        import eval_alce
                        logger.info(f"Running eval_alce on {out_path} ...")
                        cli_args = ["--f", out_path]
                        # 与单机脚本一致：非 nocite 数据集开启 citation 评测
                        if "nocite" not in dataset:
                            cli_args.append("--citations")
                        eval_alce.main(cli_args)

                    # 读取 eval_alce 生成的 .score 作为该任务的 metrics
                    with open(score_path, "r") as f:
                        alce_metrics = json.load(f)
                    metrics = alce_metrics
                except Exception as ee:
                    logger.error(f"eval_alce failed for {out_path}: {ee}")
                    # 如果 eval_alce 失败，就退回到 run_single_test 的 averaged_metrics
            
            # 5. 发送结果回主进程
            result_queue.put({
                "status": "success",
                "model": model_key,
                "dataset": dataset,
                "seed": seed,
                "metrics": metrics,
                "base_filename": base_fn
            })
        except Exception as e:
            logger.error(f"Task failed: {model_key} on {dataset} seed {seed}")
            traceback.print_exc()
            result_queue.put({
                "status": "error",
                "msg": str(e)
            })

    logger.info("Worker finished.")


def main():
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 准备任务列表
    tasks = []
    
    # 遍历配置区定义的模型
    for model_key, model_path in MODEL_CONFIGS.items():
        os.makedirs(f"{args.output_dir}/{model_key}", exist_ok=True)

        # 遍历配置区定义的数据集名字
        for dataset_name in DATASET_NAMES:
            os.makedirs(f"{args.output_dir}/{model_key}/{dataset_name}", exist_ok=True)

            yaml_path = os.path.join(DATASET_CONFIG_DIR, f"{dataset_name}.yaml")
            
            if not os.path.exists(yaml_path):
                logger.warning(f"Config file not found for {dataset_name}: {yaml_path}, skipping...")
                continue

            with open(yaml_path, 'r', encoding='utf-8') as f:
                ds_config = yaml.safe_load(f)
            
            datasets = ds_config["datasets"].split(",")
            test_files = ds_config["test_files"].split(",")
            demo_files = ds_config["demo_files"].split(",")

            max_lengths = ([int(ds_config["input_max_length"])] * len(datasets)) if isinstance(ds_config["input_max_length"], int) or len(ds_config["input_max_length"].split(",")) == 1 else [int(l) for l in ds_config["input_max_length"].split(",")]
            gen_lengths = ([int(ds_config["generation_max_length"])] * len(datasets)) if isinstance(ds_config["generation_max_length"], int) or len(ds_config["generation_max_length"].split(",")) == 1 else [int(l) for l in ds_config["generation_max_length"].split(",")]

            use_chat_template = ds_config["use_chat_template"]
            max_test_samples = ds_config["max_test_samples"]
            shots = ds_config["shots"]
            stop_new_line = ds_config["stop_new_line"]
            
            # 生成该 (模型, 数据集) 下的所有种子任务
            for seed in SEEDS:
                tasks.extend([
                    (
                        model_key,
                        model_path,
                        dataset,
                        test_file,
                        demo_file,
                        seed,
                        int(max_length),
                        int(gen_length),
                        dataset_name,
                        use_chat_template,
                        max_test_samples,
                        shots,
                        stop_new_line,
                    )
                    for dataset, test_file, demo_file, max_length, gen_length
                    in zip(datasets, test_files, demo_files, max_lengths, gen_lengths)
                ])
    
    logger.info(f"Total tasks generated: {len(tasks)}")
    print(tasks)

    # 创建队列
    task_queue = mp.Queue()
    result_queue = mp.Queue()

    # 填充任务队列
    for task in tasks:
        task_queue.put(task)
    
    # 放入停止信号 (每个 Worker 一个)
    gpu_list = AVAILABLE_GPUS if AVAILABLE_GPUS else list(range(torch.cuda.device_count()))
    num_workers = len(gpu_list)
    for _ in range(num_workers):
        task_queue.put(None)

    logger.info(f"Launching {num_workers} workers on GPUs: {gpu_list}")

    # 启动 Worker
    processes = []
    for gpu_id in gpu_list:
        p = mp.Process(target=gpu_worker_process, args=(gpu_id, task_queue, result_queue, args))
        p.start()
        processes.append(p)

    # 聚合结果容器
    # key: (model_key, dataset_name) -> list of metrics
    aggregator = defaultdict(list)
    # key: (model_key, dataset_name) -> base_filename (for stat file naming)
    filename_map = {}

    # 监听结果
    # 我们知道总共有 len(tasks) 个结果需要接收
    total_tasks = len(tasks)
    completed_tasks = 0
    
    pbar = tqdm(total=total_tasks, desc="Pipeline Progress")

    while completed_tasks < total_tasks:
        # 阻塞获取结果
        res = result_queue.get()
        
        if res["status"] == "success":
            m_key = res["model"]
            d_set = res["dataset"]
            metrics = res["metrics"]
            base_fn = res["base_filename"]
            
            key = (m_key, d_set)
            aggregator[key].append(metrics)
            filename_map[key] = base_fn
            
            # 检查是否该组合的所有 Seed 都跑完了
            if len(aggregator[key]) == len(SEEDS):
                # 计算并输出 .stat 文件
                metrics_list = aggregator[key]
                stat_output = {}
                
                # 假设 metrics_list 是 list of dicts: [{'acc': 10}, {'acc': 20}, {'acc': 30}]
                # 拿到所有的 metric keys
                all_keys = metrics_list[0].keys()
                
                for metric_k in all_keys:
                    values = [m.get(metric_k, 0) for m in metrics_list]
                    stat_output[metric_k] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "raw_values": values
                    }
                
                # 输出 .stat 文件
                # 文件名规则：base_filename (不含seed) + .score.stat
                stat_path = os.path.join(args.output_dir, f"{base_fn}.score.stat")
                with open(stat_path, "w") as f:
                    json.dump(stat_output, f, indent=4)
                
                logger.info(f"Aggregated stats saved to {stat_path}")
        
        elif res["status"] == "error":
            logger.error(f"Worker reported error: {res['msg']}")
            
        completed_tasks += 1
        pbar.update(1)

    pbar.close()
    
    # 等待进程结束
    for p in processes:
        p.join()

    logger.info("All evaluations finished.")

if __name__ == "__main__":
    main()