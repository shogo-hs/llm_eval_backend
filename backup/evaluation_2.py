from typing import List, Dict, Any, Optional
import asyncio
from pathlib import Path
import json
import litellm
from litellm import acompletion
import logging
import pandas as pd
from tqdm import tqdm
import sys
import datetime

# 設定のインポート
from app.config import get_settings

# 評価メトリクスのインポート
from app.metrics.char_f1 import CharF1
from app.metrics.exact_match import ExactMatch

# 設定の取得
settings = get_settings()

# 結果ディレクトリの作成
settings.initialize_dirs()

# ロギングの設定
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# 評価メトリクスの関数マッピング
METRICS_FUNC_MAP = {
    "char_f1": CharF1().calculate,
    "exact_match": ExactMatch().calculate,
}

async def get_few_shot_samples(dataset_name: str, n_shots: int) -> List[Dict[str, str]]:
    """
    Few-shotサンプルを取得する関数

    Args:
        dataset_name: データセット名
        n_shots: サンプル数

    Returns:
        Few-shotサンプルのリスト
    """
    if n_shots == 0:
        return []

    dataset_path = settings.TRAIN_DIR / f"{dataset_name}.json"

    with dataset_path.open(encoding="utf-8") as f:
        train_data = json.load(f)

    few_shots = []
    for i in range(min(n_shots, len(train_data["samples"]))):
        sample = train_data["samples"][i]
        few_shots.append({"role": "user", "content": sample["input"]})
        few_shots.append({"role": "assistant", "content": sample["output"]})

    return few_shots

async def format_prompt(instruction: str, input_text: str, few_shots: List[Dict[str, str]] = None) -> List[Dict[str, str]]:
    """
    プロンプトをフォーマットする関数

    Args:
        instruction: タスクの指示
        input_text: 入力テキスト
        few_shots: Few-shotサンプル

    Returns:
        メッセージのリスト
    """
    messages = []

    # システムメッセージの作成
    is_english = "mmlu_en" in instruction
    if is_english:
        message_intro = "The following text provides instructions for a certain task."
    else:
        message_intro = "以下に、あるタスクを説明する指示があり、それに付随する入力が更なる文脈を提供しています。リクエストを適切に完了するための回答を記述してください。"

    system_message = f"{message_intro}\n\n{instruction}"
    messages.append({"role": "system", "content": system_message})

    # Few-shotサンプルの追加
    if few_shots:
        messages.extend(few_shots)

    # ユーザー入力の追加
    messages.append({"role": "user", "content": input_text})

    return messages

async def call_model_with_litellm(messages: List[Dict[str, str]],
                                model_name: str,
                                provider_name: str,
                                max_tokens: int = settings.DEFAULT_MAX_TOKENS,
                                temperature: float = settings.DEFAULT_TEMPERATURE) -> str:
    """
    LiteLLMを使用してモデルを呼び出す関数

    Args:
        messages: メッセージのリスト
        model_name: モデル名
        provider_name: プロバイダ名
        max_tokens: 最大トークン数
        temperature: 温度

    Returns:
        モデルの出力テキスト
    """
    # プロバイダとモデル名を結合（LiteLLMの形式に合わせる）
    full_model_name = f"{provider_name}/{model_name}"

    try:
        response = await acompletion(
            model=full_model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            base_url=settings.LITELLM_BASE_URL
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error calling model {full_model_name}: {e}")
        return ""

async def process_batch(batch: List[Dict], model_name: str, provider_name: str,
                       instruction: str, output_length: int, n_shots: int, dataset_name: str):
    """
    バッチ処理を行う関数

    Args:
        batch: サンプルのバッチ
        model_name: モデル名
        provider_name: プロバイダ名
        instruction: タスクの指示
        output_length: 出力の最大長
        n_shots: Few-shotサンプル数
        dataset_name: データセット名

    Returns:
        処理結果のリスト
    """
    few_shots = await get_few_shot_samples(dataset_name, n_shots)
    results = []

    for sample in batch:
        messages = await format_prompt(instruction, sample["input"], few_shots)
        raw_output = await call_model_with_litellm(
            messages=messages,
            model_name=model_name,
            provider_name=provider_name,
            max_tokens=output_length
        )

        # 出力の前処理（テキスト整形など）
        processed_output = raw_output.strip()
        # 必要に応じて、さらに処理を追加

        results.append({
            "input": sample["input"],
            "expected_output": sample["output"],
            "raw_output": raw_output,
            "processed_output": processed_output,
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages]
        })

    return results

async def run_evaluation(
    dataset_name: str,
    provider_name: str,
    model_name: str,
    num_samples: int,
    n_shots: List[int]
) -> Dict[str, Any]:
    """
    評価の実行プロセス

    Args:
        dataset_name: 評価対象のデータセット名
        provider_name: モデルのプロバイダ名
        model_name: モデル名
        num_samples: 評価するサンプル数
        n_shots: Few-shotサンプル数のリスト

    Returns:
        評価結果を含む辞書
    """
    dataset_path = settings.DATASET_DIR / f"{dataset_name}.json"
    batch_size = settings.BATCH_SIZE

    with dataset_path.open(encoding="utf-8") as f:
        dataset = json.load(f)

    metrics = dataset["metrics"]
    instruction = dataset["instruction"]
    output_length = dataset["output_length"]
    samples = dataset["samples"][:num_samples]

    all_results = {}

    for shot in n_shots:
        shot_results = []

        for i in range(0, len(samples), batch_size):
            batch = samples[i:i+batch_size]
            batch_results = await process_batch(
                batch=batch,
                model_name=model_name,
                provider_name=provider_name,
                instruction=instruction,
                output_length=output_length,
                n_shots=shot,
                dataset_name=dataset_name
            )
            shot_results.extend(batch_results)

        for metric_name in metrics:
            if metric_name in METRICS_FUNC_MAP:
                metric_func = METRICS_FUNC_MAP[metric_name]
                scores = [
                    metric_func(result["processed_output"], result["expected_output"])
                    for result in shot_results
                ]
                avg_score = sum(scores) / len(scores) if scores else 0
                all_results[f"{dataset_name}_{shot}shot_{metric_name}"] = avg_score

        all_results[f"{dataset_name}_{shot}shot_details"] = shot_results

    summary = []
    for shot in n_shots:
        row = {
            "dataset": dataset_name,
            "model": f"{provider_name}/{model_name}",
            "n_shots": shot,
            "num_samples": len(samples)
        }
        # all_results のキーから実際に測定された指標だけ追加
        prefix = f"{dataset_name}_{shot}shot_"
        for key, value in all_results.items():
            if key.startswith(prefix) and not key.endswith("_details"):
                metric_name = key[len(prefix):]
                row[metric_name] = value
        summary.append(row)

    return {
        "summary": summary,       # DataFrame ではなく List[Dict]
        "details": all_results,
        "metadata": {
            "dataset": dataset_name,
            "model": f"{provider_name}/{model_name}",
            "num_samples": num_samples,
            "n_shots": n_shots,
            "timestamp": datetime.datetime.now().isoformat()
        }
    }

async def run_multiple_evaluations(
    datasets: List[str],
    provider_name: str,
    model_name: str,
    num_samples: int,
    n_shots: List[int]
) -> Dict[str, Any]:
    results = {}
    all_summary: List[Dict[str, Any]] = []
    timestamp = datetime.datetime.now().isoformat()

    for dataset_name in datasets:
        dataset_results = await run_evaluation(
            dataset_name, provider_name, model_name, num_samples, n_shots
        )
        results[dataset_name] = dataset_results
        # 辞書リストをそのままマージ
        all_summary.extend(dataset_results["summary"])

    return {
        "results": results,
        "summary": all_summary,   # List[Dict]
        "metadata": {
            "provider_name": provider_name,
            "model_name": model_name,
            "datasets": datasets,
            "num_samples": num_samples,
            "n_shots": n_shots,
            "timestamp": timestamp
        }
    }

def save_results_as_json(results: Dict[str, Any], provider_name: str, model_name: str) -> Path:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{provider_name}_{model_name}_{timestamp}.json"
    file_path = settings.RESULTS_DIR / filename

    # pandas 不要。results["summary"] は List[Dict] のまま
    json_results = {
        "summary": results["summary"],
        "metadata": results["metadata"],
        "datasets": {}
    }
    for ds, ds_res in results["results"].items():
        json_results["datasets"][ds] = {
            "metadata": ds_res["metadata"],
            "details": ds_res["details"]
        }

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved to {file_path}")
    return file_path


async def main():
    """
    メイン関数
    """
    # 評価設定 - 設定からデフォルト値を使用
    provider_name = "ollama"
    model_name = "phi4:latest"
    datasets = ["aio", "janli"]  # 評価するデータセット
    num_samples = settings.DEFAULT_NUM_SAMPLES  # 評価するサンプル数
    n_shots = settings.DEFAULT_N_SHOTS  # Few-shotサンプル数

    logger.info(f"Starting evaluation: {provider_name}/{model_name}")
    logger.info(f"Datasets: {datasets}")
    logger.info(f"Number of samples: {num_samples}")
    logger.info(f"Number of shots: {n_shots}")

    # 評価の実行
    results = await run_multiple_evaluations(
        datasets=datasets,
        provider_name=provider_name,
        model_name=model_name,
        num_samples=num_samples,
        n_shots=n_shots
    )

    # 結果の保存
    results_file = save_results_as_json(results, provider_name, model_name)
    logger.info(f"Evaluation completed. Results saved to {results_file}")

if __name__ == "__main__":
    asyncio.run(main())
