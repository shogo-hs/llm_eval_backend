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
import time
import random
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# 設定のインポート
from app.config import get_settings

# 評価メトリクスのインポート（動的読み込みを使用）
from app.metrics import get_metrics_functions

# モデル管理コントローラーのインポート
from app.utils.model_management import get_model_controller

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

# モデル管理コントローラーのインスタンス取得
model_controller = get_model_controller()

# 評価メトリクスの関数マッピングを動的に取得
METRICS_FUNC_MAP = get_metrics_functions()

# LiteLLM呼び出しの例外定義
class LiteLLMAPIError(Exception):
    """LiteLLM API呼び出し中のエラー"""
    pass

class LiteLLMTimeoutError(Exception):
    """LiteLLM API呼び出しのタイムアウトエラー"""
    pass

class LiteLLMRateLimitError(Exception):
    """LiteLLM APIのレート制限エラー"""
    pass

class ModelNotAvailableError(Exception):
    """モデルが利用できないエラー"""
    pass

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

# リトライポリシーを使用したモデル呼び出し関数
@retry(
    retry=retry_if_exception_type((LiteLLMTimeoutError, LiteLLMRateLimitError)), 
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
async def call_model_with_retry(
    messages: List[Dict[str, str]],
    model_name: str,
    provider_name: str,
    max_tokens: int,
    temperature: float,
    additional_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    リトライロジックを含むモデル呼び出し関数

    Args:
        messages: メッセージのリスト
        model_name: モデル名
        provider_name: プロバイダ名
        max_tokens: 最大トークン数
        temperature: 温度
        additional_params: 追加パラメータ

    Returns:
        LiteLLMのレスポンス
    """
    # モデルが利用可能かどうかをチェック
    if settings.MODEL_CHECK_BEFORE_CALL:
        available, message = await model_controller.ensure_model_availability(provider_name, model_name)
        if not available:
            logger.error(f"Model {model_name} is not available: {message}")
            raise ModelNotAvailableError(message)
    
    # プロバイダとモデル名を結合（LiteLLMの形式に合わせる）
    full_model_name = f"{provider_name}/{model_name}"
    
    # リクエストパラメータの設定
    request_params = {
        "model": full_model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "base_url": settings.LITELLM_BASE_URL,
    }
    
    # 追加パラメータの適用
    if additional_params:
        request_params.update(additional_params)
    
    try:
        # カスタムタイムアウト設定でAPIを呼び出し
        response = await asyncio.wait_for(
            acompletion(**request_params),
            timeout=settings.MODEL_TIMEOUT
        )
        return response
    except asyncio.TimeoutError:
        logger.warning(f"Timeout calling model {full_model_name}")
        raise LiteLLMTimeoutError(f"Timeout calling model {full_model_name}")
    except Exception as e:
        error_message = str(e).lower()
        if "rate limit" in error_message or "too many requests" in error_message:
            logger.warning(f"Rate limit error calling model {full_model_name}: {e}")
            # レート制限エラーは再試行
            raise LiteLLMRateLimitError(f"Rate limit error: {str(e)}")
        else:
            # その他のエラーは記録して再発生
            logger.error(f"Error calling model {full_model_name}: {e}")
            raise LiteLLMAPIError(f"API error: {str(e)}")

async def try_fallback_providers(
    messages: List[Dict[str, str]],
    primary_provider: str,
    primary_model: str,
    max_tokens: int,
    temperature: float,
    additional_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    フォールバックプロバイダーを使用して処理を試みる関数
    
    Args:
        messages: メッセージのリスト
        primary_provider: 最初に使用するプロバイダー
        primary_model: 最初に使用するモデル
        max_tokens: 最大トークン数
        temperature: 温度
        additional_params: 追加パラメータ
        
    Returns:
        (レスポンス, 使用したプロバイダー, 使用したモデル)のタプル
    """
    # 最初のプロバイダーを試す
    try:
        response = await call_model_with_retry(
            messages=messages,
            model_name=primary_model,
            provider_name=primary_provider,
            max_tokens=max_tokens,
            temperature=temperature,
            additional_params=additional_params
        )
        return response, primary_provider, primary_model
    except (LiteLLMAPIError, LiteLLMTimeoutError, LiteLLMRateLimitError, ModelNotAvailableError) as e:
        logger.warning(f"Primary provider {primary_provider} failed: {e}")
        
    # フォールバックプロバイダーが設定されていない場合はそのままエラーを発生
    if not settings.FALLBACK_PROVIDERS:
        raise LiteLLMAPIError(f"Primary provider {primary_provider} failed and no fallback providers configured")
    
    # フォールバックプロバイダーを順番に試す
    for fallback_provider in settings.FALLBACK_PROVIDERS:
        # 同じプロバイダーはスキップ
        if fallback_provider == primary_provider:
            continue
            
        logger.info(f"Trying fallback provider: {fallback_provider}")
        
        # フォールバック用のモデル名を決定（実装によって最適なモデルを選択できる）
        fallback_model = primary_model
        if fallback_provider != primary_provider:
            # ここで必要に応じてプロバイダーに適したモデルに変更
            # 例: OpenAIのgpt-4-turboに対応するAnthropicのモデルを選ぶなど
            # この実装例では単純化のため同じモデル名を使用
            pass
        
        try:
            response = await call_model_with_retry(
                messages=messages,
                model_name=fallback_model,
                provider_name=fallback_provider,
                max_tokens=max_tokens,
                temperature=temperature,
                additional_params=additional_params
            )
            return response, fallback_provider, fallback_model
        except (LiteLLMAPIError, LiteLLMTimeoutError, LiteLLMRateLimitError, ModelNotAvailableError) as e:
            logger.warning(f"Fallback provider {fallback_provider} failed: {e}")
    
    # すべてのフォールバックが失敗
    raise LiteLLMAPIError("All providers failed")

async def call_model_with_litellm(
    messages: List[Dict[str, str]],
    model_name: str,
    provider_name: str,
    max_tokens: int = settings.DEFAULT_MAX_TOKENS,
    temperature: float = settings.DEFAULT_TEMPERATURE,
    additional_params: Optional[Dict[str, Any]] = None,
    use_fallback: bool = True
) -> Dict[str, Any]:
    """
    LiteLLMを使用してモデルを呼び出す関数

    Args:
        messages: メッセージのリスト
        model_name: モデル名
        provider_name: プロバイダ名
        max_tokens: 最大トークン数
        temperature: 温度
        additional_params: 追加パラメータ辞書
        use_fallback: フォールバックプロバイダーを使用するかどうか

    Returns:
        (モデルの出力テキスト、使用したプロバイダー、使用したモデル)の辞書
    """
    try:
        if use_fallback and settings.FALLBACK_PROVIDERS:
            # フォールバックロジックを使用
            response, used_provider, used_model = await try_fallback_providers(
                messages=messages,
                primary_provider=provider_name,
                primary_model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                additional_params=additional_params
            )
        else:
            # リトライロジックを含む関数を呼び出し（フォールバックなし）
            response = await call_model_with_retry(
                messages=messages,
                model_name=model_name,
                provider_name=provider_name,
                max_tokens=max_tokens,
                temperature=temperature,
                additional_params=additional_params
            )
            used_provider = provider_name
            used_model = model_name
            
        return {
            "content": response.choices[0].message.content,
            "provider": used_provider,
            "model": used_model
        }
            
    except (LiteLLMAPIError, LiteLLMTimeoutError, LiteLLMRateLimitError, ModelNotAvailableError) as e:
        logger.error(f"Failed after retries: {e}")
        return {
            "content": f"ERROR: {str(e)}",
            "provider": provider_name,
            "model": model_name
        }
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {
            "content": f"ERROR: Unexpected error occurred: {str(e)}",
            "provider": provider_name,
            "model": model_name
        }

async def process_batch(batch: List[Dict], model_name: str, provider_name: str,
                       instruction: str, output_length: int, n_shots: int, dataset_name: str,
                       additional_params: Optional[Dict[str, Any]] = None):
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
        additional_params: 追加パラメータ辞書

    Returns:
        処理結果のリスト
    """
    few_shots = await get_few_shot_samples(dataset_name, n_shots)
    results = []

    for sample in batch:
        messages = await format_prompt(instruction, sample["input"], few_shots)
        response = await call_model_with_litellm(
            messages=messages,
            model_name=model_name,
            provider_name=provider_name,
            max_tokens=output_length,
            additional_params=additional_params
        )

        # 出力の前処理（テキスト整形など）
        raw_output = response["content"]
        processed_output = raw_output.strip()
        # 必要に応じて、さらに処理を追加

        results.append({
            "input": sample["input"],
            "expected_output": sample["output"],
            "raw_output": raw_output,
            "processed_output": processed_output,
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
            "provider": response["provider"],  # 使用したプロバイダー
            "model": response["model"]         # 使用したモデル
        })

    return results

async def run_evaluation(
    dataset_name: str,
    provider_name: str,
    model_name: str,
    num_samples: int,
    n_shots: List[int],
    additional_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    評価の実行プロセス

    Args:
        dataset_name: 評価対象のデータセット名
        provider_name: モデルのプロバイダ名
        model_name: モデル名
        num_samples: 評価するサンプル数
        n_shots: Few-shotサンプル数のリスト
        additional_params: 追加パラメータ辞書

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
                dataset_name=dataset_name,
                additional_params=additional_params
            )
            shot_results.extend(batch_results)

        # プロバイダー別にカウント
        providers_used = {}
        for result in shot_results:
            used_provider = result.get("provider", provider_name)
            providers_used[used_provider] = providers_used.get(used_provider, 0) + 1
            
        # プロバイダー使用状況のログ
        for provider, count in providers_used.items():
            percentage = (count / len(shot_results)) * 100
            logger.info(f"Provider {provider} used for {count}/{len(shot_results)} samples ({percentage:.2f}%)")

        error_count = sum(1 for result in shot_results if result["processed_output"].startswith("ERROR:"))
        if error_count > 0:
            logger.warning(f"{error_count} out of {len(shot_results)} samples failed with errors")

        for metric_name in metrics:
            if metric_name in METRICS_FUNC_MAP:
                metric_func = METRICS_FUNC_MAP[metric_name]
                scores = [
                    metric_func(result["processed_output"], result["expected_output"])
                    for result in shot_results if not result["processed_output"].startswith("ERROR:")
                ]
                if scores:  # エラーを除いたスコアがある場合のみ平均を計算
                    avg_score = sum(scores) / len(scores)
                    all_results[f"{dataset_name}_{shot}shot_{metric_name}"] = avg_score
                    # エラー率も記録
                    all_results[f"{dataset_name}_{shot}shot_{metric_name}_error_rate"] = error_count / len(shot_results)
                else:
                    all_results[f"{dataset_name}_{shot}shot_{metric_name}"] = 0
                    all_results[f"{dataset_name}_{shot}shot_{metric_name}_error_rate"] = 1.0
            else:
                logger.warning(f"Metric '{metric_name}' specified in dataset but not found in registry")

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
            if key.startswith(prefix) and not key.endswith("_details") and not key.endswith("_error_rate"):
                metric_name = key[len(prefix):]
                row[metric_name] = value
                # エラー率も追加
                error_rate_key = f"{prefix}{metric_name}_error_rate"
                if error_rate_key in all_results:
                    row[f"{metric_name}_error_rate"] = all_results[error_rate_key]
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
    n_shots: List[int],
    additional_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    複数データセットに対する評価を実行する関数

    Args:
        datasets: 評価するデータセットのリスト
        provider_name: プロバイダ名
        model_name: モデル名
        num_samples: 評価するサンプル数
        n_shots: Few-shotサンプル数のリスト
        additional_params: 追加パラメータ辞書

    Returns:
        評価結果の辞書
    """
    results = {}
    all_summary: List[Dict[str, Any]] = []
    timestamp = datetime.datetime.now().isoformat()

    for dataset_name in datasets:
        dataset_results = await run_evaluation(
            dataset_name, provider_name, model_name, num_samples, n_shots, additional_params
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
            "timestamp": timestamp,
            "additional_params": additional_params
        }
    }

def save_results_as_json(results: Dict[str, Any], provider_name: str, model_name: str) -> Path:
    """
    評価結果をJSONファイルとして保存する関数

    Args:
        results: 評価結果辞書
        provider_name: プロバイダ名
        model_name: モデル名

    Returns:
        保存したファイルのパス
    """
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
    provider_name = settings.DEFAULT_PROVIDER
    model_name = "phi4:latest"
    datasets = ["aio", "janli"]  # 評価するデータセット
    num_samples = settings.DEFAULT_NUM_SAMPLES  # 評価するサンプル数
    n_shots = settings.DEFAULT_N_SHOTS  # Few-shotサンプル数
    
    # プロバイダーとモデルが利用可能か確認
    available, message = await model_controller.ensure_model_availability(provider_name, model_name)
    if not available:
        logger.error(f"Model {model_name} is not available: {message}")
        if settings.AUTO_DOWNLOAD_MODELS and provider_name == "ollama":
            logger.info(f"Attempting to download model {model_name}")
            success, dl_message = await model_controller.pull_ollama_model(model_name)
            if not success:
                logger.error(f"Failed to download model {model_name}: {dl_message}")
                sys.exit(1)
            logger.info(f"Model {model_name} downloaded successfully")
        else:
            logger.error("Model is not available and auto-download is disabled")
            sys.exit(1)
    
    # 追加パラメータ（例：カスタムヘッダー）
    additional_params = {
        "headers": {"User-Agent": "LLM-Evaluation-Tool/1.0"}
    }

    # 利用可能なメトリクスを表示
    logger.info(f"Available metrics: {list(METRICS_FUNC_MAP.keys())}")

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
        n_shots=n_shots,
        additional_params=additional_params
    )

    # 結果の保存
    results_file = save_results_as_json(results, provider_name, model_name)
    logger.info(f"Evaluation completed. Results saved to {results_file}")

if __name__ == "__main__":
    asyncio.run(main())
