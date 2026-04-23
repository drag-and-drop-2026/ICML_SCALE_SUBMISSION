import asyncio
import base64
import io
import json
import os
from argparse import ArgumentParser
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from PIL import Image
from tqdm.asyncio import tqdm as tqdm_asyncio

load_dotenv()

COORD_MAX = 1000
SCALES = list(range(1, 20))

FORMAT = {
    "properties": {
        "action": {
            "const": "drag_and_drop",
            "default": "drag_and_drop",
            "title": "Action",
            "type": "string",
        },
        "x1": {
            "description": "The x coordinate of the start of the drag, normalized between 0 and 1000",
            "maximum": 1000,
            "minimum": 0,
            "title": "X1",
            "type": "integer",
        },
        "y1": {
            "description": "The y coordinate of the start of the drag, normalized between 0 and 1000",
            "maximum": 1000,
            "minimum": 0,
            "title": "Y1",
            "type": "integer",
        },
        "x2": {
            "description": "The x coordinate of the end of the drag, normalized between 0 and 1000",
            "maximum": 1000,
            "minimum": 0,
            "title": "X2",
            "type": "integer",
        },
        "y2": {
            "description": "The y coordinate of the end of the drag, normalized between 0 and 1000",
            "maximum": 1000,
            "minimum": 0,
            "title": "Y2",
            "type": "integer",
        },
    },
    "required": ["x1", "y1", "x2", "y2"],
    "title": "DragAndDropAction",
    "type": "object",
}

_FORMAT_JSON = json.dumps(FORMAT)
PROMPT = (
    "Localize the beginning and end of the vector on the GUI image according to the task and output the coordinates of the beginning and end of the vector. "
    "You must output a valid JSON following the format: {format_json} "
    "Coordinates must be between 0 and {coord_max}. "
    "Your drag and drop task is: {task}"
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("sample"))
    parser.add_argument("--model-id", default="qwen3-5-397b-a17b-fp8")
    parser.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY"))
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument(
        "--reasoning-effort",
        default="minimal",
        choices=["minimal", "low", "medium", "high"],
    )
    return parser.parse_args()


def image_to_data_url(path: Path) -> str:
    buf = io.BytesIO()
    Image.open(path).convert("RGB").save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def point_in_bbox(x: float, y: float, bbox: list[float]) -> bool:
    """bbox is [x_min, y_min, x_max, y_max] in normalized [0, 1] coordinates."""
    x_min, y_min, x_max, y_max = bbox
    return x_min <= x <= x_max and y_min <= y <= y_max


def scale_bbox(bbox: list[float], scale: float) -> list[float]:
    """Scale bbox around its center by `scale` (linear), clipped to [0, 1]."""
    x_min, y_min, x_max, y_max = bbox
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    half_w = (x_max - x_min) / 2 * scale
    half_h = (y_max - y_min) / 2 * scale
    return [
        max(0.0, cx - half_w),
        max(0.0, cy - half_h),
        min(1.0, cx + half_w),
        min(1.0, cy + half_h),
    ]


def extract_coords(pred: dict | None) -> tuple[float, float, float, float] | None:
    """Return normalized (x1, y1, x2, y2) in [0, 1], or None if pred is malformed."""
    if not isinstance(pred, dict):
        return None
    keys = ("x1", "y1", "x2", "y2")
    if not all(isinstance(pred.get(k), (int, float)) for k in keys):
        return None
    x1, y1, x2, y2 = (pred[k] / COORD_MAX for k in keys)
    return x1, y1, x2, y2


def evaluate_pred(pred: dict | None, sample: dict) -> tuple[bool, dict[int, bool]]:
    """Return (start_ok, end_ok_by_scale) for a single prediction."""
    coords = extract_coords(pred)
    if coords is None:
        return False, {s: False for s in SCALES}
    x1, y1, x2, y2 = coords
    start_ok = point_in_bbox(x1, y1, sample["start_bbox"])
    end_by_scale = {s: point_in_bbox(x2, y2, scale_bbox(sample["end_bbox"], s)) for s in SCALES}
    return start_ok, end_by_scale


def iter_samples(data_dir: Path):
    for img_path in sorted(data_dir.glob("*.jpg")):
        json_path = img_path.with_suffix(".json")
        if not json_path.exists():
            continue
        with json_path.open() as f:
            for i, sample in enumerate(json.load(f)):
                yield img_path, i, sample


async def predict(client: AsyncOpenAI, args, image_url: str, task: str) -> dict:
    enable_thinking = args.reasoning_effort != "minimal"
    extra_kwargs = {"reasoning_effort": args.reasoning_effort} if enable_thinking else {}
    response = await client.chat.completions.create(
        model=args.model_id,
        messages=[
            {"role": "system", "content": "You are a GUI grounding assistant."},
            {
                "role": "user",
                "content": PROMPT.format(format_json=_FORMAT_JSON, coord_max=COORD_MAX, task=task),
            },
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": image_url}}],
            },
        ],
        response_format={"type": "json_object"},
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
        **extra_kwargs,
    )
    return json.loads(response.choices[0].message.content)


async def safe_predict(
    client: AsyncOpenAI, args, image_url: str, task: str
) -> tuple[dict | None, str | None]:
    """Run `predict`, returning (None, error_repr) instead of raising.

    Needed so a single failed API call does not abort the whole batch.
    """
    try:
        return await predict(client, args, image_url, task), None
    except Exception as e:
        return None, repr(e)


def build_result(
    img_path: Path,
    index: int,
    sample: dict,
    pred: dict | None,
    start_ok: bool,
    end_by_scale: dict[int, bool],
    error: str | None,
) -> dict:
    by_scale = {s: start_ok and end_by_scale[s] for s in SCALES}
    return {
        "image": img_path.name,
        "index": index,
        "intent": sample["intent"],
        "start_bbox": sample["start_bbox"],
        "end_bbox": sample["end_bbox"],
        "prediction": pred,
        "correct": by_scale[1],
        "correct_start": start_ok,
        "correct_end_by_scale": {str(s): end_by_scale[s] for s in SCALES},
        "correct_by_scale": {str(s): by_scale[s] for s in SCALES},
        "error": error,
    }


async def run_one(
    sem: asyncio.Semaphore,
    client: AsyncOpenAI,
    args,
    img_path: Path,
    image_url: str,
    index: int,
    sample: dict,
) -> dict:
    async with sem:
        pred, error = await safe_predict(client, args, image_url, sample["intent"])
    start_ok, end_by_scale = evaluate_pred(pred, sample)
    result = build_result(img_path, index, sample, pred, start_ok, end_by_scale, error)
    if args.verbose:
        status = "OK" if result["correct"] else "FAIL"
        intent = sample["intent"]
        tqdm_asyncio.write(
            f"[{status}] {img_path.name}#{index}: {intent[:60]!r} -> {pred} err={error}"
        )
    return result


def summarize(results: list[dict], args) -> dict:
    total = len(results)
    safe = total or 1
    correct_start = sum(int(r["correct_start"]) for r in results)
    correct_end = {s: sum(int(r["correct_end_by_scale"][str(s)]) for r in results) for s in SCALES}
    correct = {s: sum(int(r["correct_by_scale"][str(s)]) for r in results) for s in SCALES}
    return {
        "accuracy": correct[1] / safe,
        "correct": correct[1],
        "total": total,
        "start_accuracy": correct_start / safe,
        "correct_start": correct_start,
        "end_accuracy_by_scale": {f"end@{s}x": correct_end[s] / safe for s in SCALES},
        "correct_end_by_scale": {f"end@{s}x": correct_end[s] for s in SCALES},
        "accuracy_by_scale": {f"pass@{s}x": correct[s] / safe for s in SCALES},
        "correct_by_scale": {f"pass@{s}x": correct[s] for s in SCALES},
        "model_id": args.model_id,
        "reasoning_effort": args.reasoning_effort,
        "concurrency": args.concurrency,
        "results": results,
    }


def build_tasks(args, client: AsyncOpenAI, sem: asyncio.Semaphore) -> list:
    image_urls: dict[Path, str] = {}
    tasks = []
    for img_path, i, sample in iter_samples(args.data_dir):
        if img_path not in image_urls:
            image_urls[img_path] = image_to_data_url(img_path)
        tasks.append(run_one(sem, client, args, img_path, image_urls[img_path], i, sample))
    return tasks


async def gather_results(tasks: list) -> list[dict]:
    results: list[dict] = []
    correct_start = 0
    correct_end_1x = 0
    correct_pass_1x = 0
    pbar = tqdm_asyncio(asyncio.as_completed(tasks), total=len(tasks), desc="eval", unit="sample")
    async for coro in pbar:
        r = await coro
        results.append(r)
        correct_start += int(r["correct_start"])
        correct_end_1x += int(r["correct_end_by_scale"]["1"])
        correct_pass_1x += int(r["correct_by_scale"]["1"])
        n = len(results)
        pbar.set_postfix(
            start=f"{correct_start / n:.3f}",
            end_1x=f"{correct_end_1x / n:.3f}",
            pass_1x=f"{correct_pass_1x / n:.3f}",
        )
    return results


async def run_eval(args) -> dict:
    client = AsyncOpenAI(base_url=f"{args.base_url}/{args.model_id}", api_key=args.api_key)
    sem = asyncio.Semaphore(args.concurrency)
    tasks = build_tasks(args, client, sem)
    results = await gather_results(tasks)
    results.sort(key=lambda r: (r["image"], r["index"]))
    return summarize(results, args)


def main():
    args = parse_args()
    summary = asyncio.run(run_eval(args))
    args.output.write_text(json.dumps(summary, indent=2))
    total = summary["total"]
    print(f"start: {summary['correct_start']}/{total} = {summary['start_accuracy']:.4f}")
    for acc_key, count_key in [
        ("end_accuracy_by_scale", "correct_end_by_scale"),
        ("accuracy_by_scale", "correct_by_scale"),
    ]:
        for label, acc in summary[acc_key].items():
            print(f"{label}: {summary[count_key][label]}/{total} = {acc:.4f}")
    print(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()
