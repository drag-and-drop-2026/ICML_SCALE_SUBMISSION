import asyncio
import base64
import io
import json
import os
import re
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from PIL import Image
from tqdm.asyncio import tqdm as tqdm_asyncio

load_dotenv()

COORD_MAX = 1000
SCALES = [1, 2, 3]


def _coord_prop(axis: str, kind: str, coord_format: str, x_max: int, y_max: int) -> dict:
    """Build a JSON schema property for an x/y coordinate."""
    is_x = axis == "x"
    title = f"{axis.upper()}{kind}"
    point = "start" if kind == "1" else "end"
    if coord_format == "pixel":
        max_value = x_max if is_x else y_max
        units = f"in image pixels (image is {x_max} wide by {y_max} tall)"
        type_ = "integer"
    elif coord_format == "unit":
        max_value = 1
        units = "as a fraction of the image, between 0 and 1"
        type_ = "number"
    else:
        max_value = COORD_MAX
        units = f"normalized between 0 and {COORD_MAX}"
        type_ = "integer"
    return {
        "description": f"The {axis} coordinate of the {point} of the drag, {units}",
        "maximum": max_value,
        "minimum": 0,
        "title": title,
        "type": type_,
    }


def build_format(coord_format: str, width: int, height: int) -> dict:
    return {
        "properties": {
            "action": {
                "const": "drag_and_drop",
                "default": "drag_and_drop",
                "title": "Action",
                "type": "string",
            },
            "x1": _coord_prop("x", "1", coord_format, width, height),
            "y1": _coord_prop("y", "1", coord_format, width, height),
            "x2": _coord_prop("x", "2", coord_format, width, height),
            "y2": _coord_prop("y", "2", coord_format, width, height),
        },
        "required": ["x1", "y1", "x2", "y2"],
        "title": "DragAndDropAction",
        "type": "object",
    }


PROMPT_NORMALIZED = (
    "Localize the beginning and end of the vector on the GUI image according to the task and output the coordinates of the beginning and end of the vector. "
    "You must output a valid JSON following the format: {format_json} "
    "Coordinates must be between 0 and {coord_max}. "
    "Your drag and drop task is: {task}"
)

PROMPT_PIXEL = (
    "Localize the beginning and end of the vector on the GUI image according to the task and output the coordinates of the beginning and end of the vector. "
    "You must output a valid JSON following the format: {format_json} "
    "Coordinates must be in image pixels: x in [0, {width}] and y in [0, {height}]. "
    "Your drag and drop task is: {task}"
)

PROMPT_UNIT = (
    "Localize the beginning and end of the vector on the GUI image according to the task and output the coordinates of the beginning and end of the vector. "
    "You must output a valid JSON following the format: {format_json} "
    "Coordinates must be expressed as fractions of the image dimensions, between 0.0 and 1.0 (use decimals, e.g. 0.523). "
    "Your drag and drop task is: {task}"
)


def format_prompt(coord_format: str, format_json: str, width: int, height: int, task: str) -> str:
    if coord_format == "pixel":
        return PROMPT_PIXEL.format(format_json=format_json, width=width, height=height, task=task)
    if coord_format == "unit":
        return PROMPT_UNIT.format(format_json=format_json, task=task)
    return PROMPT_NORMALIZED.format(format_json=format_json, coord_max=COORD_MAX, task=task)


BACKENDS = {
    "vllm": {
        "default_base_url": os.getenv("VLLM_BASE_URL"),
        "api_key": os.getenv("VLLM_API_KEY"),
        "coord_format": "normalized",
    },
    "openai": {
        "default_base_url": "https://api.openai.com/v1",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "coord_format": "pixel",
    },
    "anthropic": {
        "default_base_url": "https://api.anthropic.com/v1",
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
        "coord_format": "pixel",
    },
    "together": {
        "default_base_url": "https://api.together.xyz/v1",
        "api_key": os.getenv("TOGETHER_API_KEY"),
        "coord_format": "unit",
    },
}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model-id", default="qwen3-5-397b-a17b-fp8")
    parser.add_argument("--backend", default="vllm", choices=list(BACKENDS))
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory to write results_<model>_<timestamp>.json and aggregated_<model>_<timestamp>.json",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--concurrency", type=int, default=50)
    args = parser.parse_args()
    cfg = BACKENDS[args.backend]
    if args.base_url is None:
        args.base_url = cfg["default_base_url"]
    if args.api_key is None:
        args.api_key = cfg["api_key"]
    args.coord_format = cfg["coord_format"]
    if not args.base_url:
        parser.error(f"--base-url not set for backend={args.backend!r}")
    return args


def load_image(path: Path) -> tuple[str, int, int]:
    """Return (data_url, width, height) for an image."""
    img = Image.open(path).convert("RGB")
    width, height = img.size
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data_url = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
    return data_url, width, height


def normalize_bbox(bbox: list[float], width: int, height: int) -> list[float]:
    """Convert a pixel-space bbox [x_min, y_min, x_max, y_max] to normalized [0, 1]."""
    x_min, y_min, x_max, y_max = bbox
    return [x_min / width, y_min / height, x_max / width, y_max / height]


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


def extract_coords(
    pred: dict | None, coord_format: str, width: int, height: int
) -> tuple[float, float, float, float] | None:
    """Return normalized (x1, y1, x2, y2) in [0, 1], or None if pred is malformed."""
    if not isinstance(pred, dict):
        return None
    keys = ("x1", "y1", "x2", "y2")
    if not all(isinstance(pred.get(k), (int, float)) for k in keys):
        return None
    if coord_format == "pixel":
        x1 = pred["x1"] / width
        y1 = pred["y1"] / height
        x2 = pred["x2"] / width
        y2 = pred["y2"] / height
    elif coord_format == "unit":
        x1, y1, x2, y2 = (float(pred[k]) for k in keys)
    else:
        x1, y1, x2, y2 = (pred[k] / COORD_MAX for k in keys)
    return x1, y1, x2, y2


def evaluate_pred(
    pred: dict | None, sample: dict, coord_format: str, width: int, height: int
) -> tuple[bool, dict[int, bool]]:
    """Return (start_ok, end_ok_by_scale) for a single prediction."""
    coords = extract_coords(pred, coord_format, width, height)
    if coords is None:
        return False, {s: False for s in SCALES}
    x1, y1, x2, y2 = coords
    start_ok = point_in_bbox(x1, y1, sample["start_bbox"])
    end_by_scale = {s: point_in_bbox(x2, y2, scale_bbox(sample["end_bbox"], s)) for s in SCALES}
    return start_ok, end_by_scale


def iter_image_paths(data_dir: Path):
    """Yield (domain, image_path, tasks_path) tuples for all images under data_dir.

    Supports both the new layout (data_dir/<domain>/{images,tasks}/<id>.{jpg,json})
    and a single-domain layout (data_dir/{images,tasks}/<id>.{jpg,json}).
    """
    images_dirs = sorted(data_dir.rglob("images"))
    for images_dir in images_dirs:
        if not images_dir.is_dir():
            continue
        tasks_dir = images_dir.with_name("tasks")
        if not tasks_dir.is_dir():
            continue
        domain = images_dir.parent.name
        for img_path in sorted(images_dir.glob("*.jpg")):
            tasks_path = tasks_dir / f"{img_path.stem}.json"
            if tasks_path.exists():
                yield domain, img_path, tasks_path


def iter_samples(data_dir: Path):
    for domain, img_path, tasks_path in iter_image_paths(data_dir):
        with tasks_path.open() as f:
            for i, sample in enumerate(json.load(f)):
                yield domain, img_path, i, sample


def completion_kwargs(args) -> dict:
    if args.backend == "anthropic":
        return {"extra_body": {"thinking": {"type": "disabled"}}}
    if args.backend == "together":
        return {
            "response_format": {"type": "json_object"},
            "temperature": 0.6,
            "top_p": 0.95,
            "extra_body": {"reasoning": {"enabled": False}},
        }
    if args.backend == "vllm":
        return {
            "response_format": {"type": "json_object"},
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
    return {"response_format": {"type": "json_object"}}


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _extract_json_object(text: str) -> str | None:
    """Find the first balanced {...} JSON object in text (ignoring braces in strings)."""
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                return text[start : i + 1]
    return None


def parse_json_response(content: str) -> dict:
    """Parse JSON from a model response, tolerating code fences and surrounding prose."""
    if content is None:
        raise json.JSONDecodeError("empty response", "", 0)
    text = content.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    obj = _extract_json_object(text)
    if obj is not None:
        return json.loads(obj)
    raise json.JSONDecodeError(f"no JSON object found in: {text[:200]!r}", text, 0)


async def predict(
    client: AsyncOpenAI, args, image_url: str, width: int, height: int, task: str
) -> dict:
    format_json = json.dumps(build_format(args.coord_format, width, height))
    prompt = format_prompt(args.coord_format, format_json, width, height, task)
    response = await client.chat.completions.create(
        model=args.model_id,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a GUI grounding assistant. "
                    "Respond with ONLY a single JSON object matching the requested schema, "
                    "with no prose, no explanations, and no markdown code fences."
                ),
            },
            {"role": "user", "content": prompt},
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": image_url}}],
            },
        ],
        **completion_kwargs(args),
    )
    try:
        return parse_json_response(response.choices[0].message.content)
    except json.JSONDecodeError:
        tqdm_asyncio.write(f"\nRAW_RESPONSE {response.model_dump_json()}")
        raise


async def safe_predict(
    client: AsyncOpenAI, args, image_url: str, width: int, height: int, task: str
) -> tuple[dict | None, str | None]:
    try:
        return await predict(client, args, image_url, width, height, task), None
    except Exception as e:
        return None, repr(e)


def build_result(
    domain: str,
    img_path: Path,
    index: int,
    sample: dict,
    norm_sample: dict,
    pred: dict | None,
    start_ok: bool,
    end_by_scale: dict[int, bool],
    error: str | None,
) -> dict:
    by_scale = {s: start_ok and end_by_scale[s] for s in SCALES}
    return {
        "domain": domain,
        "subtype": sample.get("subtype"),
        "image": img_path.name,
        "image_id": sample.get("image_id", img_path.stem),
        "index": index,
        "intent": sample["intent"],
        "start_bbox_px": sample["start_bbox"],
        "end_bbox_px": sample["end_bbox"],
        "start_bbox": norm_sample["start_bbox"],
        "end_bbox": norm_sample["end_bbox"],
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
    domain: str,
    img_path: Path,
    image_url: str,
    width: int,
    height: int,
    index: int,
    sample: dict,
    norm_sample: dict,
) -> dict:
    async with sem:
        pred, error = await safe_predict(client, args, image_url, width, height, sample["intent"])
    start_ok, end_by_scale = evaluate_pred(pred, norm_sample, args.coord_format, width, height)
    result = build_result(
        domain, img_path, index, sample, norm_sample, pred, start_ok, end_by_scale, error
    )
    if args.verbose:
        status = "OK" if result["correct"] else "FAIL"
        intent = sample["intent"]
        tqdm_asyncio.write(
            f"[{status}] {domain}/{img_path.name}#{index}: {intent[:60]!r} -> {pred} err={error}"
        )
    return result


def _aggregate(results: list[dict]) -> dict:
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
        "accuracy_by_scale": {f"accuracy@{s}x": correct[s] / safe for s in SCALES},
        "correct_by_scale": {f"accuracy@{s}x": correct[s] for s in SCALES},
    }


def summarize(results: list[dict], args) -> dict:
    by_domain: dict[str, list[dict]] = {}
    for r in results:
        by_domain.setdefault(r["domain"], []).append(r)
    return {
        **_aggregate(results),
        "by_domain": {d: _aggregate(rs) for d, rs in sorted(by_domain.items())},
        "model_id": args.model_id,
        "concurrency": args.concurrency,
        "results": results,
    }


def build_tasks(args, client: AsyncOpenAI, sem: asyncio.Semaphore) -> list:
    cache: dict[Path, tuple[str, int, int]] = {}
    tasks = []
    for domain, img_path, i, sample in iter_samples(args.data_dir):
        if img_path not in cache:
            cache[img_path] = load_image(img_path)
        image_url, width, height = cache[img_path]
        norm_sample = {
            **sample,
            "start_bbox": normalize_bbox(sample["start_bbox"], width, height),
            "end_bbox": normalize_bbox(sample["end_bbox"], width, height),
        }
        tasks.append(
            run_one(
                sem,
                client,
                args,
                domain,
                img_path,
                image_url,
                width,
                height,
                i,
                sample,
                norm_sample,
            )
        )
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


def client_base_url(args) -> str:
    if args.backend == "vllm":
        return f"{args.base_url}/{args.model_id}"
    return args.base_url


async def run_eval(args) -> dict:
    client = AsyncOpenAI(base_url=client_base_url(args), api_key=args.api_key)
    sem = asyncio.Semaphore(args.concurrency)
    tasks = build_tasks(args, client, sem)
    results = await gather_results(tasks)
    results.sort(key=lambda r: (r["domain"], r["image"], r["index"]))
    return summarize(results, args)


def print_aggregate(label: str, agg: dict):
    total = agg["total"]
    print(f"\n== {label} (n={total}) ==")
    print(f"start: {agg['correct_start']}/{total} = {agg['start_accuracy']:.4f}")
    for acc_key, count_key in [
        ("end_accuracy_by_scale", "correct_end_by_scale"),
        ("accuracy_by_scale", "correct_by_scale"),
    ]:
        for k, acc in agg[acc_key].items():
            print(f"{k}: {agg[count_key][k]}/{total} = {acc:.4f}")


def aggregate_only(summary: dict) -> dict:
    return {k: v for k, v in summary.items() if k != "results"} | {
        "by_domain": {d: dict(agg) for d, agg in summary["by_domain"].items()},
    }


def main():
    args = parse_args()
    print(args)
    summary = asyncio.run(run_eval(args))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_model_id = args.model_id.replace("/", "_")
    results_path = args.output_dir / f"results_{safe_model_id}_{timestamp}.json"
    aggregated_path = args.output_dir / f"aggregated_{safe_model_id}_{timestamp}.json"

    results_path.write_text(json.dumps(summary, indent=2))
    aggregated_path.write_text(json.dumps(aggregate_only(summary), indent=2))

    for domain, agg in summary["by_domain"].items():
        print_aggregate(domain, agg)
    print_aggregate("overall", summary)
    print(f"\nWrote full results to {results_path}")
    print(f"Wrote aggregated results to {aggregated_path}")


if __name__ == "__main__":
    main()
