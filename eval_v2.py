import asyncio
import base64
import io
import json
import os
from argparse import ArgumentParser
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI
from PIL import Image
from tqdm.asyncio import tqdm as tqdm_asyncio

load_dotenv()


SCALES = (1, 2, 3)


@dataclass
class BackendConfig:
    base_url: str | None
    api_key: str | None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


BACKENDS: dict[str, BackendConfig] = {
    "vllm": BackendConfig(
        base_url=os.getenv("VLLM_BASE_URL"),
        api_key=os.getenv("VLLM_API_KEY"),
        extra_kwargs={"response_format": {"type": "json_object"}},
    ),
    "openai": BackendConfig(
        base_url="https://api.openai.com/v1",
        api_key=os.getenv("OPENAI_API_KEY"),
        extra_kwargs={"response_format": {"type": "json_object"}},
    ),
    "anthropic": BackendConfig(
        base_url="https://api.anthropic.com/v1",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        extra_kwargs={"extra_body": {"thinking": {"type": "disabled"}}},
    ),
    "together": BackendConfig(
        base_url="https://api.together.xyz/v1",
        api_key=os.getenv("TOGETHER_API_KEY"),
        extra_kwargs={
            "response_format": {"type": "json_object"},
            "temperature": 0.6,
            "top_p": 0.95,
            "extra_body": {"reasoning": {"enabled": False}},
        },
    ),
}


@dataclass
class ModelConfig:
    model_id: str
    backend: str
    coord_format: str
    reasoning: bool = False

    def completion_kwargs(self) -> dict[str, Any]:
        kwargs = dict(BACKENDS[self.backend].extra_kwargs)
        if self.backend == "vllm":
            kwargs = {
                **kwargs,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": self.reasoning}},
            }
        return kwargs

    def client_base_url(self) -> str:
        base = BACKENDS[self.backend].base_url
        # Our vLLM router exposes one model per sub-path.
        return f"{base}/{self.model_id}" if self.backend == "vllm" else base


MODELS: dict[str, ModelConfig] = {
    m.model_id: m
    for m in [
        ModelConfig("claude-sonnet-4-5", backend="anthropic", coord_format="pixel"),
        ModelConfig("claude-opus-4-7", backend="anthropic", coord_format="pixel"),
        ModelConfig("gpt-5.4", backend="openai", coord_format="pixel"),
        ModelConfig("moonshotai/Kimi-K2.5", backend="together", coord_format="unit"),
        ModelConfig("holo3-35b-a3b", backend="vllm", coord_format="normalized"),
        ModelConfig("drag", backend="vllm", coord_format="normalized"),
        ModelConfig(
            "qwen3-5-35b-a3b-fp8", backend="vllm", coord_format="normalized", reasoning=True
        ),
        ModelConfig(
            "qwen3-5-122b-a10b-fp8", backend="vllm", coord_format="normalized", reasoning=True
        ),
        ModelConfig(
            "qwen3-5-397b-a17b-fp8", backend="vllm", coord_format="normalized", reasoning=True
        ),
    ]
}


def coord_schema(coord_format: str, width: int, height: int) -> dict:
    """JSON schema for the four drag coordinates."""
    if coord_format == "pixel":
        x_spec = {"type": "integer", "minimum": 0, "maximum": width}
        y_spec = {"type": "integer", "minimum": 0, "maximum": height}
        units = f"in image pixels (image is {width} wide by {height} tall)"
    elif coord_format == "unit":
        x_spec = y_spec = {"type": "number", "minimum": 0, "maximum": 1}
        units = "as a fraction of the image, between 0 and 1"
    else:  # normalized
        x_spec = y_spec = {"type": "integer", "minimum": 0, "maximum": 1000}
        units = "normalized between 0 and 1000"

    def _prop(axis: str, point: str) -> dict:
        spec = x_spec if axis == "x" else y_spec
        return {**spec, "description": f"{axis} coordinate of the {point} of the drag, {units}"}

    return {
        "type": "object",
        "title": "DragAndDropAction",
        "properties": {
            "action": {"const": "drag_and_drop", "default": "drag_and_drop", "type": "string"},
            "x1": _prop("x", "start"),
            "y1": _prop("y", "start"),
            "x2": _prop("x", "end"),
            "y2": _prop("y", "end"),
        },
        "required": ["x1", "y1", "x2", "y2"],
    }


_COORD_INSTRUCTION = {
    "pixel": "Coordinates must be in image pixels: x in [0, {width}] and y in [0, {height}].",
    "unit": "Coordinates must be expressed as fractions of the image dimensions, between 0.0 and 1.0 (use decimals, e.g. 0.523).",
    "normalized": "Coordinates must be between 0 and 1000.",
}


def build_prompt(coord_format: str, width: int, height: int, task: str) -> str:
    schema = json.dumps(coord_schema(coord_format, width, height))
    coord_rule = _COORD_INSTRUCTION[coord_format].format(width=width, height=height)
    return (
        "Localize the beginning and end of the vector on the GUI image according to the task "
        "and output the coordinates of the beginning and end of the vector. "
        f"You must output a valid JSON following the format: {schema} "
        f"{coord_rule} "
        f"Your drag and drop task is: {task}"
    )


def build_tool_spec(coord_format: str, width: int, height: int) -> dict:
    """OpenAI-compatible tool spec used to force structured output (Anthropic)."""
    return {
        "type": "function",
        "function": {
            "name": "drag_and_drop",
            "description": "Emit the start and end coordinates of a drag-and-drop on the GUI image.",
            "parameters": coord_schema(coord_format, width, height),
        },
    }


SYSTEM_PROMPT = (
    "You are a GUI grounding assistant. "
    "Respond with ONLY a single JSON object matching the requested schema, "
    "with no prose, no explanations, and no markdown code fences."
)


def load_image(path: Path) -> tuple[str, int, int]:
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data_url = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
    return data_url, img.size[0], img.size[1]


def iter_samples(data_dir: Path):
    for images_dir in sorted(data_dir.rglob("images")):
        tasks_dir = images_dir.with_name("tasks")
        if not (images_dir.is_dir() and tasks_dir.is_dir()):
            continue
        domain = images_dir.parent.name
        for img_path in sorted(images_dir.glob("*.jpg")):
            tasks_path = tasks_dir / f"{img_path.stem}.json"
            if not tasks_path.exists():
                continue
            with tasks_path.open() as f:
                for i, sample in enumerate(json.load(f)):
                    yield domain, img_path, i, sample


def normalize_bbox(bbox: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = bbox
    return [x1 / width, y1 / height, x2 / width, y2 / height]


def scale_bbox(bbox: list[float], scale: float) -> list[float]:
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = (x2 - x1) / 2 * scale, (y2 - y1) / 2 * scale
    return [max(0.0, cx - hw), max(0.0, cy - hh), min(1.0, cx + hw), min(1.0, cy + hh)]


def point_in_bbox(x: float, y: float, bbox: list[float]) -> bool:
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def extract_coords(
    pred: dict | None, coord_format: str, width: int, height: int
) -> tuple[float, float, float, float] | None:
    if not isinstance(pred, dict):
        return None
    keys = ("x1", "y1", "x2", "y2")
    if not all(isinstance(pred.get(k), (int, float)) for k in keys):
        return None
    x1, y1, x2, y2 = (float(pred[k]) for k in keys)
    if coord_format == "pixel":
        return x1 / width, y1 / height, x2 / width, y2 / height
    if coord_format == "unit":
        return x1, y1, x2, y2
    return tuple(c / 1000 for c in (x1, y1, x2, y2))  # type: ignore[return-value]


def evaluate(
    pred: dict | None, norm_sample: dict, coord_format: str, width: int, height: int
) -> dict[int, bool]:
    """Per-scale correctness: both endpoints fall in their target bbox (end bbox is scaled)."""
    coords = extract_coords(pred, coord_format, width, height)
    if coords is None:
        return {s: False for s in SCALES}
    x1, y1, x2, y2 = coords
    in_start = point_in_bbox(x1, y1, norm_sample["start_bbox"])
    return {
        s: in_start and point_in_bbox(x2, y2, scale_bbox(norm_sample["end_bbox"], s))
        for s in SCALES
    }


async def predict(
    client: AsyncOpenAI, model: ModelConfig, image_url: str, width: int, height: int, task: str
) -> dict:
    kwargs = model.completion_kwargs()
    if model.backend == "anthropic":
        kwargs["tools"] = [build_tool_spec(model.coord_format, width, height)]
        kwargs["tool_choice"] = {"type": "function", "function": {"name": "drag_and_drop"}}

    response = await client.chat.completions.create(
        model=model.model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(model.coord_format, width, height, task)},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}}]},
        ],
        **kwargs,
    )
    message = response.choices[0].message
    raw = message.tool_calls[0].function.arguments if message.tool_calls else message.content
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        tqdm_asyncio.write(f"\nRAW_RESPONSE {response.model_dump_json()}")
        raise


async def safe_predict(*args, **kwargs) -> tuple[dict | None, str | None]:
    try:
        return await predict(*args, **kwargs), None
    except Exception as e:
        return None, repr(e)


def build_result(
    domain: str,
    img_path: Path,
    index: int,
    sample: dict,
    norm_sample: dict,
    pred: dict | None,
    by_scale: dict[int, bool],
    error: str | None,
) -> dict:
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
        "correct_by_scale": {str(s): by_scale[s] for s in SCALES},
        "error": error,
    }


async def run_one(
    sem: asyncio.Semaphore,
    client: AsyncOpenAI,
    model: ModelConfig,
    domain: str,
    img_path: Path,
    image_url: str,
    width: int,
    height: int,
    index: int,
    sample: dict,
    norm_sample: dict,
    verbose: bool,
) -> dict:
    async with sem:
        pred, error = await safe_predict(client, model, image_url, width, height, sample["intent"])
    by_scale = evaluate(pred, norm_sample, model.coord_format, width, height)
    result = build_result(domain, img_path, index, sample, norm_sample, pred, by_scale, error)
    if verbose:
        status = "OK" if result["correct"] else "FAIL"
        tqdm_asyncio.write(
            f"[{status}] {domain}/{img_path.name}#{index}: {sample['intent'][:60]!r} -> {pred} err={error}"
        )
    return result


def aggregate(results: list[dict]) -> dict:
    n = len(results) or 1
    correct = {s: sum(int(r["correct_by_scale"][str(s)]) for r in results) for s in SCALES}
    return {
        "accuracy": correct[1] / n,
        "correct": correct[1],
        "total": len(results),
        "accuracy_by_scale": {f"accuracy@{s}x": correct[s] / n for s in SCALES},
        "correct_by_scale": {f"accuracy@{s}x": correct[s] for s in SCALES},
    }


def summarize(results: list[dict], model: ModelConfig, concurrency: int) -> dict:
    by_domain: dict[str, list[dict]] = {}
    for r in results:
        by_domain.setdefault(r["domain"], []).append(r)
    return {
        **aggregate(results),
        "by_domain": {d: aggregate(rs) for d, rs in sorted(by_domain.items())},
        "model_id": model.model_id,
        "backend": model.backend,
        "coord_format": model.coord_format,
        "reasoning": model.reasoning,
        "concurrency": concurrency,
        "results": results,
    }


def print_aggregate(label: str, agg: dict) -> None:
    total = agg["total"]
    print(f"\n== {label} (n={total}) ==")
    for k, acc in agg["accuracy_by_scale"].items():
        print(f"{k}: {agg['correct_by_scale'][k]}/{total} = {acc:.4f}")


async def run_eval(model: ModelConfig, data_dir: Path, concurrency: int, verbose: bool) -> dict:
    backend = BACKENDS[model.backend]
    client = AsyncOpenAI(base_url=model.client_base_url(), api_key=backend.api_key)
    sem = asyncio.Semaphore(concurrency)
    tasks = []
    for domain, img_path, i, sample in iter_samples(data_dir):
        image_url, width, height = load_image(img_path)
        norm_sample = {
            **sample,
            "start_bbox": normalize_bbox(sample["start_bbox"], width, height),
            "end_bbox": normalize_bbox(sample["end_bbox"], width, height),
        }
        tasks.append(
            run_one(
                sem,
                client,
                model,
                domain,
                img_path,
                image_url,
                width,
                height,
                i,
                sample,
                norm_sample,
                verbose,
            )
        )

    results: list[dict] = []
    running = {f"acc@{s}x": 0 for s in SCALES}
    pbar = tqdm_asyncio(asyncio.as_completed(tasks), total=len(tasks), desc="eval", unit="sample")
    async for coro in pbar:
        r = await coro
        results.append(r)
        for s in SCALES:
            running[f"acc@{s}x"] += int(r["correct_by_scale"][str(s)])
        n = len(results)
        pbar.set_postfix({k: f"{v / n:.3f}" for k, v in running.items()})

    results.sort(key=lambda r: (r["domain"], r["image"], r["index"]))
    return summarize(results, model, concurrency)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    model = MODELS[args.model]
    print(f"Running eval: model={model}")

    summary = asyncio.run(run_eval(model, args.data_dir, args.concurrency, args.verbose))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_id = model.model_id.replace("/", "_")
    results_path = args.output_dir / f"results_{safe_id}_{timestamp}.json"
    aggregated_path = args.output_dir / f"aggregated_{safe_id}_{timestamp}.json"

    results_path.write_text(json.dumps(summary, indent=2))
    aggregated_path.write_text(
        json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2)
    )

    for domain, agg in summary["by_domain"].items():
        print_aggregate(domain, agg)
    print_aggregate("overall", summary)
    print(f"\nWrote full results to {results_path}")
    print(f"Wrote aggregated results to {aggregated_path}")


if __name__ == "__main__":
    main()
