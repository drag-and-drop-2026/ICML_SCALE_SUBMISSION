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

COORDINATES = [0, 1000]

PROMPT = (
    "Localize the beginning and end of the vector on the GUI image according to the task and output the coordinates of the beginning and end of the vector. "
    f"You must output a valid JSON following the format: {json.dumps(FORMAT)} "
    f"Coordinates must be between {COORDINATES[0]} and {COORDINATES[1]}. "
    "Your drag and drop task is: {task}"
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
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


def image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


async def predict(
    client: AsyncOpenAI,
    model_id: str,
    image_url: str,
    task: str,
    reasoning_effort: str,
) -> dict:
    enable_thinking = reasoning_effort != "minimal"
    kwargs = {}
    if enable_thinking:
        kwargs["reasoning_effort"] = reasoning_effort
    response = await client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": "You are a GUI grounding assistant."},
            {"role": "user", "content": PROMPT.replace("{task}", task)},
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": image_url}}],
            },
        ],
        response_format={"type": "json_object"},
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
        **kwargs,
    )
    msg = response.choices[0].message
    return json.loads(msg.content)


def point_in_bbox(x: float, y: float, bbox: list[float]) -> bool:
    """bbox is [x_min, y_min, x_max, y_max] in normalized [0, 1] coordinates."""
    x_min, y_min, x_max, y_max = bbox
    return x_min <= x <= x_max and y_min <= y <= y_max


def is_correct(pred: dict, sample: dict) -> bool:
    try:
        x1 = pred["x1"] / COORDINATES[1]
        y1 = pred["y1"] / COORDINATES[1]
        x2 = pred["x2"] / COORDINATES[1]
        y2 = pred["y2"] / COORDINATES[1]
    except (KeyError, TypeError):
        return False
    return point_in_bbox(x1, y1, sample["start_bbox"]) and point_in_bbox(x2, y2, sample["end_bbox"])


def iter_samples(data_dir: Path):
    for img_path in sorted(data_dir.glob("*.jpg")):
        json_path = img_path.with_suffix(".json")
        if not json_path.exists():
            continue
        with json_path.open() as f:
            samples = json.load(f)
        yield img_path, json_path, samples


async def run_one(
    sem: asyncio.Semaphore,
    client: AsyncOpenAI,
    model_id: str,
    reasoning_effort: str,
    img_path: Path,
    image_url: str,
    index: int,
    sample: dict,
    state: dict,
    verbose: bool,
) -> dict:
    intent = sample["intent"]
    async with sem:
        try:
            pred = await predict(
                client=client,
                model_id=model_id,
                image_url=image_url,
                task=intent,
                reasoning_effort=reasoning_effort,
            )
            ok = is_correct(pred, sample)
            error = None
        except Exception as e:
            pred = None
            ok = False
            error = repr(e)

    state["total"] += 1
    state["correct"] += int(ok)
    if verbose:
        status = "OK" if ok else "FAIL"
        tqdm_asyncio.write(
            f"[{status}] {img_path.name}#{index}: {intent[:60]!r} -> {pred} err={error}"
        )
    return {
        "image": img_path.name,
        "index": index,
        "intent": intent,
        "start_bbox": sample["start_bbox"],
        "end_bbox": sample["end_bbox"],
        "prediction": pred,
        "correct": ok,
        "error": error,
    }


async def run_eval(args) -> dict:
    url = f"{args.base_url}/{args.model_id}"
    client = AsyncOpenAI(base_url=url, api_key=args.api_key)
    sem = asyncio.Semaphore(args.concurrency)

    work = []
    image_url_cache: dict[Path, str] = {}
    for img_path, _json_path, samples in iter_samples(args.data_dir):
        image = Image.open(img_path).convert("RGB")
        image_url_cache[img_path] = image_to_data_url(image)
        for i, sample in enumerate(samples):
            work.append((img_path, i, sample))

    state = {"correct": 0, "total": 0}
    tasks = [
        run_one(
            sem=sem,
            client=client,
            model_id=args.model_id,
            reasoning_effort=args.reasoning_effort,
            img_path=img_path,
            image_url=image_url_cache[img_path],
            index=i,
            sample=sample,
            state=state,
            verbose=args.verbose,
        )
        for (img_path, i, sample) in work
    ]

    results = []
    pbar = tqdm_asyncio(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc="eval",
        unit="sample",
    )
    async for coro in pbar:
        result = await coro
        results.append(result)
        pbar.set_postfix(
            acc=f"{state['correct']}/{state['total']}={state['correct'] / state['total']:.3f}"
        )

    results.sort(key=lambda r: (r["image"], r["index"]))
    return {
        "accuracy": state["correct"] / state["total"] if state["total"] else 0.0,
        "correct": state["correct"],
        "total": state["total"],
        "model_id": args.model_id,
        "reasoning_effort": args.reasoning_effort,
        "concurrency": args.concurrency,
        "results": results,
    }


def main():
    args = parse_args()
    summary = asyncio.run(run_eval(args))
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"Accuracy: {summary['correct']}/{summary['total']} = {summary['accuracy']:.4f}")
    print(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()
