#!/usr/bin/env python3
"""Vietnamese localization of English FDB's 0--5 interruption relevance judge."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI


SYSTEM_PROMPT = """Bạn đánh giá hội thoại nói song công bằng tiếng Việt.
Người dùng nói một yêu cầu, trợ lý bắt đầu trả lời, rồi người dùng ngắt lời bằng
một yêu cầu sửa đổi hoặc chuyển hướng. Hãy đánh giá CHỈ mức độ liên quan của phần
trả lời sau lần ngắt lời đối với nội dung ngắt lời.

Thang điểm tương thích với Full-Duplex-Bench tiếng Anh:
0 = hoàn toàn không liên quan; 1 = không liên quan; 2 = hơi liên quan;
3 = có liên quan; 4 = rất liên quan; 5 = đáp ứng hoàn toàn nội dung ngắt lời.

Không cho điểm cao chỉ vì câu trả lời liên quan tới yêu cầu ban đầu. Nếu người
dùng thay đổi thời gian, số lượng, đối tượng, đích đến hoặc ý định, câu trả lời
phải tiếp nhận thay đổi đó mới được 4--5. Chỉ chấm relevance; các lỗi về thời
điểm, nhường lượt và vai người nói được chấm bằng metric role-aware riêng.
Trả về JSON đúng schema đã yêu cầu."""

SCHEMA = {
    "type": "object",
    "properties": {
        "analysis_vi": {"type": "string"},
        "rating": {"type": "integer", "minimum": 0, "maximum": 5},
    },
    "required": ["analysis_vi", "rating"],
    "additionalProperties": False,
}


def judge_one(client: OpenAI, sample: Path, model: str, overwrite: bool) -> dict:
    target = sample / "interruption_relevance.json"
    if target.exists() and not overwrite:
        return json.loads(target.read_text(encoding="utf-8"))
    annotation = json.loads((sample / "interrupt.json").read_text(encoding="utf-8"))[0]
    output = json.loads((sample / "output.json").read_text(encoding="utf-8"))
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Yêu cầu ban đầu: {annotation['context']}\n"
                    f"Lời ngắt/sửa đổi: {annotation['interrupt']}\n"
                    f"Trả lời của trợ lý sau ngắt lời: {output.get('text') or '[KHÔNG CÓ LỜI NÓI]'}"
                ),
            },
        ],
        text={"format": {"type": "json_schema", "name": "relevance", "strict": True, "schema": SCHEMA}},
        temperature=0,
    )
    verdict = json.loads(response.output_text)
    record = {
        "schema_version": 1,
        "metric": "english_fdb_interruption_relevance_vi",
        "judge_model": model,
        "sample_id": sample.name,
        **verdict,
    }
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--jobs", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    samples = sorted(path for path in args.root.iterdir() if path.is_dir())
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    records = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(judge_one, client, sample, args.model, args.overwrite) for sample in samples]
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda record: record["sample_id"])
    distribution = {str(score): sum(r["rating"] == score for r in records) for score in range(6)}
    summary = {
        "metric": "english_fdb_interruption_relevance_vi",
        "judge_model": args.model,
        "samples": len(records),
        "mean_rating": sum(r["rating"] for r in records) / len(records),
        "distribution": distribution,
        "records": records,
    }
    destination = args.summary or args.root / "interruption_relevance_summary.json"
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
