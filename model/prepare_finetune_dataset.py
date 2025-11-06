import json
from collections import defaultdict
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
INPUT_PATH = THIS_DIR / "eval_results_1.jsonl"
OUTPUT_PATH = THIS_DIR / "gemma_lora_dataset.jsonl"

MAX_CTX_CHARS = 400


def clamp_tail(text: str, limit: int) -> str:
    if not text or limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return "…" + text[-(limit - 1):]


def load_meeting_contexts(meeting_path: Path):
    data = json.loads(meeting_path.read_text(encoding="utf-8", errors="ignore"))
    utterances = data.get("utterance") or []
    history = defaultdict(list)
    contexts = {}
    for idx, entry in enumerate(utterances):
        if not isinstance(entry, dict):
            continue
        speaker = (entry.get("speaker_id") or "").strip()
        sentence = (entry.get("form") or entry.get("original_form") or "").strip()
        context_text = " ".join(history[speaker])
        contexts[idx] = clamp_tail(context_text, MAX_CTX_CHARS)
        if speaker and sentence:
            history[speaker].append(sentence)
    return contexts


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {INPUT_PATH}")

    meeting_cache = {}
    processed = 0

    with OUTPUT_PATH.open("w", encoding="utf-8") as fout, INPUT_PATH.open(
        "r", encoding="utf-8"
    ) as fin:
        for line in fin:
            if not line.strip():
                continue
            record = json.loads(line)
            file_path = PROJECT_ROOT / record["file_path"]
            utter_idx = record["utterance_index"]
            meeting_id = record["meeting_id"]

            if file_path not in meeting_cache:
                meeting_cache[file_path] = load_meeting_contexts(file_path)
            contexts = meeting_cache[file_path]
            context_text = contexts.get(utter_idx, "")

            # New prompt: include topic, speaker_id, speaker_context, sentence (up to sentence line)
            prompt = (
                f"topic: {record['topic']}\n"
                f"speaker_id: {record['speaker_id']}\n"
                f"speaker_context: {context_text or '없음'}\n"
                f"sentence: {record['sentence']}\n"
            )

            # New response: "<이유> (subj, nov)"
            def _fmt(x: float) -> str:
                s = f"{float(x):.1f}"
                return s.rstrip("0").rstrip(".")

            reason = (record.get("reason") or "").strip().replace("\n", " ")
            subj = _fmt(record.get("subject_score"))
            nov = _fmt(record.get("novelty_score"))

            output = {
                "meeting_id": meeting_id,
                "utterance_index": utter_idx,
                "prompt": prompt,
                "response": f"{reason} ({subj}, {nov})",
            }
            fout.write(json.dumps(output, ensure_ascii=False) + "\n")
            processed += 1

    print(f"✅ 총 {processed}개의 샘플을 {OUTPUT_PATH}에 저장했습니다.")


if __name__ == "__main__":
    main()
