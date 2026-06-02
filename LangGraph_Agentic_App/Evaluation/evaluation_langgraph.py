from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Iterable, Optional

import dotenv
import pandas as pd
from datasets import Dataset
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
)
# LLM_MODEL = "gpt-4o-mini"
LLM_PROVIDER = "anthropic"
LLM_MODEL = "claude-sonnet-4-6"

LANGGRAPH_ROOT = Path(__file__).resolve().parent.parent
if str(LANGGRAPH_ROOT) not in sys.path:
	sys.path.insert(0, str(LANGGRAPH_ROOT))

ENV_PATH = LANGGRAPH_ROOT.parent / ".env"
print(f"Loading environment variables from: {ENV_PATH}")
if dotenv is not None:
	dotenv.load_dotenv(dotenv_path=ENV_PATH)
import config  # noqa: F401  # Loads LangSmith env vars for tracing.
from core.llm import get_llm
from langsmith import traceable

# Disable tracing by default to avoid RAGAS callback handler crashes on empty traces.
# Tracing is conditionally enabled in _prepare_langsmith_tracing() if API credentials are present.
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
EVALUATION_ROOT = Path(__file__).resolve().parent
GROUND_TRUTHS_DIR = EVALUATION_ROOT / "ground_truths"
OUTPUTS_DIR = EVALUATION_ROOT / "outputs"
RESULTS_DIR = EVALUATION_ROOT / "results"

# Set these before running, or override via CLI arguments.
GROUND_TRUTH_FILENAME = "GT_NR_cell_selection_qrxlevmin_reselection.csv"
PREDICTION_FILENAME   = "NR_cell_selection_qrxlevmin_reselection_sequence_messages_1.csv"


ENABLE_RULE_BASED_SCORING = False
AUTO_DERIVE_RULES_FROM_GROUND_TRUTH = True

# Optional explicit mandatory subsequence rules. The sequence values are compared
# after normalization, so spacing and case differences are ignored.
RULE_SEQUENCE_DEFINITIONS = [
	# Example:
	# {
	#     "rule_name": "rrc_setup_handshake",
	#     "category": "UE STATE 1N-A",
	#     "sequence": ["RRCSetupRequest", "RRCSetup", "RRCSetupComplete"],
	# },
]

REQUIRED_COLUMNS = {"STEP", "MESSAGE NAME", "CATEGORY"}
OPTIONAL_COLUMNS = {"CELL", "DIRECTION", "LAYER"}


def _prepare_langsmith_tracing() -> bool:
	api_key = str(os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY") or "").strip()
	if api_key:
		return True
	os.environ["LANGCHAIN_TRACING_V2"] = "false"
	os.environ["LANGSMITH_TRACING"] = "false"
	return False


LANGSMITH_TRACING_ENABLED = _prepare_langsmith_tracing()


def _initialize_ragas_evaluator():
	"""Initialize RAGAS LLM and embeddings for answer evaluation."""
	try:
		llm = get_llm(provider=LLM_PROVIDER, model=LLM_MODEL, temperature=0.1)
		model_cache_dir = str(LANGGRAPH_ROOT.parent.parent / "models")
		embeddings = HuggingFaceEmbeddings(
			model_name="BAAI/bge-large-en-v1.5",
			cache_folder=model_cache_dir,
			show_progress=False,
			encode_kwargs={"normalize_embeddings": True},
		)
		print(f"✓ RAGAS initialized with LLM={LLM_MODEL}, embeddings=BAAI/bge-large-en-v1.5")
		return llm, embeddings
	except Exception as e:
		print(f"✗ RAGAS initialization failed: {e}. Skipping answer metrics.")
		return None, None


RAGAS_LLM, RAGAS_EMBEDDINGS = _initialize_ragas_evaluator()
print(RAGAS_LLM, RAGAS_EMBEDDINGS)


@dataclass(frozen=True)
class SequenceRecord:
	step: Optional[int]
	cell: str
	direction: str
	layer: str
	name: str
	category: str
	normalized_name: str
	normalized_direction: str
	normalized_layer: str
	normalized_category: str
	token: str

	def to_export_dict(self) -> dict:
		return {
			"step": self.step,
			"cell": self.cell,
			"direction": self.direction,
			"layer": self.layer,
			"name": self.name,
			"category": self.category,
		}


@dataclass(frozen=True)
class PhaseBlock:
	phase_id: str
	category: str
	normalized_category: str
	occurrence_index: int
	records: tuple[SequenceRecord, ...]


def _clean_filename(value: str) -> str:
	return re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())


def _canonical_text(value: object) -> str:
	text = str(value or "").strip().upper()
	return re.sub(r"[^A-Z0-9]+", "", text)


def _normalize_label(value: object) -> str:
	text = str(value or "").strip().upper()
	return re.sub(r"\s+", " ", text)


def _normalize_direction(value: object) -> str:
	text = _canonical_text(value)
	aliases = {
		"UETOGNB": "UE_TO_GNB",
		"GNBTOUE": "GNB_TO_UE",
		"BIDIRECTIONAL": "BIDIRECTIONAL",
		"UNKNOWN": "UNKNOWN",
	}
	return aliases.get(text, _normalize_label(value) or "UNKNOWN")


def _safe_int(value: object) -> Optional[int]:
	text = str(value or "").strip()
	if not text:
		return None
	try:
		return int(float(text))
	except ValueError:
		return None


def _sequence_to_tokens(records: Iterable[SequenceRecord]) -> list[str]:
	return [record.token for record in records]


def _json_dump(value: object) -> str:
	return json.dumps(value, ensure_ascii=False)


def _multiset_overlap(left: Iterable[str], right: Iterable[str]) -> int:
	return sum((Counter(left) & Counter(right)).values())


def _lcs_length(left: list[str], right: list[str]) -> int:
	if not left or not right:
		return 0
	previous = [0] * (len(right) + 1)
	for left_token in left:
		current = [0]
		for index, right_token in enumerate(right, start=1):
			if left_token == right_token:
				current.append(previous[index - 1] + 1)
			else:
				current.append(max(previous[index], current[-1]))
		previous = current
	return previous[-1]


def _is_subsequence(needle: list[str], haystack: list[str]) -> bool:
	if not needle:
		return True
	iterator = iter(haystack)
	return all(any(candidate == token for candidate in iterator) for token in needle)


def _format_transition(edge: tuple[str, str]) -> str:
	return f"{edge[0]} -> {edge[1]}"


def _parse_metadata_and_table(csv_path: Path) -> tuple[dict[str, str], pd.DataFrame]:
	text = csv_path.read_text(encoding="utf-8")
	lines = text.splitlines(keepends=True)

	header_index = None
	for index, line in enumerate(lines):
		if line.strip().upper().startswith("STEP,"):
			header_index = index
			break

	if header_index is None:
		raise ValueError(f"Could not find STEP header row in {csv_path}")

	metadata: dict[str, str] = {}
	for line in lines[:header_index]:
		if not line.strip():
			continue
		parsed = next(csv.reader([line]))
		if not parsed:
			continue
		key = str(parsed[0] or "").strip()
		if not key:
			continue
		value = ",".join(parsed[1:]).strip()
		metadata[key] = value

	data_frame = pd.read_csv(StringIO("".join(lines[header_index:])))
	data_frame.columns = [str(column).strip().upper() for column in data_frame.columns]

	missing_columns = REQUIRED_COLUMNS - set(data_frame.columns)
	if missing_columns:
		missing_text = ", ".join(sorted(missing_columns))
		raise ValueError(f"{csv_path} is missing required columns: {missing_text}")

	for optional_column in OPTIONAL_COLUMNS:
		if optional_column not in data_frame.columns:
			data_frame[optional_column] = ""

	# Drop only trailing blank rows so footer delimiters like ',,,,,' do not
	# become synthetic 'nan' records while preserving in-table row alignment.
	if not data_frame.empty:
		required_subset = data_frame[list(REQUIRED_COLUMNS)].copy()
		normalized_subset = required_subset.apply(lambda column: column.astype(str).str.strip().str.upper())
		valid_row_mask = (normalized_subset != "") & (normalized_subset != "NAN")
		valid_row_mask = valid_row_mask.any(axis=1)
		if valid_row_mask.any():
			last_valid_label = valid_row_mask[valid_row_mask].index[-1]
			last_valid_position = data_frame.index.get_loc(last_valid_label)
			data_frame = data_frame.iloc[: last_valid_position + 1]

	if "STEP" in data_frame.columns:
		data_frame["_STEP_SORT"] = data_frame["STEP"].apply(_safe_int)
		data_frame = data_frame.sort_values(by=["_STEP_SORT"], kind="stable", na_position="last")
		data_frame = data_frame.drop(columns=["_STEP_SORT"])

	return metadata, data_frame.reset_index(drop=True)


def _records_from_dataframe(data_frame: pd.DataFrame) -> list[SequenceRecord]:
	records: list[SequenceRecord] = []
	for _, row in data_frame.iterrows():
		name = str(row.get("MESSAGE NAME", "") or "").strip()
		if not name:
			continue
		direction = _normalize_direction(row.get("DIRECTION", ""))
		record = SequenceRecord(
			step=_safe_int(row.get("STEP")),
			cell=str(row.get("CELL", "") or "").strip(),
			direction=direction,
			layer=str(row.get("LAYER", "") or "").strip(),
			name=name,
			category=str(row.get("CATEGORY", "") or "").strip() or "UNCLASSIFIED",
			normalized_name=_canonical_text(name),
			normalized_direction=direction,
			normalized_layer=_normalize_label(row.get("LAYER", "")),
			normalized_category=_normalize_label(row.get("CATEGORY", "") or "UNCLASSIFIED"),
			token=f"{_canonical_text(name)}|{direction}",
		)
		records.append(record)
	return records


def _build_phase_blocks(records: list[SequenceRecord]) -> list[PhaseBlock]:
	if not records:
		return []

	occurrences: defaultdict[str, int] = defaultdict(int)
	blocks: list[PhaseBlock] = []
	current_records: list[SequenceRecord] = []
	current_category_key = records[0].normalized_category
	current_category_label = records[0].category

	def finalize_block() -> None:
		if not current_records:
			return
		occurrences[current_category_key] += 1
		occurrence_index = occurrences[current_category_key]
		phase_id = f"{current_category_key}__{occurrence_index}"
		blocks.append(
			PhaseBlock(
				phase_id=phase_id,
				category=current_category_label,
				normalized_category=current_category_key,
				occurrence_index=occurrence_index,
				records=tuple(current_records),
			)
		)

	for record in records:
		if record.normalized_category != current_category_key:
			finalize_block()
			current_records = []
			current_category_key = record.normalized_category
			current_category_label = record.category
		current_records.append(record)
	finalize_block()
	return blocks


def _sequence_to_text(records: list[SequenceRecord]) -> str:
	"""Convert a sequence of records to human-readable text for RAGAS evaluation."""
	if not records:
		return ""
	return " -> ".join([record.name for record in records])


@traceable(name="load_sequence_csv", run_type="parser", tags=["evaluation", "csv"])
def load_sequence_csv(csv_path: Path) -> dict:
	metadata, data_frame = _parse_metadata_and_table(csv_path)
	records = _records_from_dataframe(data_frame)
	phase_blocks = _build_phase_blocks(records)
	return {
		"path": str(csv_path),
		"metadata": metadata,
		"records": records,
		"phase_blocks": phase_blocks,
	}


@traceable(name="score_phase_block", run_type="tool", tags=["evaluation", "phase"])
def score_phase_block(
	ground_truth_block: PhaseBlock,
	prediction_block: Optional[PhaseBlock],
	enable_rule_based_scoring: bool,
) -> dict:
	gt_records = list(ground_truth_block.records)
	pred_records = list(prediction_block.records) if prediction_block else []

	gt_tokens = _sequence_to_tokens(gt_records)
	pred_tokens = _sequence_to_tokens(pred_records)

	lcs_length = _lcs_length(gt_tokens, pred_tokens)
	lcs_score = lcs_length / len(gt_tokens) if gt_tokens else 1.0

	overlap = _multiset_overlap(gt_tokens, pred_tokens)
	precision = overlap / len(pred_tokens) if pred_tokens else (1.0 if not gt_tokens else 0.0)
	recall = overlap / len(gt_tokens) if gt_tokens else 1.0
	f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0

	phase_level_score = (0.65 * lcs_score) + (0.35 * f1)

	gt_transitions = list(zip(gt_tokens, gt_tokens[1:]))
	pred_transitions = list(zip(pred_tokens, pred_tokens[1:]))
	matched_transitions = _multiset_overlap(gt_transitions, pred_transitions)
	transition_recall = matched_transitions / len(gt_transitions) if gt_transitions else 1.0
	transition_precision = matched_transitions / len(pred_transitions) if pred_transitions else (1.0 if not gt_transitions else 0.0)
	transition_accuracy = transition_recall
	transition_f1 = (
		2 * transition_precision * transition_recall / (transition_precision + transition_recall)
		if transition_precision + transition_recall
		else 0.0
	)

	gt_transition_counter = Counter(gt_transitions)
	pred_transition_counter = Counter(pred_transitions)
	missing_transitions = list((gt_transition_counter - pred_transition_counter).elements())
	unexpected_transitions = list((pred_transition_counter - gt_transition_counter).elements())

	rule_based_score: Optional[float]
	if enable_rule_based_scoring:
		rule_based_score = 1.0 if _is_subsequence(gt_tokens, pred_tokens) else 0.0
	else:
		rule_based_score = None

	return {
		"scope": "phase",
		"phase_id": ground_truth_block.phase_id,
		"category": ground_truth_block.category,
		"gt_count": len(gt_records),
		"pred_count": len(pred_records),
		"phase_level_score": round(phase_level_score, 6),
		"lcs_score": round(lcs_score, 6),
		"transition_accuracy": round(transition_accuracy, 6),
		"transition_precision": round(transition_precision, 6),
		"transition_recall": round(transition_recall, 6),
		"transition_f1": round(transition_f1, 6),
		"missing_transitions": _json_dump([_format_transition(edge) for edge in missing_transitions[:25]]),
		"unexpected_transitions": _json_dump([_format_transition(edge) for edge in unexpected_transitions[:25]]),
		"ground_truth_list": _json_dump([record.to_export_dict() for record in gt_records]),
		"prediction_list": _json_dump([record.to_export_dict() for record in pred_records]),
	}


def _build_rule_catalog(gt_blocks: list[PhaseBlock]) -> list[dict]:
	rules = [dict(rule) for rule in RULE_SEQUENCE_DEFINITIONS]
	if not AUTO_DERIVE_RULES_FROM_GROUND_TRUTH:
		return rules

	for block in gt_blocks:
		rules.append(
			{
				"rule_name": f"phase_subsequence::{block.phase_id}",
				"category": block.category,
				"sequence": [record.name for record in block.records],
			}
		)
	return rules


@traceable(name="score_rule_catalog", run_type="tool", tags=["evaluation", "rules"])
def score_rule_catalog(gt_blocks: list[PhaseBlock], pred_blocks: list[PhaseBlock]) -> dict:
	rule_catalog = _build_rule_catalog(gt_blocks)
	if not rule_catalog:
		return {"rule_based_score": None, "rule_details": []}

	gt_phase_lookup: dict[tuple[str, int], PhaseBlock] = {
		(block.normalized_category, block.occurrence_index): block for block in gt_blocks
	}
	pred_phase_lookup: dict[tuple[str, int], PhaseBlock] = {
		(block.normalized_category, block.occurrence_index): block for block in pred_blocks
	}

	results: list[dict] = []
	passes = 0
	for rule in rule_catalog:
		category_key = _normalize_label(rule.get("category", ""))
		matching_gt_block = None
		for lookup_key, gt_block in gt_phase_lookup.items():
			if lookup_key[0] == category_key:
				matching_gt_block = gt_block
				break

		if matching_gt_block is None:
			continue

		matching_pred_block = pred_phase_lookup.get(
			(matching_gt_block.normalized_category, matching_gt_block.occurrence_index)
		)

		# Match only on message names for rule checking so direction changes can be
		# diagnosed separately without double-penalizing the rule layer.
		pred_name_tokens = [record.normalized_name for record in matching_pred_block.records] if matching_pred_block else []
		required_name_tokens = [_canonical_text(name) for name in rule.get("sequence", [])]
		passed = _is_subsequence(required_name_tokens, pred_name_tokens)
		passes += int(passed)
		results.append(
			{
				"rule_name": str(rule.get("rule_name", "unnamed_rule")),
				"category": matching_gt_block.category,
				"passed": passed,
				"required_sequence": required_name_tokens,
				"prediction_sequence": pred_name_tokens,
			}
		)

	score = passes / len(results) if results else None
	return {"rule_based_score": score, "rule_details": results}

@traceable(name="compute_ragas_answer_metrics", run_type="tool", tags=["evaluation", "ragas"])
def compute_ragas_answer_metrics(gt_records: list[SequenceRecord], pred_records: list[SequenceRecord]) -> dict:
	"""Compute RAGAS answer metrics (relevancy, correctness, accuracy) for sequences."""
	if RAGAS_LLM is None or RAGAS_EMBEDDINGS is None:
		return {"answer_relevancy": None, "answer_correctness": None, "answer_accuracy": None}

	try:
		gt_text = _sequence_to_text(gt_records)
		pred_text = _sequence_to_text(pred_records)

		if not gt_text or not pred_text:
			return {"answer_relevancy": None, "answer_correctness": None, "answer_accuracy": None}

		ragas_df = pd.DataFrame({
			"question": ["sequence_evaluation"],
			"answer": [pred_text],
			"ground_truth": [gt_text],
		})

		ragas_dataset = Dataset.from_pandas(ragas_df)
		
		try:
			ragas_result = evaluate(
				ragas_dataset,
				metrics=[answer_correctness],
				embeddings=RAGAS_EMBEDDINGS,
				llm=RAGAS_LLM,
				show_progress=True,
			)
		except Exception as eval_error:
			print(f"Warning: RAGAS evaluate() failed: {eval_error}")
			import traceback
			traceback.print_exc()
			return {"answer_correctness": None}

		try:
			result_df = ragas_result.to_pandas()
		except Exception as to_pandas_error:
			print(f"Warning: RAGAS to_pandas() failed: {to_pandas_error}")
			return {"answer_correctness": None}

		if result_df is None or result_df.empty or len(result_df) == 0:
			print("Warning: RAGAS returned empty or None result DataFrame")
			return {"answer_correctness": None}

		try:
			scores = result_df.iloc[0].to_dict()
		except (IndexError, KeyError) as idx_error:
			print(f"Warning: Could not extract RAGAS scores from result: {idx_error}. DataFrame shape: {result_df.shape}, columns: {list(result_df.columns)}")
			return {"answer_correctness": None}

		return {
			"answer_correctness": round(float(scores.get("answer_correctness", 0)), 6) if scores.get("answer_correctness") is not None else None,
		}
	except Exception as e:
		print(f"Warning: RAGAS metric computation failed (outer exception): {e}")
		import traceback
		traceback.print_exc()
		return {"answer_correctness": None}

@traceable(name="evaluate_sequence_pair", run_type="tool", tags=["evaluation", "sequence"])
def evaluate_sequence_pair(
	ground_truth_csv: Path,
	prediction_csv: Path,
	output_csv: Path,
	enable_rule_based_scoring: bool,
) -> pd.DataFrame:
	gt_bundle = load_sequence_csv(ground_truth_csv)
	pred_bundle = load_sequence_csv(prediction_csv)

	gt_records: list[SequenceRecord] = gt_bundle["records"]
	pred_records: list[SequenceRecord] = pred_bundle["records"]
	gt_blocks: list[PhaseBlock] = gt_bundle["phase_blocks"]
	pred_blocks: list[PhaseBlock] = pred_bundle["phase_blocks"]

	pred_block_map = {block.phase_id: block for block in pred_blocks}

	rows: list[dict] = []
	for gt_block in gt_blocks:
		rows.append(score_phase_block(gt_block, pred_block_map.get(gt_block.phase_id), enable_rule_based_scoring))

	gt_tokens = _sequence_to_tokens(gt_records)
	pred_tokens = _sequence_to_tokens(pred_records)
	global_lcs_length = _lcs_length(gt_tokens, pred_tokens)
	global_lcs_score = global_lcs_length / len(gt_tokens) if gt_tokens else 1.0

	gt_transitions = list(zip(gt_tokens, gt_tokens[1:]))
	pred_transitions = list(zip(pred_tokens, pred_tokens[1:]))
	matched_transitions = _multiset_overlap(gt_transitions, pred_transitions)
	transition_accuracy = matched_transitions / len(gt_transitions) if gt_transitions else 1.0
	transition_precision = matched_transitions / len(pred_transitions) if pred_transitions else (1.0 if not gt_transitions else 0.0)
	transition_f1 = (
		2 * transition_precision * transition_accuracy / (transition_precision + transition_accuracy)
		if transition_precision + transition_accuracy
		else 0.0
	)

	total_gt_steps = max(len(gt_records), 1)
	phase_level_score = sum(
		row["phase_level_score"] * row["gt_count"] for row in rows
	) / total_gt_steps if rows else 0.0

	rule_summary = score_rule_catalog(gt_blocks, pred_blocks) if enable_rule_based_scoring else {
		"rule_based_score": None,
		"rule_details": [],
	}

	ragas_metrics = compute_ragas_answer_metrics(gt_records, pred_records)

	test_name = gt_bundle["metadata"].get("TEST NAME") or pred_bundle["metadata"].get("TEST NAME") or ""
	objective = gt_bundle["metadata"].get("OBJECTIVE") or pred_bundle["metadata"].get("OBJECTIVE") or ""
	test_purpose = gt_bundle["metadata"].get("TEST PURPOSE") or pred_bundle["metadata"].get("TEST PURPOSE") or ""

	summary_row = {
		"scope": "overall",
		"phase_id": "overall",
		"category": "ALL",
		"test_name": test_name,
		"objective": objective,
		"test_purpose": test_purpose,
		"ground_truth_file": str(ground_truth_csv),
		"prediction_file": str(prediction_csv),
		"gt_count": len(gt_records),
		"pred_count": len(pred_records),
		"phase_level_score": round(phase_level_score, 6),
		"lcs_score": round(global_lcs_score, 6),
		"transition_accuracy": round(transition_accuracy, 6),
		"transition_precision": round(transition_precision, 6),
		"transition_recall": round(transition_accuracy, 6),
		"transition_f1": round(transition_f1, 6),
		"answer_correctness": ragas_metrics.get("answer_correctness"),
		"missing_transitions": _json_dump([]),
		"unexpected_transitions": _json_dump([]),
		"ground_truth_list": _json_dump([record.to_export_dict() for record in gt_records]),
		"prediction_list": _json_dump([record.to_export_dict() for record in pred_records]),
	}

	for row in rows:
		row.setdefault("test_name", test_name)
		row.setdefault("objective", objective)
		row.setdefault("test_purpose", test_purpose)
		row.setdefault("ground_truth_file", str(ground_truth_csv))
		row.setdefault("prediction_file", str(prediction_csv))

	frame = pd.DataFrame([summary_row, *rows])
	output_csv.parent.mkdir(parents=True, exist_ok=True)
	frame.to_csv(output_csv, index=False)
	return frame


def _resolve_configured_path(base_dir: Path, configured_name: str, cli_value: Optional[str]) -> Path:
	raw_value = str(cli_value or configured_name or "").strip()
	if not raw_value:
		raise ValueError(
			f"Missing file selection for {base_dir.name}. Set the filename variable in evaluation_langgraph.py or pass it with CLI."
		)
	candidate = Path(raw_value)
	return candidate if candidate.is_absolute() else (base_dir / candidate).resolve()


def _build_default_results_path(ground_truth_csv: Path, prediction_csv: Path) -> Path:
	file_name = f"{_clean_filename(ground_truth_csv.stem)}__vs__{_clean_filename(prediction_csv.stem)}_scores.csv"
	return RESULTS_DIR / file_name


@traceable(name="Evaluation_Sequence", run_type="chain", tags=["evaluation", "langgraph", "csv"])
def run_sequence_evaluation(
	ground_truth_csv: Path,
	prediction_csv: Path,
	output_csv: Optional[Path] = None,
	enable_rule_based_scoring: bool = ENABLE_RULE_BASED_SCORING,
) -> pd.DataFrame:
	if not ground_truth_csv.exists():
		raise FileNotFoundError(f"Ground truth CSV not found: {ground_truth_csv}")
	if not prediction_csv.exists():
		raise FileNotFoundError(f"Prediction CSV not found: {prediction_csv}")

	destination = output_csv or _build_default_results_path(ground_truth_csv, prediction_csv)
	return evaluate_sequence_pair(
		ground_truth_csv=ground_truth_csv,
		prediction_csv=prediction_csv,
		output_csv=destination,
		enable_rule_based_scoring=enable_rule_based_scoring,
	)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Evaluate LangGraph sequence CSV output against a ground-truth CSV with LangSmith tracing."
	)
	parser.add_argument(
		"--ground-truth-csv",
		help="Filename inside Evaluation/ground_truths or an absolute path.",
	)
	parser.add_argument(
		"--prediction-csv",
		help="Filename inside Evaluation/outputs or an absolute path.",
	)
	parser.add_argument(
		"--output-csv",
		help="Destination CSV path. Defaults to Evaluation/results/<gt>__vs__<pred>_scores.csv.",
	)
	parser.add_argument(
		"--disable-rule-based-scoring",
		action="store_true",
		help="Disable the optional mandatory-sequence rule score.",
	)
	args = parser.parse_args()

	ground_truth_csv = _resolve_configured_path(GROUND_TRUTHS_DIR, GROUND_TRUTH_FILENAME, args.ground_truth_csv)
	prediction_csv = _resolve_configured_path(OUTPUTS_DIR, PREDICTION_FILENAME, args.prediction_csv)
	output_csv = Path(args.output_csv).resolve() if args.output_csv else _build_default_results_path(ground_truth_csv, prediction_csv)

	frame = run_sequence_evaluation(
		ground_truth_csv=ground_truth_csv,
		prediction_csv=prediction_csv,
		output_csv=output_csv,
		enable_rule_based_scoring=not args.disable_rule_based_scoring,
	)

	overall = frame.iloc[0].to_dict() if not frame.empty else {}
	print("\nSequence Evaluation Summary")
	print(f"Ground truth: {ground_truth_csv}")
	print(f"Prediction:   {prediction_csv}")
	print(f"Results:      {output_csv}")
	if not LANGSMITH_TRACING_ENABLED:
		print("LangSmith tracing: disabled (set LANGCHAIN_API_KEY or LANGSMITH_API_KEY to enable remote traces)")
	if overall:
		print(f"\n--- Structural Metrics ---")
		print(f"Phase-Level Score:  {overall.get('phase_level_score')}")
		print(f"LCS Score:          {overall.get('lcs_score')}")
		print(f"Transition Accuracy:{overall.get('transition_accuracy')}")
		print(f"\n--- Answer Metrics (RAGAS) ---")
		# print(f"Answer Relevancy:   {overall.get('answer_relevancy')}")
		print(f"Answer Correctness: {overall.get('answer_correctness')}")
		# print(f"Answer Accuracy:    {overall.get('answer_accuracy')}")


if __name__ == "__main__":
	main()
