"""Jev-Style-2B-Decision-v3: typed decisions with transformers / PyTorch (CUDA, MPS, CPU).

Self-contained runtime for chaoliangUNSW/Jev-Style-2B-Decision-v3 (Apache-2.0). No dependency on any training
code: rendering (render v2 + attention blocks), block-causal attention, verdict readout and calibration are
implemented below and reproduce the reference implementation used for evaluation (see release_config.json ->
"runtime_parity"). Needs torch, transformers (with Qwen3.5 support), tokenizers and numpy.

    python jev_style_decision.py --state "..." --question "Which option?" --options '["a", "b"]'

    from jev_style_decision import JevStyleDecision
    m = JevStyleDecision(".")                                          # float32, best available device
    m.decide(state, "Is the task finished?")                           # true/false question
    m.decide_many(state, [q1, q2, q3])                                 # one state, computed once

Probabilities use the ONE calibrated global temperature of readout_config.json (temperatures.global
= 0.828) unless temperature=... is given (1.0 = uncalibrated scores). ``category=...`` is accepted
for compatibility with the 0.8B v3 runtime and ignored: this model has no category temperatures.

NOTE: this model uses a different input protocol (render v2 + block attention) from the 0.8B v3
models; the 0.8B runtimes must not be used with these weights (they would run ordinary causal
attention over a different layout and give wrong answers).
"""
# ----------------------------------------------------------------------------------------------
# Shared core (byte-identical in jev_style_decision.py, jev_style_decision_gguf.py and jev_style_decision_mlx.py):
# input rendering (v2), verdict readout, calibration, budgets, errors, manifest check, CLI/JSONL.
#
# Input layout ("macjev-render-v2-long-options", layout "sb"; token segments are encoded separately
# and concatenated, so slot positions are exact):
#
#   state prefix     State:\n<state>\n\n
#   short form       Question [<type>]: <question>\nOptions:\n
#   (question +      - <option 1> ->\n ... - <option K> ->\n
#    options + slots <= 2,048 tokens)
#   overflow form    Question [<type>]: <question>\nOptions:\n
#   (otherwise)      Option 1: <option 1>\n ... Option K: <option K>\n            (the catalogue)
#                    Judge each numbered option in the complete catalogue above:\n
#                    Option 1 ->\n ... Option K ->\n                               (the rubric)
#
# Attention blocks ([start, stop) token ranges): the state is cut into consecutive 2,048-token
# blocks; the short form is one block; in the overflow form the catalogue and the rubric are each cut
# into 2,048-token blocks. In the 6 full-attention layers every block attends to all earlier tokens
# and to itself with NO causal mask inside the block (block-causal); the Gated-DeltaNet layers are
# ordinary recurrent layers. A backend must compute exactly these blocks (never merged, never re-split).
#
# Score of option k = logit(" yes") - logit(" no") at its " ->" slot = h_slot . (W_yes - W_no) in
# float32 (final normed hidden state, tied embedding rows). Probabilities = softmax(scores / T) in
# canonical option order. T = the ONE global temperature of readout_config.json
# (temperatures.global, fitted on 2,000 calibration rows) unless temperature=... overrides it
# (1.0 = uncalibrated scores). This model has no category / group temperatures and no separate
# question/options budget, so the 0.8B v3 runtime's --category and --head-max options do not exist here.
#
# Budget: the complete input (state + question + options + readout) <= 25,600 tokens. Larger inputs
# raise InputBudgetError; nothing is ever truncated. Text inside the state, question and options is
# tokenised with special tokens disabled, so e.g. "<|im_end|>" in user text can never act as a
# control token. A non-finite score (NaN / inf) raises NonFiniteScoreError: no probabilities are made.
# ----------------------------------------------------------------------------------------------
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

MODEL_NAME = "Jev-Style-2B-Decision-v3"
TEMPLATE_VERSION = "macjev-render-v2-long-options"
READOUT_FORMAT = "macjev-readout-v2"
LAYOUT = "sb"
BLOCK = 2048                    # attention block size (a processing unit, not a content limit)
CONTEXT_LIMIT = 25_600          # state + question + options + readout, all included
QTYPES = ("choice", "score", "noul")
JSONL_GROUP_MAX = 256           # consecutive JSONL rows with one state scored in one backend call
HERE = Path(__file__).resolve().parent


class InputBudgetError(ValueError):
    """The rendered input exceeds the token budget. Nothing was truncated."""


class QuestionError(ValueError):
    """The question/options are malformed."""


class NonFiniteScoreError(FloatingPointError):
    """The model produced a non-finite decision score (NaN or inf). No probabilities are returned."""


# -- questions ------------------------------------------------------------------------------------
def option_names(question):
    """Canonical option identifiers, in the order the probabilities are returned."""
    if not isinstance(question, dict):
        raise QuestionError("question must be a dict {'t', 'ins', 'crit'}")
    t, crit = question.get("t"), question.get("crit")
    if not isinstance(question.get("ins"), str) or not question["ins"].strip():
        raise QuestionError("question text ('ins') must be a non-empty string")
    if t == "choice":
        if not isinstance(crit, dict) or not crit:
            raise QuestionError("choice needs a non-empty dict {option name: description or None}")
        return [str(k) for k in crit]
    if t == "score":
        if not isinstance(crit, list) or not 2 <= len(crit) <= 10:
            raise QuestionError("score needs a list of 2..10 level descriptions")
        return [str(i) for i in range(len(crit))]
    if t == "noul":
        if crit is not None and not isinstance(crit, dict):
            raise QuestionError("noul criteria must be None or {'false': ..., 'true': ...}")
        return ["false", "true"]
    raise QuestionError(f"unknown question type {t!r} (expected one of {QTYPES})")


def make_question(question, options=None, qtype=None):
    """Build a typed question.

    * ``question`` already a dict {"t", "ins", "crit"}: validated and returned.
    * ``qtype="choice"`` (default when ``options`` is given): ``options`` = {name: description or None}
      or a list of names.
    * ``qtype="score"``: ``options`` = list of 2..10 level descriptions (level 0 first).
    * ``qtype="noul"`` (default when no options): a true/false statement; ``options`` may be
      {"false": "...", "true": "..."} to describe the two outcomes.
    """
    if isinstance(question, dict):
        q = dict(question)
    else:
        if qtype is None:
            qtype = "choice" if options is not None else "noul"
        if qtype == "choice":
            if isinstance(options, (list, tuple)):
                if len(set(map(str, options))) != len(options):
                    raise QuestionError("duplicate option names")
                crit = {str(o): None for o in options}
            else:
                crit = options
        elif qtype == "score":
            crit = list(options) if options is not None else None
        else:
            crit = options
        q = {"t": qtype, "ins": question, "crit": crit}
    option_names(q)
    return q


def serialize_state(state):
    """Strings pass through unchanged; any other JSON value is serialised (ensure_ascii=False)."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def _criterion(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def render_options(question):
    t, crit = question["t"], question.get("crit")
    if t == "choice":
        return [k if v is None or v == "" else f"{k}: {_criterion(v)}" for k, v in crit.items()]
    if t == "score":
        return [f"level {i}: {_criterion(c)}" for i, c in enumerate(crit)]
    crit = crit or {}
    false_c, true_c = crit.get("false"), crit.get("true")
    return ["false: " + (_criterion(false_c) if false_c not in (None, "") else "no, the statement does not hold"),
            "true: " + (_criterion(true_c) if true_c not in (None, "") else "yes, the statement holds")]


# -- tokenizer + renderer -----------------------------------------------------------------------
class TextEncoder:
    """HF ``tokenizers`` tokenizer.json; no BOS/EOS, special tokens in text are split (never control tokens)."""

    def __init__(self, tokenizer_json):
        from tokenizers import Tokenizer
        self.tk = Tokenizer.from_file(str(tokenizer_json))
        self.tk.encode_special_tokens = True

    def __call__(self, text):
        return self.tk.encode(text, add_special_tokens=False).ids

    def id_to_token(self, i):
        return self.tk.id_to_token(int(i))

    def vocab_size(self):
        return self.tk.get_vocab_size(with_added_tokens=True)


def _blocks(start, stop):
    return [(s, min(s + BLOCK, stop)) for s in range(start, stop, BLOCK)]


class Rendered:
    """One rendered question: token ids, the verdict slots (one per option, canonical order), the
    attention blocks ([start, stop) pairs tiling [0, len(ids))) and the state prefix length."""
    __slots__ = ("ids", "prefix_len", "slots", "blocks", "names", "qtype", "catalogue_overflow")

    def __init__(self, ids, prefix_len, slots, blocks, names, qtype, catalogue_overflow):
        self.ids, self.prefix_len, self.slots, self.blocks = ids, prefix_len, slots, blocks
        self.names, self.qtype, self.catalogue_overflow = names, qtype, catalogue_overflow

    @property
    def state_blocks(self):
        return [b for b in self.blocks if b[1] <= self.prefix_len]

    @property
    def question_blocks(self):
        return [b for b in self.blocks if b[0] >= self.prefix_len]

    @property
    def input_tokens(self):
        return len(self.ids)

    @property
    def head_tokens(self):
        """question + options + readout tokens (everything after the state prefix)"""
        return len(self.ids) - self.prefix_len


class Renderer:
    def __init__(self, encode, readout_cfg, max_len=CONTEXT_LIMIT):
        want = {"format": READOUT_FORMAT, "template": TEMPLATE_VERSION, "layout": LAYOUT, "readout": "verdict",
                "block_size": BLOCK, "total_context_limit": CONTEXT_LIMIT}
        bad = {k: readout_cfg.get(k) for k, v in want.items() if readout_cfg.get(k) != v}
        if bad:
            raise ValueError(f"readout_config.json does not describe this runtime's protocol {want}; got {bad}")
        if not 0 < int(max_len) <= CONTEXT_LIMIT:
            raise ValueError(f"max_len must be in 1..{CONTEXT_LIMIT}")
        self.enc, self.max_len = encode, int(max_len)
        self.head_max = self.max_len    # no separate question/options cap (field kept for the 0.8B / jev-style API)
        st = readout_cfg["slot_tokens"]
        self.yes, self.no, arrow = int(st["yes"]["id"]), int(st["no"]["id"]), int(st["verdict_slot"]["id"])
        for text, want_id in ((" yes", self.yes), (" no", self.no), (" ->", arrow)):
            got = self.enc(text)
            if got != [want_id]:
                raise ValueError(f"tokenizer mismatch: {text!r} -> {got}, readout_config expects [{want_id}]")
        self.arrow = [arrow]
        self.newline = self.enc("\n")
        self.dash = self.enc("- ")
        self.rubric_head = self.enc("Judge each numbered option in the complete catalogue above:\n")
        self._numbered = {}

    def _option_label(self, pos, catalogue):
        key = (pos, catalogue)
        if key not in self._numbered:
            self._numbered[key] = self.enc(f"Option {pos + 1}: " if catalogue else f"Option {pos + 1}")
        return self._numbered[key]

    def prefix_ids(self, state):
        return self.enc("State:\n") + self.enc(serialize_state(state)) + self.enc("\n\n")

    def pieces(self, state, question, prefix=None):
        """Tokenised segments of one question (``prefix``: already tokenised state, to tokenise it once)."""
        names = option_names(question)
        return {"prefix": self.prefix_ids(state) if prefix is None else list(prefix),
                "head": self.enc(f"Question [{question['t']}]: {question['ins']}\nOptions:\n"),
                "opts": [self.enc(o) for o in render_options(question)], "names": names, "qtype": question["t"]}

    def assemble(self, pieces, max_len=None):
        maximum = self.max_len if max_len is None else min(self.max_len, int(max_len))
        prefix, head, opts = list(pieces["prefix"]), list(pieces["head"]), pieces["opts"]
        k = len(opts)
        if k < 1:
            raise QuestionError("at least one option is required")
        short, rel = list(head), []
        for o in opts:
            short += self.dash + list(o) + self.arrow
            rel.append(len(short) - 1)
            short += self.newline
        overflow = len(short) > BLOCK
        if not overflow:
            ids = prefix + short
            slots = [len(prefix) + s for s in rel]
            blocks = _blocks(0, len(prefix)) + [(len(prefix), len(ids))]
        else:
            catalogue = list(head)
            for pos, o in enumerate(opts):
                catalogue += self._option_label(pos, True) + list(o) + self.newline
            doc_end = len(prefix) + len(catalogue)
            rubric, rel = list(self.rubric_head), []
            for pos in range(k):
                rubric += self._option_label(pos, False) + self.arrow
                rel.append(len(rubric) - 1)
                rubric += self.newline
            ids = prefix + catalogue + rubric
            slots = [doc_end + s for s in rel]
            blocks = _blocks(0, len(prefix)) + _blocks(len(prefix), doc_end) + _blocks(doc_end, len(ids))
        if len(ids) > maximum:
            raise InputBudgetError(f"the complete input needs {len(ids)} tokens (state {len(prefix)} + question/"
                                   f"options/readout {len(ids) - len(prefix)}); the limit is {maximum} tokens (model "
                                   f"maximum {CONTEXT_LIMIT}). Nothing was truncated: shorten the state, the "
                                   f"question or the options.")
        return Rendered(ids, len(prefix), slots, blocks, list(pieces["names"]), pieces["qtype"], overflow)

    def render(self, state, question, max_len=None, prefix=None):
        return self.assemble(self.pieces(state, question, prefix), max_len)


# -- calibration ----------------------------------------------------------------------------------
def check_temperature(t):
    try:
        t = float(t)
    except (TypeError, ValueError):
        raise ValueError(f"temperature must be a number, got {t!r}") from None
    if not (math.isfinite(t) and t > 0):
        raise ValueError(f"temperature must be finite and > 0, got {t!r}")
    return t


def softmax_probabilities(scores, temperature):
    """softmax(scores / T) in float64 (the reference computation)."""
    z = np.asarray(scores, float) / float(temperature)
    z = np.exp(z - z.max())
    return z / z.sum()


def concentration(p):
    k = len(p)
    if k < 2:
        return 1.0
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
    return float(np.clip(1.0 - ent / math.log(k), 0.0, 1.0))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


NOT_VERIFIED = ("README.md", "eval_results.json")          # documentation / records: in manifest.json, not checked
NOT_VERIFIED_DIRS = ("assets/", "figures/", "validation/")


def verify_manifest(model_dir, only=None):
    """Re-hash the files listed in manifest.json (all, or those whose path starts with one of ``only``).
    Documentation and evaluation records (README.md, eval_results.json, assets/, figures/, validation/) are
    recorded in the manifest but not checked here (not even when named in ``only``), so a card edit never
    makes the runtime refuse to load and a download without them still verifies."""
    model_dir = Path(model_dir)
    man = json.loads((model_dir / "manifest.json").read_text())
    bad, missing, checked = [], [], 0
    for name, rec in man["files"].items():
        if name in NOT_VERIFIED or name.startswith(NOT_VERIFIED_DIRS):
            continue
        if only and not any(name == o or name.startswith(o.rstrip("/") + "/") for o in only):
            continue
        p = model_dir / name
        if not p.exists():
            missing.append(name)
        elif _sha256(p) != rec["sha256"]:
            bad.append(name)
        checked += 1
    return {"ok": not bad and not missing, "checked": checked, "bad": bad, "missing": missing}


class DecisionBase:
    """Backend-independent part. A backend implements ``_scores_many(rendered) -> list of list[float]``:
    ``rendered`` is a non-empty list of Rendered that all share ONE state (identical prefix ids and
    state blocks); it returns the raw float32 scores of every slot of every Rendered (slot order =
    canonical option order), computing the state blocks once per call. Backends keep the most recent
    state for the next call (exact prefix match only), so consecutive calls about the same state (e.g.
    JSONL rows read from stdin) reuse it."""
    backend = "base"

    def _setup(self, model_dir, tokenizer_json, max_len=CONTEXT_LIMIT, temperature=None):
        self.model_dir = Path(model_dir)
        self.readout_config = json.loads((self.model_dir / "readout_config.json").read_text())
        self.calibrated_temperature = check_temperature(self.readout_config["temperatures"]["global"])
        self.default_temperature = self.calibrated_temperature if temperature is None else check_temperature(temperature)
        self.encode = TextEncoder(tokenizer_json)
        self.renderer = Renderer(self.encode, self.readout_config, max_len=max_len)

    def _scores_many(self, rendered):
        raise NotImplementedError

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _score_all(self, rendered):
        """Raw scores (lists of floats, canonical order) for Rendered sharing one state; one backend call."""
        if not rendered:
            return []
        p = rendered[0].prefix_len
        prefix, sblocks = rendered[0].ids[:p], rendered[0].state_blocks
        for r in rendered[1:]:
            if r.prefix_len != p or r.ids[:p] != prefix or r.state_blocks != sblocks:
                raise ValueError("all questions of one backend call must share the same state")
        scores = self._scores_many(rendered)
        if len(scores) != len(rendered) or any(len(s) != len(r.slots) for s, r in zip(scores, rendered)):
            raise RuntimeError("backend returned a wrong number of scores")
        return [[float(x) for x in s] for s in scores]

    def score_pieces(self, pieces_list, max_len=None):
        """Raw scores for pre-tokenised questions sharing one state: dicts {"prefix", "head", "opts",
        "names", "qtype"} of token ids (see Renderer.pieces). For parity checks; no temperature."""
        return self._score_all([self.renderer.assemble(p, max_len) for p in pieces_list])

    def _result(self, r, scores, temperature):
        t = self.default_temperature if temperature is None else check_temperature(temperature)
        bad = [n for n, x in zip(r.names, scores) if not math.isfinite(x)]
        if bad:
            raise NonFiniteScoreError(f"non-finite decision scores for option(s) {bad} ({len(r.ids)} input tokens); "
                                      f"refusing to return probabilities. Check the weights file / engine build.")
        p = softmax_probabilities(scores, t)
        if not np.all(np.isfinite(p)):
            raise NonFiniteScoreError("non-finite probabilities")
        i = int(p.argmax())
        return {"answer": r.names[i], "probabilities": dict(zip(r.names, p.tolist())),
                "scores": dict(zip(r.names, scores)), "temperature": t, "top_probability": float(p[i]),
                "entropy_concentration": concentration(p), "input_tokens": len(r.ids),
                "state_tokens": r.prefix_len, "head_tokens": r.head_tokens, "blocks": len(r.blocks),
                "catalogue_overflow": r.catalogue_overflow, "model": MODEL_NAME, "backend": self.backend}

    def decide(self, state, question, options=None, qtype=None, category=None, temperature=None, head_max=None,
               max_len=None):
        """Score one question about ``state``.

        Returns {"answer", "probabilities" {option: p}, "scores" {option: logit(yes)-logit(no)},
        "temperature", "top_probability", "entropy_concentration", "input_tokens", "state_tokens",
        "head_tokens", "blocks", "catalogue_overflow", "model", "backend"}. ``temperature`` overrides
        the calibrated global temperature (1.0 = uncalibrated scores). ``category`` and ``head_max`` are
        accepted for compatibility with the 0.8B v3 runtime and the jev-style package, and ignored: this
        model has one global temperature and no separate question/options budget. Raises InputBudgetError
        (never truncates), QuestionError or NonFiniteScoreError."""
        q = make_question(question, options, qtype)
        r = self.renderer.render(state, q, max_len)
        return self._result(r, self._score_all([r])[0], temperature)

    def score_many(self, state, questions, category=None, temperature=None, head_max=None, max_len=None):
        """Several questions about ONE state: the state is tokenised and computed once and reused for
        every question (one backend call). ``questions``: dicts {"t","ins","crit"} (or anything
        make_question accepts). Results in order, identical to calling decide() per question. All
        questions are rendered and budget-checked before any scoring. ``category`` / ``head_max``: see
        decide() (accepted and ignored)."""
        qs = [make_question(q) for q in questions]
        if not qs:
            return []
        prefix = self.renderer.prefix_ids(state)
        rs = [self.renderer.render(state, q, max_len, prefix=prefix) for q in qs]
        return [self._result(r, sc, temperature) for r, sc in zip(rs, self._score_all(rs))]

    decide_many = score_many


def base_arg_parser(description):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--model-dir", default=str(HERE), help="folder with the weights and readout_config.json")
    ap.add_argument("--state", help="state as plain text")
    ap.add_argument("--state-json", help="state as a JSON value")
    ap.add_argument("--question", help="question text (or a JSON question {'t','ins','crit'})")
    ap.add_argument("--options", help="JSON: {name: description} or [names] (choice); [levels] (score)")
    ap.add_argument("--qtype", choices=QTYPES)
    ap.add_argument("--temperature", type=float,
                    help="override the calibrated global temperature of readout_config.json (1.0 = raw scores)")
    ap.add_argument("--max-len", type=int, default=CONTEXT_LIMIT,
                    help=f"total token budget (state + question + options + readout), at most {CONTEXT_LIMIT}")
    ap.add_argument("--jsonl", help="batch mode: input JSON lines {id?, state, question, options?, qtype?, "
                                    "temperature?} ('-' = stdin); one JSON result per line on stdout. Consecutive "
                                    "rows with an identical state share one state computation")
    ap.add_argument("--verify", action="store_true", help="check sha256 of the files in manifest.json first")
    return ap


def _jsonl_groups(src, streaming):
    """(line number, record or error text) grouped into runs of consecutive rows with one state. From a
    file up to JSONL_GROUP_MAX rows are grouped; from stdin every row is its own group (answered at once;
    the backend's kept state still makes consecutive identical states cheap)."""
    group, key = [], None
    for n, line in enumerate(src):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            if not isinstance(rec, dict):
                raise ValueError("a JSONL row must be a JSON object")
            k = serialize_state(rec.get("state", ""))
        except ValueError as e:
            if group:
                yield group
            group, key = [], None
            yield [(n, f"{type(e).__name__}: {e}")]
            continue
        if group and (k != key or len(group) >= JSONL_GROUP_MAX):
            yield group
            group = []
        group.append((n, rec))
        key = k
        if streaming:
            yield group
            group, key = [], None
    if group:
        yield group


def _run_group(engine, group, args):
    """Score one group of JSONL rows sharing a state. Returns (output rows, non-finite count)."""
    out, todo = {}, []
    prefix = None
    for n, rec in group:
        if isinstance(rec, str):
            out[n] = {"id": n, "error": rec}
            continue
        rid = rec.get("id", n)
        try:
            if "question" not in rec:
                raise QuestionError("row has no 'question'")
            q = make_question(rec["question"], options=rec.get("options"), qtype=rec.get("qtype"))
            t = rec.get("temperature", args.temperature)
            t = None if t is None else check_temperature(t)
            if prefix is None:
                prefix = engine.renderer.prefix_ids(rec.get("state", ""))
            todo.append((n, rid, engine.renderer.render(rec.get("state", ""), q, prefix=prefix), t))
        except (InputBudgetError, QuestionError, ValueError, TypeError, AttributeError) as e:
            out[n] = {"id": rid, "error": f"{type(e).__name__}: {e}"}
    nonfinite = 0
    if todo:
        scores = engine._score_all([r for _, _, r, _ in todo])
        for (n, rid, r, t), sc in zip(todo, scores):
            try:
                out[n] = {"id": rid, **engine._result(r, sc, t)}
            except NonFiniteScoreError as e:
                nonfinite += 1
                print(f"ERROR row {rid}: NonFiniteScoreError: {e}", file=sys.stderr, flush=True)
                out[n] = {"id": rid, "error": f"NonFiniteScoreError: {e}"}
    return [out[n] for n, _ in group], nonfinite


def run_cli(args, engine):
    """Exit status: 0 = ok (JSONL rows with input errors carry an "error" field), 2 = input error
    (single question), 3 = at least one non-finite score (refused, see stderr)."""
    if args.jsonl:
        streaming = args.jsonl == "-"
        src = sys.stdin if streaming else open(args.jsonl, encoding="utf-8")
        nonfinite = 0
        try:
            for group in _jsonl_groups(src, streaming):
                rows, bad = _run_group(engine, group, args)
                nonfinite += bad
                for row in rows:
                    print(json.dumps(row, ensure_ascii=False), flush=True)
        finally:
            if not streaming:
                src.close()
        return 3 if nonfinite else 0
    if args.question is None:
        raise SystemExit("--question (or --jsonl) is required")
    try:
        state = json.loads(args.state_json) if args.state_json is not None else (args.state or "")
        question = args.question
        if question.lstrip().startswith("{"):
            try:                                            # a JSON question {'t','ins','crit'}; else plain text
                parsed = json.loads(question)
            except ValueError:
                parsed = None
            if isinstance(parsed, dict):
                question = parsed
        options = json.loads(args.options) if args.options else None
        res = engine.decide(state, question, options=options, qtype=args.qtype, temperature=args.temperature)
    except (InputBudgetError, QuestionError, ValueError, TypeError) as e:  # NonFiniteScoreError is not a ValueError
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    except NonFiniteScoreError as e:
        print(f"ERROR: NonFiniteScoreError: {e}", file=sys.stderr)
        return 3
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0
# ---------------------------------------------------------------------------- end of shared core


# ------------------------------------------------------------------------------ PyTorch backend
# How the block-causal attention is computed exactly:
#
# The input is fed to the model ONE renderer block per forward call, in order, with a transformers
# cache: when block [a, b) runs, the cache holds tokens [0, a). In the 6 full-attention layers the
# queries are the block's own tokens and the keys/values are the cached [0, a) plus [a, b); attention
# over all of them WITHOUT any mask is then exactly "every block attends to all earlier tokens and to
# itself, no causal mask inside the block". The Gated-DeltaNet layers continue their conv / recurrent
# state from the cache, i.e. they run as ordinary causal recurrent layers over the whole input.
#
# State reuse: the state blocks are computed once per call (and kept for the next call when the next
# state is identical); every question continues from its own copy of that cache, so a question never
# sees another question's tokens. Because nothing after the state can influence the state blocks
# (block-causal), this equals recomputing state + question for each question.
import copy

ATTN_NAME = "jev_style_block_sdpa"
MPS_ATTN_CHUNK = 1024           # MPS: queries per SDPA call (row-wise identical, bounds the score matrix)
_ATTN_REGISTERED = False


def _block_sdpa(module, query, key, value, attention_mask=None, dropout=0.0, scaling=None, **kwargs):
    """Attention of ONE renderer block (see above): no mask; queries = the block, keys = all tokens so far.

    The backend announces each call's geometry on the attention module (``jev_expect`` = (block length,
    tokens so far)). Any other use, e.g. an ordinary whole-sequence ``model(...)`` call, is refused, so
    this function can never silently act as non-causal attention over a wrong span."""
    import torch
    import torch.nn.functional as F
    q_len, kv_len = query.shape[2], key.shape[2]
    expect = getattr(module, "jev_expect", None)
    mask = getattr(module, "jev_mask", None)
    if mask is not None:                    # CUDA-graph path: the whole input in one call, block-causal mask
        if expect is not None or tuple(mask.shape[-2:]) != (q_len, kv_len) or attention_mask is not None:
            raise RuntimeError("block-mask attention called with an unexpected geometry")
        rep = query.shape[1] // key.shape[1]
        if rep > 1:
            key, value = key.repeat_interleave(rep, 1), value.repeat_interleave(rep, 1)
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, scale=scaling)
        return out.transpose(1, 2).contiguous(), None
    if expect is None or tuple(expect) != (q_len, kv_len):
        raise RuntimeError(f"block attention called with {q_len} queries / {kv_len} keys, expected {expect}: this "
                           f"model must be run through JevStyleDecision (one renderer block per forward call)")
    if attention_mask is not None:
        raise RuntimeError("block attention does not take an attention mask")
    if query.shape[1] % key.shape[1]:
        raise ValueError("invalid grouped-query head counts")
    rep = query.shape[1] // key.shape[1]
    if rep > 1:
        key, value = key.repeat_interleave(rep, 1), value.repeat_interleave(rep, 1)
    chunk = getattr(module, "jev_attn_chunk", None)
    if not chunk or q_len <= chunk:
        out = F.scaled_dot_product_attention(query, key, value, is_causal=False, scale=scaling)
    else:
        out = torch.cat([F.scaled_dot_product_attention(query[:, :, s:s + chunk], key, value, is_causal=False,
                                                        scale=scaling) for s in range(0, q_len, chunk)], dim=2)
    return out.transpose(1, 2).contiguous(), None


def _no_mask(*args, **kwargs):
    return None


def _register_attention():
    global _ATTN_REGISTERED
    if not _ATTN_REGISTERED:
        from transformers import AttentionInterface
        from transformers.masking_utils import AttentionMaskInterface
        AttentionInterface.register(ATTN_NAME, _block_sdpa)
        AttentionMaskInterface.register(ATTN_NAME, _no_mask)
        _ATTN_REGISTERED = True
    return ATTN_NAME


def _pick_device(torch, device):
    if device is None:
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        return "mps" if mps is not None and mps.is_available() else "cpu"
    kind = torch.device(device).type
    if kind == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' was requested but CUDA is not available (no implicit fallback)")
    if kind == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        raise RuntimeError("device='mps' was requested but MPS is not available (no implicit fallback)")
    if kind not in ("cuda", "mps", "cpu"):
        raise ValueError(f"unsupported device {device!r} (cuda, mps or cpu)")
    return str(device)


# ----------------------------------------------------------------------- CUDA graphs (optional, CUDA only)
# The same computation as the block-by-block path above, in ONE forward call per question: the full-attention
# layers get an explicit block-causal mask (a token attends to every token of its own and all earlier blocks) and
# the Gated-DeltaNet layers run causally over the whole input, which is what the block path's cache carries over.
# One graph per padded input length is recorded once and replayed. The padding is its own last block, so no real
# token ever attends to it (and the recurrent layers are causal). Inputs longer than the largest recorded length,
# or with more than GRAPH_MAX_SLOTS options, run the block path.
GRAPH_LENGTHS = (64, 128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048, 2560, 3072,
                 3584, 4096)
GRAPH_MAX_SLOTS = 256
# The graph path re-reads the state for every question; the block path reads it once and shares it. Several questions
# about one state stay on the block path when that would re-read more than this many state tokens.
GRAPH_SHARED_STATE_MAX = 2048


class _GraphRunner:
    def __init__(self, model, attn_modules, direction, device, lengths=GRAPH_LENGTHS, max_slots=GRAPH_MAX_SLOTS):
        import torch
        self.torch, self.device, self.max_slots = torch, device, int(max_slots)
        self.graphs = {}
        pool = torch.cuda.graph_pool_handle()           # one memory pool: graphs are replayed one at a time
        with torch.inference_mode():
            for n in sorted({int(x) for x in lengths}, reverse=True):
                ids = torch.zeros((1, n), dtype=torch.long, device=device)
                blk = torch.zeros((n,), dtype=torch.long, device=device)
                slots = torch.zeros((self.max_slots,), dtype=torch.long, device=device)

                def forward(ids=ids, blk=blk, slots=slots):
                    mask = (blk[None, :] <= blk[:, None])[None, None]
                    for m in attn_modules:
                        m.jev_mask = mask
                    try:
                        h = model.model(input_ids=ids, use_cache=False).last_hidden_state
                    finally:
                        for m in attn_modules:
                            m.jev_mask = None
                    return h[0].index_select(0, slots).float() @ direction

                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(3):                      # warm-up: kernel autotuning happens outside the graph
                        forward()
                torch.cuda.current_stream().wait_stream(side)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool):
                    out = forward()
                self.graphs[n] = (graph, ids, blk, slots, out)
        self.lengths = sorted(self.graphs)

    def scores(self, r):
        """Scores of the verdict slots of one rendered question, or None when it does not fit a recorded graph."""
        if len(r.slots) > self.max_slots:
            return None
        n = next((m for m in self.lengths if m >= len(r.ids)), None)
        if n is None:
            return None
        torch = self.torch
        graph, s_ids, s_blk, s_slots, out = self.graphs[n]
        block_of = np.full(n, len(r.blocks), dtype=np.int64)        # padding = one more block after the input
        for j, (s, e) in enumerate(r.blocks):
            block_of[s:e] = j
        with torch.inference_mode():                    # the static buffers are inference tensors
            s_ids.zero_()
            s_ids[0, :len(r.ids)].copy_(torch.tensor(r.ids, dtype=torch.long))
            s_blk.copy_(torch.from_numpy(block_of))
            s_slots.zero_()
            s_slots[:len(r.slots)].copy_(torch.tensor(r.slots, dtype=torch.long))
            graph.replay()
            return out[:len(r.slots)].tolist()


class JevStyleDecision(DecisionBase):
    """Transformers / PyTorch runtime (CUDA, Apple MPS or CPU).

    >>> m = JevStyleDecision(".")                       # float32 on the best available device
    >>> m.decide({"messages": ["Refund still missing after 3 weeks"]},
    ...          "Which team should handle this ticket?",
    ...          options={"billing": "payments, refunds", "tech": "bugs, crashes", "sales": "pricing, plans"})

    device: None = cuda > mps > cpu; "cuda" / "mps" / "cpu" explicitly (never an implicit fallback).
    dtype: "float32" (default; the parity-tested setting on every device) or "bfloat16" (CUDA only).
    The readout h_slot . (w_yes - w_no) is always computed in float32.
    temperature: None = the calibrated global temperature of readout_config.json.
    threads: torch CPU threads (torch.set_num_threads; process-wide).
    attn_chunk: queries per SDPA call inside a block (default: 1,024 on MPS, whole block elsewhere).
    keep_state: keep the last state's cache for the next call with the identical state (exact match).
    cuda_graphs: True (device="cuda" only) records one CUDA graph per padded input length at start-up and replays
        it: one forward with an explicit block-causal mask, several times faster for short calls, same weights and
        readout (see release_config.json -> runtime.cuda_graphs); longer inputs keep the block path.
    category: accepted for compatibility with the 0.8B v3 runtime and ignored (one global temperature).

    decide(state, question, options=None, qtype=None, category=None, temperature=None, head_max=None, max_len=None)
    decide_many(state, questions, category=None, temperature=None, head_max=None, max_len=None)  (= score_many)
    (the same signatures as the 0.8B v3 runtime; category and head_max are accepted and ignored)
    Every result has "answer", "probabilities" (by option name; true/false questions: "false" / "true"),
    "scores", "temperature", "top_probability", "entropy_concentration", "input_tokens",
    "state_tokens", "head_tokens", "blocks", "catalogue_overflow", "model", "backend".
    """
    backend = "torch"

    def __init__(self, model_dir=HERE, device=None, dtype="float32", *, max_len=CONTEXT_LIMIT, temperature=None,
                 threads=None, attn_chunk=None, keep_state=True, verify=False, category=None, cuda_graphs=False):
        import torch
        self.torch = torch
        model_dir = Path(model_dir)
        if verify:
            res = verify_manifest(model_dir)
            if not res["ok"]:
                raise RuntimeError(f"integrity check failed: {res}")
            self.verified = res
        self._setup(model_dir, model_dir / "tokenizer.json", max_len, temperature)
        if threads:
            torch.set_num_threads(int(threads))
        self.device = _pick_device(torch, device)
        kind = torch.device(self.device).type
        dt = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        if dt not in (torch.float32, torch.bfloat16):
            raise ValueError(f"dtype must be float32 or bfloat16, got {dtype!r}")
        if dt == torch.bfloat16 and kind != "cuda":
            raise ValueError("bfloat16 is supported on CUDA only; use float32 on MPS / CPU")
        cfg = json.loads((model_dir / "config.json").read_text())
        layer_types = cfg.get("layer_types") or []
        if cfg.get("model_type") not in ("qwen3_5_text", "qwen3_5") or "full_attention" not in layer_types:
            raise ValueError(f"{model_dir / 'config.json'} is not the text-only Qwen3.5 checkpoint of {MODEL_NAME}")
        try:
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM as cls
        except ImportError as e:
            raise ImportError("this model needs a transformers version with Qwen3.5 support "
                              "(transformers.models.qwen3_5)") from e
        from transformers import DynamicCache
        self._cache_cls = DynamicCache
        name = _register_attention()
        try:
            model = cls.from_pretrained(str(model_dir), dtype=dt, attn_implementation=name)
        except TypeError:                                       # transformers 4.x keyword
            model = cls.from_pretrained(str(model_dir), torch_dtype=dt, attn_implementation=name)
        if getattr(model.config, "_attn_implementation", None) != name:
            model.set_attn_implementation(name)
        self.model = model.to(self.device).eval()
        self.dtype = dt
        types = list(self.model.config.layer_types)
        self._attn = [layer.self_attn for layer, t in zip(self.model.model.layers, types) if t == "full_attention"]
        if len(self._attn) != types.count("full_attention") or not self._attn:
            raise RuntimeError("could not find the full-attention layers")
        chunk = attn_chunk if attn_chunk is not None else (MPS_ATTN_CHUNK if kind == "mps" else None)
        for m in self._attn:
            m.jev_expect, m.jev_attn_chunk = None, (int(chunk) if chunk else None)
        w = self.model.get_output_embeddings().weight
        if not getattr(self.model.config, "tie_word_embeddings", False) and w is not self.model.get_input_embeddings().weight:
            raise ValueError("expected tied input/output embeddings")
        self.direction = (w[self.renderer.yes].float() - w[self.renderer.no].float()).detach()
        self.keep_state = bool(keep_state)
        self._kept = None                                       # (state token ids, cache after the state blocks)
        for m in self._attn:
            m.jev_mask = None
        self.cuda_graphs = None
        if cuda_graphs:
            if kind != "cuda":
                raise ValueError("cuda_graphs=True needs device='cuda'")
            self.cuda_graphs = _GraphRunner(self.model, self._attn, self.direction, self.device)

    # -- model calls
    def _block(self, ids, start, stop, cache):
        """Run renderer block [start, stop) on top of ``cache`` (which holds tokens [0, start)); returns
        the final normed hidden states of the block's tokens."""
        torch = self.torch
        x = torch.tensor([ids[start:stop]], dtype=torch.long, device=self.device)
        pos = torch.arange(start, stop, dtype=torch.long, device=self.device)[None]
        for m in self._attn:
            m.jev_expect = (stop - start, stop)
        try:
            out = self.model.model(input_ids=x, position_ids=pos, past_key_values=cache, use_cache=True)
        finally:
            for m in self._attn:
                m.jev_expect = None
        return out.last_hidden_state[0]

    def _state_cache(self, r):
        key = r.ids[:r.prefix_len]
        if self._kept is not None and self._kept[0] == key:
            return self._kept[1]
        self._kept = None                                       # free the old state first
        cache = self._cache_cls(config=self.model.config)
        for s, e in r.state_blocks:
            self._block(r.ids, s, e, cache)
        if self.keep_state:
            self._kept = (key, cache)
        return cache

    def _scores_many(self, rendered):
        if self.cuda_graphs is None or rendered[0].prefix_len * (len(rendered) - 1) > GRAPH_SHARED_STATE_MAX:
            return self._scores_many_blocks(rendered)
        out = [self.cuda_graphs.scores(r) for r in rendered]
        rest = [r for r, o in zip(rendered, out) if o is None]
        if rest:
            more = iter(self._scores_many_blocks(rest))
            out = [o if o is not None else next(more) for o in out]
        return out

    def _scores_many_blocks(self, rendered):
        torch = self.torch
        out = []
        with torch.inference_mode():
            for r in rendered:
                if r.state_blocks + r.question_blocks != r.blocks or r.slots != sorted(r.slots):
                    raise RuntimeError("renderer blocks do not tile the input")
            base = self._state_cache(rendered[0])
            for r in rendered:
                cache = copy.deepcopy(base)                     # the shared state cache itself is never modified
                hs = []
                for s, e in r.question_blocks:
                    h = self._block(r.ids, s, e, cache)
                    rel = [p - s for p in r.slots if s <= p < e]
                    if rel:
                        hs.append(h[torch.tensor(rel, device=h.device)])
                del cache
                h = torch.cat(hs).float()
                if h.shape[0] != len(r.slots):
                    raise RuntimeError("slot count mismatch")
                out.append((h @ self.direction).cpu().tolist())
        return out

    def close(self):
        self._kept = None


def main(argv=None):
    ap = base_arg_parser(f"{MODEL_NAME}: typed decisions with transformers / PyTorch")
    ap.add_argument("--device", choices=["cuda", "mps", "cpu"], help="default: cuda > mps > cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"],
                    help="bfloat16 on CUDA only; the readout is float32 either way")
    ap.add_argument("--threads", type=int, help="torch CPU threads")
    ap.add_argument("--cuda-graphs", action="store_true",
                    help="CUDA only: record CUDA graphs at start-up and replay them (much faster short inputs)")
    args = ap.parse_args(argv)
    engine = JevStyleDecision(args.model_dir, device=args.device, dtype=args.dtype, max_len=args.max_len,
                              threads=args.threads, verify=args.verify, cuda_graphs=args.cuda_graphs)
    return run_cli(args, engine)


if __name__ == "__main__":
    raise SystemExit(main())
