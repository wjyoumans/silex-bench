"""Command-line interface for comparative campaigns."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

from .campaign import (
    CampaignPlan,
    build_plan,
    default_run_dir,
    doctor,
    prepare_silex,
    run_campaign,
)
from .configuration import RunOverrides, load_profile, load_suite, load_tools
from .contracts import Observation
from .ledger import RunLedger
from .registry import builtin_registry
from .reporting import generate_report
from .presentation import render_campaign, render_doctor, render_list, render_plan
from .resources import builtin_names, resolve_path


def bench_root() -> Path:
    """Return the invocation root used for runs and conventional tool discovery."""

    return Path.cwd().absolute()


class _CliArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def _named_path(root: Path, kind: str, value: str) -> Path:
    del root
    return resolve_path(kind, value)


def _metric(values: list[str] | None, option: str) -> tuple[tuple[str, float], ...]:
    result: list[tuple[str, float]] = []
    for value in values or []:
        name, separator, raw = value.partition("=")
        if not separator or not name:
            raise ValueError(f"{option} values must use NAME=VALUE")
        try:
            number = float(raw)
        except ValueError as exc:
            raise ValueError(f"{option} value is not numeric: {value}") from exc
        result.append((name, number))
    return tuple(result)


def _overrides(args: argparse.Namespace) -> RunOverrides:
    return RunOverrides(
        workloads=tuple(dict.fromkeys(getattr(args, "workload", None) or [])),
        backends=tuple(dict.fromkeys(getattr(args, "backend", None) or [])),
        case_ids=tuple(dict.fromkeys(getattr(args, "case", None) or [])),
        tags=tuple(dict.fromkeys(getattr(args, "tag", None) or [])),
        metric_minima=_metric(getattr(args, "metric_min", None), "--metric-min"),
        metric_maxima=_metric(getattr(args, "metric_max", None), "--metric-max"),
        repetitions=getattr(args, "repetitions", None),
        timeout_seconds=getattr(args, "timeout", None),
        budget_seconds=getattr(args, "budget", None),
        cpu=getattr(args, "cpu", None),
        required_pairs=_required_pairs(getattr(args, "require_pair", None)),
        require_all_adapters=getattr(args, "require_all_adapters", False),
    )


def _required_pairs(values: list[str] | None) -> tuple[tuple[str, str], ...]:
    result: list[tuple[str, str]] = []
    for value in values or []:
        candidate, separator, baseline = value.partition(":")
        if not separator or not candidate or not baseline or ":" in baseline:
            raise ValueError("--require-pair values must use CANDIDATE:BASELINE")
        result.append((candidate, baseline))
    return tuple(result)


def _selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workload", action="append", help="select a workload ID")
    parser.add_argument("--backend", action="append", help="select a backend ID")
    parser.add_argument("--case", action="append", help="select a corpus case ID")
    parser.add_argument(
        "--tag",
        action="append",
        help="require an additional corpus tag, narrowing the selected profile",
    )
    parser.add_argument("--metric-min", action="append", metavar="NAME=VALUE")
    parser.add_argument("--metric-max", action="append", metavar="NAME=VALUE")
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--timeout", type=float, help="per-observation timeout in seconds")
    parser.add_argument("--budget", type=float, help="whole-campaign wall-clock budget")
    parser.add_argument("--cpu", type=int, help="Linux CPU affinity")
    parser.add_argument(
        "--require-pair",
        action="append",
        metavar="CANDIDATE:BASELINE",
        help="replace the suite's required comparison pair (repeatable)",
    )
    parser.add_argument(
        "--require-all-adapters",
        action="store_true",
        help="fail unless every selected, declared adapter capability validates",
    )


def _campaign_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--suite", default="number-field", help="suite name or TOML path")
    parser.add_argument("--profile", default="quick", help="profile name or TOML path")
    parser.add_argument("--tools", type=Path, help="untracked local tool-path TOML")
    _selection_arguments(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = _CliArgumentParser(prog="silex-bench")
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit one machine-readable JSON document instead of human output",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="list workloads, backends, suites, and profiles")

    for name, help_text in (
        ("plan", "materialize and print a campaign without probing tools"),
        ("doctor", "probe selected engines and required comparison pairs"),
        ("check", "run one correctness observation per selected case and engine"),
        ("run", "run a correctness-gated performance campaign"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        _campaign_arguments(command)
        if name == "doctor":
            command.add_argument("--build-silex", action="store_true")
        if name in {"check", "run"}:
            command.add_argument("--run-dir", type=Path)
            command.add_argument("--build-silex", action="store_true")

    resume = subparsers.add_parser("resume", help="continue an interrupted run")
    resume.add_argument("run_dir", type=Path)
    resume.add_argument("--tools", type=Path, help="the same local tool-path TOML used originally")

    report = subparsers.add_parser("report", help="generate an exploratory immutable report")
    report.add_argument("run_dir", type=Path)

    export = subparsers.add_parser("export", help="generate a publication-gated immutable bundle")
    export.add_argument("run_dir", type=Path)
    export.add_argument("--publication", action="store_true", required=True)
    return parser


def _load_plan(
    args: argparse.Namespace, *, performance: bool, root: Path | None = None
) -> CampaignPlan:
    root = bench_root() if root is None else root.expanduser().absolute()
    suite = load_suite(_named_path(root, "suites", args.suite))
    profile = load_profile(_named_path(root, "profiles", args.profile))
    local_tools = root / ".silex-bench.local.toml"
    tools = load_tools(args.tools or (local_tools if local_tools.is_file() else None))
    return build_plan(
        root,
        suite,
        profile,
        tools,
        _overrides(args),
        builtin_registry(),
        performance=performance,
    )


def _resume_plan(run_dir: Path, tools_path: Path | None) -> CampaignPlan:
    with RunLedger(run_dir.expanduser().absolute()) as ledger:
        manifest = ledger.manifest()
    plan = manifest.get("plan", {})
    overrides = plan.get("overrides", {})
    stored_tools = tools_path or plan.get("tools_path")
    namespace = argparse.Namespace(
        suite=plan["suite_path"],
        profile=plan["profile_path"],
        tools=None if stored_tools is None else Path(stored_tools),
        workload=overrides.get("workloads", []),
        backend=overrides.get("backends", []),
        case=overrides.get("case_ids", []),
        tag=overrides.get("tags", []),
        metric_min=[f"{key}={value}" for key, value in overrides.get("metric_minima", {}).items()],
        metric_max=[f"{key}={value}" for key, value in overrides.get("metric_maxima", {}).items()],
        repetitions=overrides.get("repetitions"),
        timeout=overrides.get("timeout_seconds"),
        budget=overrides.get("budget_seconds"),
        cpu=overrides.get("cpu"),
        require_pair=[":".join(pair) for pair in overrides.get("required_pairs", [])],
        require_all_adapters=overrides.get("require_all_adapters", False),
    )
    return _load_plan(
        namespace,
        performance=plan.get("mode", "performance") == "performance",
        root=Path(plan["invocation_root"]),
    )


def _listed(root: Path) -> dict[str, Any]:
    del root
    registry = builtin_registry()
    return {
        **registry.describe(),
        "suites": list(builtin_names("suites")),
        "profiles": list(builtin_names("profiles")),
    }


def _normalized_argv(argv: list[str] | None) -> list[str]:
    values = list(sys.argv[1:] if argv is None else argv)
    count = values.count("--json")
    if count > 1:
        raise ValueError("--json may be supplied only once")
    if count == 1:
        values.remove("--json")
        values.insert(0, "--json")
    return values


def _report_details(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    directory = Path(payload["directory"])
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    bundle = json.loads((directory / "bundle.json").read_text(encoding="utf-8"))
    plots = bundle.get("plots", {})
    details = {
        "directory": str(directory),
        "identity": payload["report_identity"],
        "markdown": str(directory / "report.md"),
        "plots": [str(directory / value) for value in plots.get("files", [])],
        "plot_reason": None if plots.get("generated") is True else plots.get("reason"),
        "publication_eligible": payload.get("publication_eligible") is True,
        "publication_errors": payload.get("publication_errors", []),
    }
    return details, summary


def _campaign_payload(
    snapshot: dict[str, Any], run_dir: Path, report_payload: dict[str, Any]
) -> dict[str, Any]:
    report, summary = _report_details(report_payload)
    return {
        "schema_version": snapshot["schema_version"],
        "success": snapshot["state"] == "complete",
        "state": snapshot["state"],
        "run_dir": str(run_dir.expanduser().absolute()),
        "observations": len(snapshot["observations"]),
        "agreements": len(snapshot["agreements"]),
        "backend_status": summary.get("backend_status", []),
        "agreement_status": summary.get("agreement_status", []),
        "aggregate_ratios": summary.get("aggregate_ratios", []),
        "failures": summary.get("failures", []),
        "report": report,
    }


def _human_output(command: str, payload: dict[str, Any]) -> str:
    if command == "list":
        return render_list(payload)
    if command == "plan":
        return render_plan(payload)
    if command == "doctor":
        return render_doctor(payload)
    if command in {"check", "run", "resume"}:
        return render_campaign(payload)
    if command in {"report", "export"}:
        report = payload["report"]
        markdown = Path(report["markdown"]).read_text(encoding="utf-8").rstrip()
        return f"{markdown}\n\nArtifacts: {report['directory']}"
    raise ValueError(f"no human renderer for command: {command}")


def _run_with_progress(
    plan: CampaignPlan, run_dir: Path, *, resume: bool, enabled: bool
) -> dict[str, Any]:
    terminal = enabled and sys.stdout.isatty()

    def update(observation: Observation, completed: int, total: int) -> None:
        label = f"{observation.case_key}/{observation.backend}"
        sys.stdout.write(
            f"\r\x1b[2K[{completed}/{total}] {label}: {observation.status.value}"
        )
        sys.stdout.flush()

    try:
        return run_campaign(
            plan,
            run_dir,
            resume=resume,
            progress=update if terminal else None,
        )
    finally:
        if terminal:
            sys.stdout.write("\n")
            sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    json_requested = "--json" in raw_argv
    try:
        normalized = _normalized_argv(raw_argv)
        args = parser.parse_args(normalized)
    except ValueError as exc:
        if json_requested:
            print(
                json.dumps(
                    {
                        "success": False,
                        "error": {"kind": type(exc).__name__, "message": str(exc)},
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(f"silex-bench: {exc}", file=sys.stderr)
        return 2
    try:
        if args.command == "list":
            payload = _listed(bench_root())
        elif args.command == "report":
            report_payload = generate_report(args.run_dir)
            report, _ = _report_details(report_payload)
            payload = {**report_payload, "report": report}
        elif args.command == "export":
            report_payload = generate_report(args.run_dir, publication=True)
            report, _ = _report_details(report_payload)
            payload = {**report_payload, "report": report}
        elif args.command == "resume":
            plan = _resume_plan(args.run_dir, args.tools)
            snapshot = _run_with_progress(
                plan, args.run_dir, resume=True, enabled=not args.json
            )
            payload = _campaign_payload(
                snapshot, args.run_dir, generate_report(args.run_dir)
            )
        else:
            plan = _load_plan(args, performance=args.command != "check")
            if args.command == "plan":
                payload = plan.to_json()
            elif args.command == "doctor":
                if args.build_silex:
                    prepare_silex(plan)
                payload = doctor(plan)
            elif args.command in {"check", "run"}:
                if args.build_silex:
                    prepare_silex(plan)
                run_dir = args.run_dir or default_run_dir(plan)
                snapshot = _run_with_progress(
                    plan, run_dir, resume=False, enabled=not args.json
                )
                payload = _campaign_payload(
                    snapshot, run_dir, generate_report(run_dir)
                )
            else:  # pragma: no cover - argparse owns the command domain.
                parser.error("unknown command")
                return 2
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(_human_output(args.command, payload))
        if args.command in {"doctor", "check", "run", "resume"}:
            state = payload.get("state")
            return 0 if payload.get("success") is True or state == "complete" else 1
        return 0
    except (
        KeyError,
        OSError,
        RuntimeError,
        ValueError,
        json.JSONDecodeError,
        sqlite3.Error,
    ) as exc:
        if args.json:
            print(
                json.dumps(
                    {
                        "success": False,
                        "error": {"kind": type(exc).__name__, "message": str(exc)},
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(f"silex-bench: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
