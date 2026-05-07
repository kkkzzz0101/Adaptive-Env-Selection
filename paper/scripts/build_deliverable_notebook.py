#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "deliverables" / "project_groupID.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip() + "\n")


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip() + "\n")


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.9",
        },
    }

    nb.cells = [
        md(
            """
            # DURCL Course Project Notebook

            This notebook is the single-file deliverable for the Adaptive Environment Selection course project.
            It is organized into three parts:

            1. **Training pipeline**: dataset access, project-specific training code, scheduler implementation, and the exact heavy-training command path.
            2. **Evaluation pipeline**: checked-in result aggregation by default, with an optional checkpoint-based evaluation entrypoint.
            3. **Result interpretation**: tables, figures, toy sanity checks, and the written analysis used in the report.

            The notebook is designed to be **self-bootstrapping**:

            - If it is run inside a cloned copy of the repository, it uses the local files directly.
            - If it is run as a standalone notebook, it can download the repository snapshot from GitHub.
            - Large artifacts such as model checkpoints are intentionally loaded by link rather than embedded in the notebook.
            - It installs a small set of missing Python packages automatically; it does not require a manual conda setup walkthrough.

            Heavy RL training is **not executed by default** because it requires a compatible multi-GPU environment, but the code path and command construction are included here so the full workflow is documented in one place.
            """
        ),
        code(
            """
            from __future__ import annotations

            import importlib
            import json
            import math
            import os
            import shutil
            import subprocess
            import sys
            import textwrap
            import urllib.request
            import zipfile
            from pathlib import Path


            def ensure_packages(packages: list[str]) -> None:
                missing = []
                for pkg in packages:
                    module = pkg.split("==", 1)[0].split(">=", 1)[0]
                    module = {"pillow": "PIL", "pyarrow": "pyarrow"}.get(module, module)
                    try:
                        importlib.import_module(module)
                    except Exception:
                        missing.append(pkg)
                if missing:
                    print("Installing missing packages:", missing)
                    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


            ensure_packages([
                "pandas",
                "matplotlib",
                "pillow",
                "pyarrow",
                "reportlab",
                "pytest",
            ])

            import pandas as pd
            import matplotlib.pyplot as plt
            from IPython.display import Code, Image, Markdown, display


            plt.style.use("seaborn-v0_8-whitegrid")
            pd.set_option("display.max_colwidth", 120)
            """
        ),
        code(
            """
            RUN_UNIT_TESTS = True
            RUN_HEAVY_TRAINING = False
            RUN_CHECKPOINT_EVAL = False
            INSTALL_TRAINING_EXTRAS = False
            PREPARE_MATH_ZEBRA_DATA = False
            REGENERATE_FIGURES = False
            RERUN_TOY_SIMULATION = False
            TRAINING_PROFILE = os.environ.get("AES_TRAINING_PROFILE", "math_zebra_rebucket_globalacc_from80_to200").strip()

            REPO_URL = "https://github.com/kkkzzz0101/Adaptive-Env-Selection.git"
            REPO_ZIP_URL = "https://github.com/kkkzzz0101/Adaptive-Env-Selection/archive/refs/heads/main.zip"
            RAW_BASE = "https://raw.githubusercontent.com/kkkzzz0101/Adaptive-Env-Selection/main"

            HF_MODEL_ID = os.environ.get(
                "AES_HF_MODEL_ID",
                "zkkk452/adaptive-env-selection-checkpoint",
            ).strip()
            CHECKPOINT_ARCHIVE_URL = os.environ.get("AES_CHECKPOINT_ARCHIVE_URL", "").strip()
            LOCAL_MODEL_PATH = os.environ.get("AES_LOCAL_MODEL_PATH", "").strip()
            HF_TOKEN = (
                os.environ.get("HF_TOKEN", "").strip()
                or os.environ.get("HUGGINGFACE_HUB_TOKEN", "").strip()
            )

            print({
                "RUN_UNIT_TESTS": RUN_UNIT_TESTS,
                "RUN_HEAVY_TRAINING": RUN_HEAVY_TRAINING,
                "RUN_CHECKPOINT_EVAL": RUN_CHECKPOINT_EVAL,
                "INSTALL_TRAINING_EXTRAS": INSTALL_TRAINING_EXTRAS,
                "PREPARE_MATH_ZEBRA_DATA": PREPARE_MATH_ZEBRA_DATA,
                "REGENERATE_FIGURES": REGENERATE_FIGURES,
                "RERUN_TOY_SIMULATION": RERUN_TOY_SIMULATION,
                "TRAINING_PROFILE": TRAINING_PROFILE,
            })
            """
        ),
        code(
            """
            NOTEBOOK_CWD = Path.cwd()


            def github_raw_url(path: str) -> str:
                return f"{RAW_BASE}/{path}"


            def find_repo_root(start: Path) -> Path | None:
                start = start.resolve()
                for candidate in [start, *start.parents]:
                    if (candidate / "README.md").exists() and (candidate / "paper" / "deliverables").exists():
                        return candidate
                return None


            def download_repo_snapshot(dest_dir: Path) -> Path:
                dest_dir.mkdir(parents=True, exist_ok=True)
                zip_path = dest_dir / "Adaptive-Env-Selection-main.zip"
                extract_dir = dest_dir / "Adaptive-Env-Selection-main"
                if not extract_dir.exists():
                    print(f"Downloading repository snapshot from {REPO_ZIP_URL}")
                    urllib.request.urlretrieve(REPO_ZIP_URL, zip_path)
                    with zipfile.ZipFile(zip_path) as zf:
                        zf.extractall(dest_dir)
                return extract_dir


            ROOT = find_repo_root(NOTEBOOK_CWD)
            if ROOT is None:
                ROOT = download_repo_snapshot(NOTEBOOK_CWD / "aes_notebook_runtime")
                print(f"Using downloaded repository at: {ROOT}")
            else:
                print(f"Using local repository at: {ROOT}")

            sys.path.insert(0, str(ROOT / "src"))
            sys.path.insert(0, str(ROOT / "scripts"))
            """
        ),
        code(
            """
            asset_rows = [
                {"artifact": "Repository root", "path": str(ROOT), "exists": ROOT.exists()},
                {"artifact": "Notebook", "path": str(ROOT / "paper/deliverables/project_groupID.ipynb"), "exists": (ROOT / "paper/deliverables/project_groupID.ipynb").exists()},
                {"artifact": "Math+Zebra data", "path": str(ROOT / "experiments/baselines/data_math_zebra_800"), "exists": (ROOT / "experiments/baselines/data_math_zebra_800").exists()},
                {"artifact": "SEC eval data", "path": str(ROOT / "references/sec/data"), "exists": (ROOT / "references/sec/data").exists()},
                {"artifact": "Scheduler source", "path": str(ROOT / "src/scheduler/adaptive_curriculum_scheduler.py"), "exists": (ROOT / "src/scheduler/adaptive_curriculum_scheduler.py").exists()},
                {"artifact": "Eval probe", "path": str(ROOT / "scripts/sec_inference_probe.py"), "exists": (ROOT / "scripts/sec_inference_probe.py").exists()},
                {"artifact": "Result report", "path": str(ROOT / "docs/result_report.md"), "exists": (ROOT / "docs/result_report.md").exists()},
                {"artifact": "Paper figures", "path": str(ROOT / "paper/figures"), "exists": (ROOT / "paper/figures").exists()},
            ]
            display(pd.DataFrame(asset_rows))
            """
        ),
        md(
            """
            ## Part I. Training Pipeline

            The final project story is the `math + zebra` line, not the earlier four-dataset SEC4 baseline.
            This section therefore centers the training path that leads to the final **window-fitted linear rebucketing** result:

            - `math + zebra` data preparation,
            - the adaptive scheduler / rebucketing implementation,
            - the warm-80 and `80 -> 200` continuation setup,
            - and a portable heavy-training command that can be launched from the notebook in a compatible GPU environment.

            By default the notebook only documents and previews the pipeline. It launches training only if `RUN_HEAVY_TRAINING=True`.
            """
        ),
        code(
            """
            import pyarrow.parquet as pq

            math_zebra_root = ROOT / "experiments" / "baselines" / "data_math_zebra_800" / "mixed"
            sec_eval_root = ROOT / "references" / "sec" / "data"

            rows = []
            for path in [math_zebra_root / "train.parquet", math_zebra_root / "val.parquet"]:
                if path.exists():
                    rel = path.relative_to(ROOT).as_posix()
                    rows.append({
                        "file": rel,
                        "rows": pq.read_metadata(path).num_rows,
                        "download_url": github_raw_url(rel),
                    })

            for path in sorted(sec_eval_root.rglob("*.parquet")):
                rel = path.relative_to(ROOT).as_posix()
                rows.append({
                    "file": rel,
                    "rows": pq.read_metadata(path).num_rows,
                    "download_url": github_raw_url(rel),
                })

            data_df = pd.DataFrame(rows).sort_values("file").reset_index(drop=True)
            display(data_df)
            """
        ),
        code(
            """
            def show_source(rel_path: str, start: int = 1, end: int | None = None) -> None:
                path = ROOT / rel_path
                lines = path.read_text(encoding="utf-8").splitlines()
                if end is None:
                    end = len(lines)
                snippet = "\\n".join(f"{idx:4d}: {line}" for idx, line in enumerate(lines[start - 1:end], start))
                print(f"--- {rel_path} ---")
                print(snippet)


            show_source("scripts/prepare_sec4_random_dataset.py", 1, 140)
            """
        ),
        code(
            """
            show_source("src/scheduler/adaptive_curriculum_scheduler.py", 1, 220)
            """
        ),
        code(
            """
            for rel_path in [
                "scripts/run_mathzebra_globalacc_to200_chain.sh",
                "scripts/run_scheduler_norebucket_4gpu_mathzebra_globalacc_warm80.sh",
                "scripts/run_scheduler_microbucket_4gpu_mathzebra_globalacc_from80.sh",
            ]:
                path = ROOT / rel_path
                if path.exists():
                    show_source(rel_path, 1, 160)
                else:
                    print(f"Missing script in current checkout: {rel_path}")
            """
        ),
        code(
            """
            if RUN_UNIT_TESTS:
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", "tests/test_scheduler_unit.py", "-q"],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                print(result.stdout)
            else:
                print("Skipping unit tests. Set RUN_UNIT_TESTS=True to execute the scheduler sanity suite.")
            """
        ),
        code(
            """
            TRAIN_BASE_MODEL = os.environ.get("AES_TRAIN_MODEL_PATH", "Qwen/Qwen2.5-1.5B-Instruct").strip()
            CHECKPOINT_ROOT = Path(os.environ.get("AES_CHECKPOINT_ROOT", str(ROOT / "checkpoints"))).resolve()
            MATH_ZEBRA_DATA_ROOT = ROOT / "experiments" / "baselines" / "data_math_zebra_800" / "mixed"
            VERL_ROOT = ROOT / "references" / "DUMP"

            MATH_ZEBRA_DATA_CONFIG = {
                "countdown_levels": "",
                "zebra_levels": "1,2,3,4",
                "arc_levels": "",
                "math_levels": "1,2,3,4,5",
                "train_per_bucket": 100,
                "val_per_bucket": 20,
                "test_per_bucket": 0,
                "math_train_per_level": 80,
                "math_val_per_level": 10,
                "seed": 42,
            }

            TRAINING_PROFILES = {
                "math_zebra_baseline_step200": {
                    "project_name": "aes_baseline_mathzebra",
                    "exp_name": "baseline_random_dump_mathzebra800_step200",
                    "resume": None,
                    "scheduler_env": {},
                    "trainer": {"steps": 200, "epochs": 30, "save_freq": 100, "test_freq": 100},
                },
                "math_zebra_norebucket_step200": {
                    "project_name": "aes_scheduler_mathzebra",
                    "exp_name": "scheduler_norebucket_mathzebra800_step200",
                    "resume": None,
                    "scheduler_env": {
                        "USE_ADAPTIVE_SCHEDULER": "1",
                        "ADAPTIVE_NUM_CLUSTERS": "5",
                        "ADAPTIVE_WARMUP_STEPS": "1000000",
                        "ADAPTIVE_REBUCKET_INTERVAL": "10",
                        "ADAPTIVE_MIN_OBS": "6",
                        "ADAPTIVE_ACTIVE_WINDOW": "80",
                        "ADAPTIVE_DECAY": "0.6",
                        "ADAPTIVE_UCB_BETA": "0.85",
                        "ADAPTIVE_SOFTMAX_TAU": "0.2",
                        "ADAPTIVE_PROB_FLOOR_EPS": "0.05",
                        "ADAPTIVE_MIGRATION_GAMMA": "2.0",
                        "ADAPTIVE_MIGRATION_CONSECUTIVE": "3",
                        "ADAPTIVE_ALLOW_REVERSE_MIGRATION": "0",
                        "ADAPTIVE_REVERSE_MIGRATION_GAMMA": "2.5",
                        "ADAPTIVE_REVERSE_MIGRATION_CONSECUTIVE": "4",
                        "ADAPTIVE_CALIBRATION_JSON": str(ROOT / "experiments" / "scheduler" / "calibration_math_zebra.json"),
                        "VERL_CURRICULUM_DIAGNOSTICS": "0",
                    },
                    "trainer": {"steps": 200, "epochs": 30, "save_freq": 100, "test_freq": 100},
                },
                "math_zebra_norebucket_globalacc_warm80": {
                    "project_name": "aes_scheduler_mathzebra_globalacc",
                    "exp_name": "scheduler_norebucket_mathzebra800_globalacc_warm80",
                    "resume": None,
                    "scheduler_env": {
                        "USE_ADAPTIVE_SCHEDULER": "1",
                        "ADAPTIVE_NUM_CLUSTERS": "5",
                        "ADAPTIVE_WARMUP_STEPS": "1000000",
                        "ADAPTIVE_REBUCKET_INTERVAL": "20",
                        "ADAPTIVE_MIN_OBS": "8",
                        "ADAPTIVE_ACTIVE_WINDOW": "80",
                        "ADAPTIVE_DECAY": "0.6",
                        "ADAPTIVE_UCB_BETA": "0.85",
                        "ADAPTIVE_SOFTMAX_TAU": "0.2",
                        "ADAPTIVE_PROB_FLOOR_EPS": "0.05",
                        "ADAPTIVE_MIGRATION_GAMMA": "2.0",
                        "ADAPTIVE_MIGRATION_CONSECUTIVE": "2",
                        "ADAPTIVE_ALLOW_REVERSE_MIGRATION": "0",
                        "ADAPTIVE_REVERSE_MIGRATION_GAMMA": "2.5",
                        "ADAPTIVE_REVERSE_MIGRATION_CONSECUTIVE": "4",
                        "ADAPTIVE_MICRO_BUCKETS_PER_LEVEL": "5",
                        "ADAPTIVE_TARGET_MICRO_BUCKET_SIZE": "20",
                        "ADAPTIVE_TRAJECTORY_WINDOW_OBSERVATIONS": "60",
                        "ADAPTIVE_MIN_GROUP_OBS_PER_WINDOW": "8",
                        "ADAPTIVE_CLUSTER_MIN_MICRO_BUCKETS": "2",
                        "ADAPTIVE_CLUSTER_COOLDOWN_STEPS": "40",
                        "ADAPTIVE_MAX_SWAPS_PER_PAIR": "1",
                        "ADAPTIVE_MAX_MOVES_PER_CLUSTER": "1",
                        "ADAPTIVE_ENABLE_MOVE_SECOND": "1",
                        "ADAPTIVE_CALIBRATION_JSON": str(ROOT / "experiments" / "scheduler" / "calibration_math_zebra.json"),
                        "VERL_CURRICULUM_DIAGNOSTICS": "0",
                    },
                    "trainer": {"steps": 80, "epochs": 20, "save_freq": 80, "test_freq": 80},
                },
                "math_zebra_norebucket_globalacc_from80_to200": {
                    "project_name": "aes_scheduler_mathzebra_globalacc",
                    "exp_name": "scheduler_norebucket_mathzebra800_globalacc_from80_to200",
                    "resume": {
                        "project_name": "aes_scheduler_mathzebra_globalacc",
                        "exp_name": "scheduler_norebucket_mathzebra800_globalacc_warm80",
                        "source_step": 80,
                    },
                    "scheduler_env": {
                        "USE_ADAPTIVE_SCHEDULER": "1",
                        "ADAPTIVE_NUM_CLUSTERS": "5",
                        "ADAPTIVE_WARMUP_STEPS": "1000000",
                        "ADAPTIVE_REBUCKET_INTERVAL": "20",
                        "ADAPTIVE_MIN_OBS": "8",
                        "ADAPTIVE_ACTIVE_WINDOW": "80",
                        "ADAPTIVE_DECAY": "0.6",
                        "ADAPTIVE_UCB_BETA": "0.85",
                        "ADAPTIVE_SOFTMAX_TAU": "0.2",
                        "ADAPTIVE_PROB_FLOOR_EPS": "0.05",
                        "ADAPTIVE_MIGRATION_GAMMA": "2.0",
                        "ADAPTIVE_MIGRATION_CONSECUTIVE": "2",
                        "ADAPTIVE_ALLOW_REVERSE_MIGRATION": "0",
                        "ADAPTIVE_REVERSE_MIGRATION_GAMMA": "2.5",
                        "ADAPTIVE_REVERSE_MIGRATION_CONSECUTIVE": "4",
                        "ADAPTIVE_MICRO_BUCKETS_PER_LEVEL": "5",
                        "ADAPTIVE_TARGET_MICRO_BUCKET_SIZE": "20",
                        "ADAPTIVE_TRAJECTORY_WINDOW_OBSERVATIONS": "60",
                        "ADAPTIVE_MIN_GROUP_OBS_PER_WINDOW": "8",
                        "ADAPTIVE_CLUSTER_MIN_MICRO_BUCKETS": "2",
                        "ADAPTIVE_CLUSTER_COOLDOWN_STEPS": "40",
                        "ADAPTIVE_MAX_SWAPS_PER_PAIR": "1",
                        "ADAPTIVE_MAX_MOVES_PER_CLUSTER": "1",
                        "ADAPTIVE_ENABLE_MOVE_SECOND": "1",
                        "ADAPTIVE_IGNORE_CHECKPOINT_STATE": "0",
                        "ADAPTIVE_IGNORE_DATALOADER_CHECKPOINT": "1",
                        "ADAPTIVE_CALIBRATION_JSON": str(ROOT / "experiments" / "scheduler" / "calibration_math_zebra.json"),
                        "VERL_CURRICULUM_DIAGNOSTICS": "0",
                    },
                    "trainer": {"steps": 200, "epochs": 30, "save_freq": 100, "test_freq": 100},
                },
                "math_zebra_rebucket_globalacc_from80_to200": {
                    "project_name": "aes_scheduler_mathzebra_globalacc",
                    "exp_name": "scheduler_microbucket_mathzebra800_globalacc_from80_to200",
                    "resume": {
                        "project_name": "aes_scheduler_mathzebra_globalacc",
                        "exp_name": "scheduler_norebucket_mathzebra800_globalacc_warm80",
                        "source_step": 80,
                    },
                    "scheduler_env": {
                        "USE_ADAPTIVE_SCHEDULER": "1",
                        "ADAPTIVE_NUM_CLUSTERS": "5",
                        "ADAPTIVE_WARMUP_STEPS": "80",
                        "ADAPTIVE_REBUCKET_INTERVAL": "20",
                        "ADAPTIVE_MIN_OBS": "8",
                        "ADAPTIVE_ACTIVE_WINDOW": "80",
                        "ADAPTIVE_DECAY": "0.6",
                        "ADAPTIVE_UCB_BETA": "0.85",
                        "ADAPTIVE_SOFTMAX_TAU": "0.2",
                        "ADAPTIVE_PROB_FLOOR_EPS": "0.05",
                        "ADAPTIVE_MIGRATION_GAMMA": "2.0",
                        "ADAPTIVE_MIGRATION_CONSECUTIVE": "2",
                        "ADAPTIVE_ALLOW_REVERSE_MIGRATION": "0",
                        "ADAPTIVE_REVERSE_MIGRATION_GAMMA": "2.5",
                        "ADAPTIVE_REVERSE_MIGRATION_CONSECUTIVE": "4",
                        "ADAPTIVE_MICRO_BUCKETS_PER_LEVEL": "5",
                        "ADAPTIVE_TARGET_MICRO_BUCKET_SIZE": "20",
                        "ADAPTIVE_TRAJECTORY_WINDOW_OBSERVATIONS": "60",
                        "ADAPTIVE_MIN_GROUP_OBS_PER_WINDOW": "8",
                        "ADAPTIVE_CLUSTER_MIN_MICRO_BUCKETS": "2",
                        "ADAPTIVE_CLUSTER_COOLDOWN_STEPS": "40",
                        "ADAPTIVE_MAX_SWAPS_PER_PAIR": "1",
                        "ADAPTIVE_MAX_MOVES_PER_CLUSTER": "1",
                        "ADAPTIVE_ENABLE_MOVE_SECOND": "1",
                        "ADAPTIVE_IGNORE_CHECKPOINT_STATE": "0",
                        "ADAPTIVE_IGNORE_DATALOADER_CHECKPOINT": "1",
                        "ADAPTIVE_CALIBRATION_JSON": str(ROOT / "experiments" / "scheduler" / "calibration_math_zebra.json"),
                        "VERL_CURRICULUM_DIAGNOSTICS": "0",
                    },
                    "trainer": {"steps": 200, "epochs": 30, "save_freq": 100, "test_freq": 50},
                },
            }


            def environment_report() -> pd.DataFrame:
                rows = []
                for module in ["torch", "transformers", "ray"]:
                    try:
                        imported = importlib.import_module(module)
                        rows.append({"component": module, "ready": True, "detail": getattr(imported, "__version__", "unknown")})
                    except Exception as exc:
                        rows.append({"component": module, "ready": False, "detail": type(exc).__name__})
                rows.append({"component": "vendored verl", "ready": VERL_ROOT.exists(), "detail": str(VERL_ROOT)})
                try:
                    import torch
                    rows.append({"component": "cuda_available", "ready": bool(torch.cuda.is_available()), "detail": str(torch.cuda.device_count())})
                except Exception as exc:
                    rows.append({"component": "cuda_available", "ready": False, "detail": type(exc).__name__})
                return pd.DataFrame(rows)


            if INSTALL_TRAINING_EXTRAS:
                ensure_packages(["transformers", "ray", "sentencepiece", "protobuf"])
                print("Installed lightweight Python extras. Torch/CUDA still needs to match the local GPU runtime.")
            else:
                print("Skipping optional training-extra installation. Set INSTALL_TRAINING_EXTRAS=True to install transformers/ray helpers.")

            display(environment_report())
            print("Training base model:", TRAIN_BASE_MODEL)
            print("Training profile:", TRAINING_PROFILE)
            print("Checkpoint root:", CHECKPOINT_ROOT)
            """
        ),
        code(
            """
            prepare_cmd = [
                sys.executable,
                str(ROOT / "scripts" / "prepare_sec4_random_dataset.py"),
                "--countdown-levels", MATH_ZEBRA_DATA_CONFIG["countdown_levels"],
                "--zebra-levels", MATH_ZEBRA_DATA_CONFIG["zebra_levels"],
                "--arc-levels", MATH_ZEBRA_DATA_CONFIG["arc_levels"],
                "--math-levels", MATH_ZEBRA_DATA_CONFIG["math_levels"],
                "--train-per-bucket", str(MATH_ZEBRA_DATA_CONFIG["train_per_bucket"]),
                "--val-per-bucket", str(MATH_ZEBRA_DATA_CONFIG["val_per_bucket"]),
                "--test-per-bucket", str(MATH_ZEBRA_DATA_CONFIG["test_per_bucket"]),
                "--math-train-per-level", str(MATH_ZEBRA_DATA_CONFIG["math_train_per_level"]),
                "--math-val-per-level", str(MATH_ZEBRA_DATA_CONFIG["math_val_per_level"]),
                "--seed", str(MATH_ZEBRA_DATA_CONFIG["seed"]),
                "--out-dir", str(MATH_ZEBRA_DATA_ROOT.parent),
            ]

            print("Math+Zebra data-preparation command:")
            print(" ".join(prepare_cmd))

            if PREPARE_MATH_ZEBRA_DATA:
                subprocess.run(prepare_cmd, cwd=ROOT, check=True)
            else:
                print("Dataset preparation is skipped by default because the checked-in math+zebra parquet files are already available.")
            """
        ),
        code(
            """
            training_profile_df = pd.DataFrame([
                {
                    "profile": name,
                    "project_name": spec["project_name"],
                    "exp_name": spec["exp_name"],
                    "resume_from": f"{spec['resume']['exp_name']}@{spec['resume']['source_step']}" if spec["resume"] else "",
                    "steps": spec["trainer"]["steps"],
                    "test_freq": spec["trainer"]["test_freq"],
                    "scheduler": "window rebucket" if "microbucket" in spec["exp_name"] else ("no rebucket" if "scheduler" in spec["exp_name"] else "random baseline"),
                }
                for name, spec in TRAINING_PROFILES.items()
            ])
            display(training_profile_df)
            """
        ),
        code(
            """
            COMMON_TRAIN_KWARGS = {
                "train_batch_size": 32,
                "val_batch_size": 64,
                "rollout_n": 4,
                "max_prompt_length": 960,
                "max_response_length": 384,
                "actor_ppo_micro_batch_size_per_gpu": 1,
                "critic_ppo_micro_batch_size_per_gpu": 2,
                "critic_forward_micro_batch_size_per_gpu": 2,
                "rollout_micro_batch_size": 2,
                "actor_logprob_micro_batch_size": 1,
                "ref_logprob_micro_batch_size": 1,
                "gpu_memory_utilization": 0.85,
                "max_num_batched_tokens": 16384,
                "max_num_seqs": 2048,
                "n_gpus_per_node": int(os.environ.get("N_GPUS_PER_NODE", "4")),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3"),
            }


            def prepare_resume_link(spec: dict[str, object]) -> None:
                resume = spec["resume"]
                if not resume:
                    return
                source_dir = CHECKPOINT_ROOT / str(resume["project_name"]) / str(resume["exp_name"]) / f"global_step_{resume['source_step']}"
                target_dir = CHECKPOINT_ROOT / str(spec["project_name"]) / str(spec["exp_name"])
                target_dir.mkdir(parents=True, exist_ok=True)
                link_path = target_dir / f"global_step_{resume['source_step']}"
                if not link_path.exists():
                    link_path.symlink_to(source_dir)
                (target_dir / "latest_checkpointed_iteration.txt").write_text(str(resume["source_step"]), encoding="utf-8")


            def build_training_command(profile_name: str) -> tuple[list[str], dict[str, str]]:
                spec = TRAINING_PROFILES[profile_name]
                prepare_resume_link(spec)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = COMMON_TRAIN_KWARGS["cuda_visible_devices"]
                env["PYTHONPATH"] = str(VERL_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
                env["PYTORCH_CUDA_ALLOC_CONF"] = env.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
                env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = env.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
                env.update(spec["scheduler_env"])

                run_dir = CHECKPOINT_ROOT / spec["project_name"] / spec["exp_name"]
                cmd = [
                    sys.executable,
                    "-m",
                    "verl.trainer.main_ppo",
                    "algorithm.adv_estimator=grpo",
                    f"data.train_files={MATH_ZEBRA_DATA_ROOT / 'train.parquet'}",
                    f"data.val_files={MATH_ZEBRA_DATA_ROOT / 'val.parquet'}",
                    f"data.train_batch_size={COMMON_TRAIN_KWARGS['train_batch_size']}",
                    f"data.val_batch_size={COMMON_TRAIN_KWARGS['val_batch_size']}",
                    f"data.enable_curriculum_learning={'True' if spec['scheduler_env'] else 'False'}",
                    "data.data_source_key=null",
                    f"data.max_prompt_length={COMMON_TRAIN_KWARGS['max_prompt_length']}",
                    f"data.max_response_length={COMMON_TRAIN_KWARGS['max_response_length']}",
                    f"actor_rollout_ref.model.path={TRAIN_BASE_MODEL}",
                    "actor_rollout_ref.model.use_remove_padding=False",
                    "actor_rollout_ref.model.enable_gradient_checkpointing=True",
                    "actor_rollout_ref.actor.use_dynamic_bsz=True",
                    "actor_rollout_ref.actor.optim.lr=1e-6",
                    f"actor_rollout_ref.actor.ppo_mini_batch_size={COMMON_TRAIN_KWARGS['train_batch_size']}",
                    f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={COMMON_TRAIN_KWARGS['actor_ppo_micro_batch_size_per_gpu']}",
                    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192",
                    "actor_rollout_ref.actor.use_kl_loss=True",
                    "actor_rollout_ref.actor.kl_loss_coef=0.001",
                    "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
                    "actor_rollout_ref.actor.fsdp_config.param_offload=False",
                    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
                    "actor_rollout_ref.ref.fsdp_config.param_offload=False",
                    f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={COMMON_TRAIN_KWARGS['ref_logprob_micro_batch_size']}",
                    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192",
                    "actor_rollout_ref.rollout.name=hf",
                    f"actor_rollout_ref.rollout.n={COMMON_TRAIN_KWARGS['rollout_n']}",
                    "actor_rollout_ref.rollout.temperature=1.0",
                    "actor_rollout_ref.rollout.top_p=1.0",
                    "actor_rollout_ref.rollout.top_k=0",
                    "actor_rollout_ref.rollout.do_sample=True",
                    "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
                    f"+actor_rollout_ref.rollout.micro_batch_size={COMMON_TRAIN_KWARGS['rollout_micro_batch_size']}",
                    f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={COMMON_TRAIN_KWARGS['actor_logprob_micro_batch_size']}",
                    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192",
                    f"actor_rollout_ref.rollout.gpu_memory_utilization={COMMON_TRAIN_KWARGS['gpu_memory_utilization']}",
                    f"actor_rollout_ref.rollout.max_num_batched_tokens={COMMON_TRAIN_KWARGS['max_num_batched_tokens']}",
                    f"actor_rollout_ref.rollout.max_num_seqs={COMMON_TRAIN_KWARGS['max_num_seqs']}",
                    f"critic.model.path={TRAIN_BASE_MODEL}",
                    f"critic.model.tokenizer_path={TRAIN_BASE_MODEL}",
                    "critic.model.use_remove_padding=False",
                    "critic.model.enable_gradient_checkpointing=False",
                    "critic.optim.lr=1e-5",
                    f"critic.ppo_micro_batch_size_per_gpu={COMMON_TRAIN_KWARGS['critic_ppo_micro_batch_size_per_gpu']}",
                    f"critic.forward_micro_batch_size_per_gpu={COMMON_TRAIN_KWARGS['critic_forward_micro_batch_size_per_gpu']}",
                    "critic.ppo_max_token_len_per_gpu=12288",
                    "algorithm.kl_ctrl.kl_coef=0.001",
                    "trainer.critic_warmup=0",
                    "trainer.logger=['console']",
                    f"trainer.project_name={spec['project_name']}",
                    f"trainer.experiment_name={spec['exp_name']}",
                    f"trainer.n_gpus_per_node={COMMON_TRAIN_KWARGS['n_gpus_per_node']}",
                    "trainer.nnodes=1",
                    f"trainer.default_local_dir={run_dir}",
                    "trainer.default_hdfs_dir=null",
                    f"trainer.save_freq={spec['trainer']['save_freq']}",
                    f"trainer.test_freq={spec['trainer']['test_freq']}",
                    f"trainer.total_epochs={spec['trainer']['epochs']}",
                    "+trainer.val_before_train=False",
                    f"trainer.total_training_steps={spec['trainer']['steps']}",
                ]
                return cmd, env


            train_parquet = MATH_ZEBRA_DATA_ROOT / "train.parquet"
            val_parquet = MATH_ZEBRA_DATA_ROOT / "val.parquet"
            if not (train_parquet.exists() and val_parquet.exists()):
                raise FileNotFoundError("math+zebra parquet files are missing. Set PREPARE_MATH_ZEBRA_DATA=True to regenerate them.")

            cmd, env = build_training_command(TRAINING_PROFILE)
            print("Training command preview for", TRAINING_PROFILE)
            print(" ".join(str(x) for x in cmd[:18]), "...")
            print("This profile will download or reuse the base model:", TRAIN_BASE_MODEL)

            if RUN_HEAVY_TRAINING:
                print("Launching heavy RL training. This requires a compatible multi-GPU CUDA environment and vendored verl dependencies.")
                subprocess.run(cmd, cwd=ROOT, env=env, check=True)
            else:
                print("Heavy RL training is skipped by default. Set RUN_HEAVY_TRAINING=True only after the environment report is ready.")
            """
        ),
        md(
            """
            ## Part II. Evaluation Pipeline

            The notebook supports two evaluation modes:

            1. **Default**: load the checked-in `math + zebra` result summaries already stored in the repository.
            2. **Optional checkpoint eval**: load a checkpoint from Hugging Face or a local path and run the SEC inference probe.

            This keeps the notebook lightweight for grading while still documenting the eval path for the two checkpoint targets we still care about:

            - the `baseline_mathzebra200` family,
            - and the `window rebucket` run recovered at step 100.

            The public checkpoint target reserved for those uploads is:
            - `zkkk452/adaptive-env-selection-checkpoint`
            """
        ),
        md(
            """
            ### Final Rebucket Run Used In The Submission

            The final notebook should center the **window-fitted linear rebucketing method** rather than the earlier rebucketing variants that still appear in some checked-in reports.

            The authoritative rebucketing result used in the course submission is:

            - `rebucket 80 -> 200`
            - step 100: `math = 0.440`, `zebra = 0.237`
            - step 150: `math = 0.520`, `zebra = 0.237`
            - step 200: `math = 0.580`, `zebra = 0.300`

            The final comparison shown below focuses on the **step-200** result.
            """
        ),
        code(
            """
            final_rebucket = pd.read_csv(
                ROOT / "experiments" / "results" / "final_rebucket_window_linear_80_200.csv"
            )
            display(final_rebucket)
            """
        ),
        code(
            """
            mz = pd.read_csv(ROOT / "experiments" / "results" / "math_zebra_2data" / "baseline_vs_norebucket_metrics.csv")
            display(mz)

            step200 = mz[mz["step"] == 200].copy()
            if len(step200) == 2:
                baseline = step200[step200["run"] == "baseline_random"].iloc[0]
                scheduler = step200[step200["run"] == "scheduler_no_rebucket"].iloc[0]
                rebucket = final_rebucket[final_rebucket["step"] == 200].iloc[0]

                final_comparison = pd.DataFrame([
                    {
                        "task": "Math",
                        "random_step200": baseline["math_train"],
                        "no_rebucket_step200": scheduler["math_train"],
                        "rebucket_80_200_step200": rebucket["math"],
                        "rebucket_vs_random": rebucket["math"] - baseline["math_train"],
                        "rebucket_vs_no_rebucket": rebucket["math"] - scheduler["math_train"],
                    },
                    {
                        "task": "Zebra",
                        "random_step200": baseline["zebra_train"],
                        "no_rebucket_step200": scheduler["zebra_train"],
                        "rebucket_80_200_step200": rebucket["zebra"],
                        "rebucket_vs_random": rebucket["zebra"] - baseline["zebra_train"],
                        "rebucket_vs_no_rebucket": rebucket["zebra"] - scheduler["zebra_train"],
                    },
                ])
                display(final_comparison)
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            for ax, task in zip(axes, ["math", "zebra"]):
                ax.plot(final_rebucket["step"], final_rebucket[task], marker="o", linewidth=2.2)
                ax.set_title(f"Rebucket 80->200 ({task.capitalize()})")
                ax.set_xlabel("Step")
                ax.set_ylabel("Validation score")
                ax.set_xticks(final_rebucket["step"].tolist())
                ax.set_ylim(0.20 if task == "zebra" else 0.40, 0.62)
                for _, row in final_rebucket.iterrows():
                    ax.text(row["step"], row[task] + 0.01, f"{row[task]:.3f}", ha="center", fontsize=9)

            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            recovered_inventory = pd.read_csv(
                ROOT / "experiments" / "results" / "window_rebucket_globalacc_80_200" / "checkpoint_inventory.csv"
            )
            recovered_signature = pd.read_csv(
                ROOT / "experiments" / "results" / "window_rebucket_globalacc_80_200" / "step100_signature_summary.csv"
            )

            display(recovered_inventory)
            display(recovered_signature)
            """
        ),
        code(
            """
            def ensure_eval_dependencies() -> None:
                ensure_packages(["transformers", "torch"])


            def resolve_model_reference(root: Path) -> str:
                if HF_MODEL_ID:
                    return HF_MODEL_ID
                if LOCAL_MODEL_PATH:
                    return LOCAL_MODEL_PATH
                if CHECKPOINT_ARCHIVE_URL:
                    archive_dir = root / "artifacts" / "downloaded_checkpoint"
                    archive_dir.mkdir(parents=True, exist_ok=True)
                    archive_path = archive_dir / "checkpoint_archive"
                    if not archive_path.exists():
                        print(f"Downloading checkpoint archive from {CHECKPOINT_ARCHIVE_URL}")
                        urllib.request.urlretrieve(CHECKPOINT_ARCHIVE_URL, archive_path)
                    raise ValueError("Downloaded checkpoint archives should be extracted manually or provided as AES_LOCAL_MODEL_PATH.")
                raise ValueError("No checkpoint source configured. Set AES_HF_MODEL_ID or AES_LOCAL_MODEL_PATH.")


            if RUN_CHECKPOINT_EVAL:
                ensure_eval_dependencies()
                model_ref = resolve_model_reference(ROOT)
                probe_out = ROOT / "experiments" / "notebook_eval"
                env = os.environ.copy()
                env["AES_ROOT"] = str(ROOT)
                env["AES_MODEL_PATH"] = model_ref
                if HF_TOKEN:
                    env["HF_TOKEN"] = HF_TOKEN
                    env["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

                datasets = ["countdown", "zebra", "arc", "math"]
                for dataset in datasets:
                    cmd = [
                        sys.executable,
                        str(ROOT / "scripts" / "sec_inference_probe.py"),
                        "--dataset",
                        dataset,
                        "--model-path",
                        model_ref,
                        "--sec-root",
                        str(ROOT / "references" / "sec" / "data"),
                        "--out-dir",
                        str(probe_out / dataset),
                        "--n-per-difficulty",
                        "10",
                        "--rollouts",
                        "1",
                    ]
                    print("Running:", " ".join(cmd))
                    subprocess.run(cmd, cwd=ROOT, env=env, check=True)

                summaries = sorted(probe_out.rglob("*_summary.csv"))
                display(pd.DataFrame({"summary_csv": [p.relative_to(ROOT).as_posix() for p in summaries]}))
            else:
                print(
                    "Checkpoint eval is disabled. To enable it, set RUN_CHECKPOINT_EVAL=True. "
                    f"The current default Hugging Face target is: {HF_MODEL_ID}"
                )
            """
        ),
        md(
            """
            ## Part III. Result Interpretation

            This section keeps only the report context that still matches the final submission story.
            Some checked-in documents were written before the final `rebucket 80 -> 200` result was available, so the notebook should not treat every older report number as authoritative.

            The current notebook should therefore:

            - use the final `rebucket 80 -> 200` step-200 result as the main rebucketing outcome,
            - use the checked-in random and no-rebucket CSV as the baseline comparison source,
            - and keep older rebucketing writeups only as background context.

            The guiding idea is that the notebook should serve as a compact, inspectable version of the repository for course submission.
            """
        ),
        code(
            """
            checked_in_report_note = pd.DataFrame([
                {
                    "file": "docs/result_report.md",
                    "status": "outdated for final notebook",
                    "reason": "still summarizes the older difficulty-init rebucketing numbers",
                },
                {
                    "file": "docs/course_project_report.md",
                    "status": "outdated for final notebook",
                    "reason": "still highlights the earlier 0.540 / 0.300 acc-init rebucketing result",
                },
                {
                    "file": "README.md",
                    "status": "partially outdated",
                    "reason": "contains older rebucketing highlights and should not override the final 80->200 submission result",
                },
            ])
            display(checked_in_report_note)
            """
        ),
        code(
            """
            if REGENERATE_FIGURES:
                subprocess.run([sys.executable, str(ROOT / "paper" / "scripts" / "make_figures.py")], cwd=ROOT, check=True)
            else:
                print("Using checked-in figures. Set REGENERATE_FIGURES=True to rebuild them.")

            figure_names = [
                "fig_method_pipeline",
                "fig_step200_math_zebra",
                "fig_ucb_score_drift",
                "fig_initial_accuracy_profile",
                "fig_rebucket_composition",
                "fig_inferred_transition_matrix",
                "fig_toy_rebucket_guardrails",
            ]

            for name in figure_names:
                png_path = ROOT / "paper" / "figures" / f"{name}.png"
                display(Markdown(f"### {name}"))
                display(Image(filename=str(png_path)))
            """
        ),
        code(
            """
            if RERUN_TOY_SIMULATION:
                subprocess.run([sys.executable, str(ROOT / "scripts" / "toy_simulation_rebucket.py")], cwd=ROOT, check=True)
            else:
                print("Using checked-in toy-simulation artifacts. Set RERUN_TOY_SIMULATION=True to rerun them.")

            toy_summary = json.loads((ROOT / "artifacts" / "toy_rebucket_summary.json").read_text(encoding="utf-8"))
            display(pd.DataFrame(toy_summary["scenarios"]))
            display(Image(filename=str(ROOT / "artifacts" / "toy_rebucket_simulation.png")))
            """
        ),
        md(
            """
            ## Final Takeaway

            This notebook intentionally separates **heavy training**, **replayable evaluation**, and **checked-in analysis**.
            That matches the reality of the project:

            - the repository already contains the important result tables, diagnostic traces, and paper figures;
            - the notebook can bootstrap the repo and inspect those assets from a single file;
            - checkpoint-based evaluation is supported through links;
            - and full RL retraining remains available as a documented code path for a compatible multi-GPU environment.

            In short, the notebook is meant to be both a readable submission artifact and a faithful index to the complete project workflow.
            """
        ),
    ]
    return nb


def main() -> None:
    nb = build_notebook()
    OUT.write_text(nbf.writes(nb), encoding="utf-8")
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
