#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

WORKSPACE_ROOT = Path("./")
# CONF_DIR = WORKSPACE_ROOT / "geo-synthetic/geometry-synthetic/geo_position/conf_1"
CONF_DIR = WORKSPACE_ROOT / "geo-synthetic/geometry-synthetic/geo_position/conf_2"
DEST_BASE = WORKSPACE_ROOT / "data/eval/full/geo_problem/conf_2"
LEGACY_DIR = WORKSPACE_ROOT / "data/eval/full/geo_problem/legacy"

# Conf_1
# POS_ONLY_COUNT = 200
# COUNT_ONLY_COUNT = 200
# OVERLAP_COUNT = 100

# Conf_2
POS_ONLY_COUNT = 400
COUNT_ONLY_COUNT = 400
OVERLAP_COUNT = 100
TOTAL_SCENES_REQUIRED = POS_ONLY_COUNT + COUNT_ONLY_COUNT + OVERLAP_COUNT
RNG_SEED = 2025


def _load_description(desc_path: Path) -> Dict[str, Any]:
    with desc_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _choose_problem(entry_list: List[Dict[str, Any]], rng: random.Random) -> Tuple[str, str]:
    if not entry_list:
        raise ValueError("Template list is empty, cannot sample problem")
    entry = rng.choice(entry_list)
    problem = entry["problem"]
    answer = entry["answer"]
    return problem, answer


def _prepare_output_dirs() -> Dict[str, Dict[str, Path]]:
    outputs = {}
    for split in ("positioning", "counting"):
        split_dir = DEST_BASE / split
        images_dir = split_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        # Clean old images and test files, ensure reproducibility
        if any(images_dir.iterdir()):
            shutil.rmtree(images_dir)
            images_dir.mkdir(parents=True, exist_ok=True)
        test_path = split_dir / "test.jsonl"
        if test_path.exists():
            test_path.unlink()
        outputs[split] = {
            "split_dir": split_dir,
            "images_dir": images_dir,
            "test_path": test_path,
        }
    return outputs


def _collect_candidates(rng: random.Random) -> List[Tuple[Path, Dict[str, Any]]]:
    if not CONF_DIR.exists():
        raise FileNotFoundError(f"Directory not found: {CONF_DIR}")
    candidates = [p for p in CONF_DIR.iterdir() if p.is_dir()]
    rng.shuffle(candidates)
    selected: List[Tuple[Path, Dict[str, Any]]] = []
    for scene_dir in candidates:
        desc_path = scene_dir / "description.json"
        img_path = scene_dir / "3D_view.png"
        if not desc_path.is_file() or not img_path.is_file():
            continue
        desc = _load_description(desc_path)
        query = desc.get("query") or {}
        template_1 = query.get("template_1") or []
        template_2 = query.get("template_2") or []
        if not template_1 or not template_2:
            continue
        selected.append((scene_dir, desc))
        if len(selected) >= TOTAL_SCENES_REQUIRED:
            break
    if len(selected) < TOTAL_SCENES_REQUIRED:
        raise RuntimeError(
            f"Valid samples insufficient {TOTAL_SCENES_REQUIRED} found, only found {len(selected)}"
        )
    return selected


def _dump_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")


def main() -> None:
    rng = random.Random(RNG_SEED)
    outputs = _prepare_output_dirs()
    selected_scenes = _collect_candidates(rng)

    positioning_records: List[Dict[str, Any]] = []
    counting_records: List[Dict[str, Any]] = []
    pos_idx = 1
    count_idx = 1

    def add_positioning(scene_dir: Path, desc: Dict[str, Any]) -> None:
        nonlocal pos_idx
        problem_pos, answer_pos = _choose_problem(desc["query"].get("template_1") or [], rng)
        pos_id = f"positioning_conf1_{pos_idx:04d}"
        img_src = scene_dir / "3D_view.png"
        shutil.copy(img_src, outputs["positioning"]["images_dir"] / f"{pos_id}.png")
        positioning_records.append(
            {
                "id": pos_id,
                "problem": problem_pos,
                "answer": answer_pos,
                "images": [f"./images/{pos_id}.png"],
            }
        )
        pos_idx += 1

    def add_counting(scene_dir: Path, desc: Dict[str, Any]) -> None:
        nonlocal count_idx
        problem_count, answer_count = _choose_problem(
            desc["query"].get("template_2") or [], rng
        )
        count_id = f"counting_conf1_{count_idx:04d}"
        img_src = scene_dir / "3D_view.png"
        shutil.copy(img_src, outputs["counting"]["images_dir"] / f"{count_id}.png")
        counting_records.append(
            {
                "id": count_id,
                "problem": problem_count,
                "answer": answer_count,
                "images": [f"./images/{count_id}.png"],
            }
        )
        count_idx += 1

    pos_only = selected_scenes[:POS_ONLY_COUNT]
    count_only = selected_scenes[POS_ONLY_COUNT : POS_ONLY_COUNT + COUNT_ONLY_COUNT]
    overlap = selected_scenes[
        POS_ONLY_COUNT + COUNT_ONLY_COUNT : TOTAL_SCENES_REQUIRED
    ]

    for scene_dir, desc in pos_only:
        add_positioning(scene_dir, desc)

    for scene_dir, desc in count_only:
        add_counting(scene_dir, desc)

    for scene_dir, desc in overlap:
        add_positioning(scene_dir, desc)
        add_counting(scene_dir, desc)

    _dump_jsonl(positioning_records, outputs["positioning"]["test_path"])
    _dump_jsonl(counting_records, outputs["counting"]["test_path"])

    LEGACY_DIR.mkdir(parents=True, exist_ok=True)
    for scene_dir, _ in selected_scenes[:TOTAL_SCENES_REQUIRED]:
        target = LEGACY_DIR / scene_dir.name
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(scene_dir), str(target))

    print(
        f"Generated {len(positioning_records)} positioning samples and {len(counting_records)} counting samples"
    )
    print(f"positioning test: {outputs['positioning']['test_path']}")
    print(f"counting test: {outputs['counting']['test_path']}")
    print(f"Cut {len(selected_scenes)} original directories to {LEGACY_DIR}")


if __name__ == "__main__":
    main()
