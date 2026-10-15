"""Non-training smoke checks for the curated paper code release."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def literal_constants(path: Path) -> dict:
    constants = {}
    for node in ast.parse(path.read_text(encoding="utf-8-sig")).body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                constants[target.id] = value
            elif isinstance(target, ast.Tuple) and isinstance(value, tuple):
                for item, part in zip(target.elts, value):
                    if isinstance(item, ast.Name):
                        constants[item.id] = part
    return constants


def main() -> None:
    syntax_count = 0
    for path in ROOT.rglob("*.py"):
        compile(path.read_text(encoding="utf-8-sig"), str(path), "exec")
        syntax_count += 1
    manifest = json.loads((ROOT / "release_manifest.json").read_text(encoding="utf-8"))
    for record in manifest["source_files"]:
        actual = hashlib.sha256((ROOT / record["path"]).read_bytes()).hexdigest()
        assert actual == record["release_sha256"], f"Changed release source: {record['path']}"
    scripts = ROOT / "implementations/yolov8_det/scripts"
    cable = literal_constants(scripts / "leakfix_det_pipeline.py")
    assert (cable["EPOCHS"], cable["IMGSZ"], cable["BATCH"], cable["PATIENCE"]) == (30, 640, 4, 0)
    for name in ["q3_neu_projection_pipeline.py", "q3_neu_comparators_pipeline.py",
                 "gc10_external_pipeline.py", "gc10_copypaste_pipeline.py", "gc10_gan_external_pipeline.py"]:
        constants = literal_constants(scripts / name)
        assert constants["BATCH"] == 4, name
        if "EPOCHS" in constants:
            assert constants["EPOCHS"] == 30, name
        if "Q" in constants:
            assert constants["Q"] == 300, name
    prohibited = {".pt", ".pth", ".ckpt", ".safetensors", ".docx", ".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".log"}
    for path in ROOT.rglob("*"):
        if path.is_file():
            assert path.suffix.lower() not in prohibited, f"Excluded artifact present: {path.relative_to(ROOT)}"
    print(json.dumps({"python_syntax_files": syntax_count,
                      "verified_source_hashes": len(manifest["source_files"]),
                      "detector_protocol_constants": "passed",
                      "excluded_artifact_check": "passed",
                      "gpu_or_training_started": False}, indent=2))


if __name__ == "__main__":
    main()
