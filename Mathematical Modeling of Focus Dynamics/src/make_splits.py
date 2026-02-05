from pathlib import Path
import shutil
import random
import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def yload(p: Path):
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def list_group_dirs(cls_dir: Path):
    groups = [d for d in cls_dir.iterdir() if d.is_dir()]
    groups.sort(key=lambda x: x.name.lower())
    return groups


def list_imgs(dir_: Path):
    files = [p for p in dir_.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXT]
    files.sort(key=lambda x: x.name.lower())
    return files


def choose_group_split(groups, tr_ratio, va_ratio, te_ratio, seed):
    g = list(groups)
    rng = random.Random(seed)
    rng.shuffle(g)
    n = len(g)
    if n == 0:
        return {"train": [], "val": [], "test": []}

    n_te = int(round(n * te_ratio))
    n_va = int(round(n * va_ratio))

    if n >= 3:
        n_te = max(1, n_te)
        n_va = max(1, n_va)
    elif n == 2:
        n_te = 1
        n_va = 0
    else:
        n_te = 0
        n_va = 0

    if n - (n_te + n_va) < 1:
        if n_va > 0:
            n_va -= 1
        elif n_te > 0:
            n_te -= 1

    n_te = max(0, min(n_te, n))
    n_va = max(0, min(n_va, n - n_te))
    n_tr = n - n_te - n_va

    test_groups = g[:n_te]
    val_groups = g[n_te: n_te + n_va]
    train_groups = g[n_te + n_va:]

    return {"train": train_groups, "val": val_groups, "test": test_groups}


def copy_group_images(group_dir: Path, dest_dir: Path, prefix: str, cap: int | None, seed: int):
    files = list_imgs(group_dir)
    if cap is not None and cap > 0 and len(files) > cap:
        rng = random.Random(seed)
        rng.shuffle(files)
        files = files[:cap]

    dest_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        out_name = f"{prefix}__{f.name}"
        shutil.copy2(f, dest_dir / out_name)

    return len(files), len(list_imgs(group_dir))


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = yload(root / "config.yaml")

    raw = root / "data" / "raw"
    out = root / "data" / "splits"

    tr = float(cfg["split"]["train"])
    va = float(cfg["split"]["val"])
    te = float(cfg["split"]["test"])
    if abs((tr + va + te) - 1.0) > 1e-6:
        raise ValueError("train + val + test must sum to 1.0")

    seed = int(cfg.get("seed", 7))

    cap_train_per_group = cfg.get("cap_train_per_group", 200)
    if cap_train_per_group is not None:
        cap_train_per_group = int(cap_train_per_group)

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    print("=== GROUP SPLIT (by ongle folders) ===")
    print(f"cap_train_per_group: {cap_train_per_group}")

    for cls in cfg["classes"]:
        cls_dir = raw / cls
        if not cls_dir.exists():
            raise ValueError(f"Missing class folder: {cls_dir}")

        groups = list_group_dirs(cls_dir)
        if len(groups) == 0:
            raise ValueError(f"No group folders found in: {cls_dir}")

        cls_seed = seed + (abs(hash(cls)) % 10_000)

        split_groups = choose_group_split(groups, tr, va, te, cls_seed)

        print(f"\n[{cls}] groups={len(groups)} | "
              f"train_groups={len(split_groups['train'])} val_groups={len(split_groups['val'])} test_groups={len(split_groups['test'])}")

        for split_name, gdirs in split_groups.items():
            dst = out / split_name / cls
            dst.mkdir(parents=True, exist_ok=True)

            used_total = 0
            real_total = 0

            for gd in gdirs:
                cap = cap_train_per_group if split_name == "train" else None
                used, real = copy_group_images(
                    gd,
                    dst,
                    prefix=gd.name,
                    cap=cap,
                    seed=cls_seed + (abs(hash(gd.name)) % 10_000)
                )
                used_total += used
                real_total += real

            print(f"  - {split_name:>5}: images_used={used_total} (raw_total_in_groups={real_total})")

    print("\nDone. Splits created in: data/splits/{train,val,test}/<class>/")


if __name__ == "__main__":
    main()
