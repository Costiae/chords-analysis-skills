#!/usr/bin/env python3
"""远程模型拉取与版本管理(统一入口)。

设计目标(与"实时更新 + 无网降级"对齐):
  1. 所有远程权重走 manifest(名称/URL/镜像路径/sha256/大小/许可),不散落在代码里
  2. 下载支持镜像:优先环境变量 CHORDS_HF_MIRROR 或 HF_ENDPOINT
  3. 下载后强制 sha256 校验,原子落盘(先 .part 再 rename)
  4. 断点续传(resume)+ 超时重试
  5. `--check-updates`:只发 HEAD 请求比对 ETag/Last-Modified,不下载
  6. 任何失败都不抛给管线:返回 None,调用方按既有规则降级

用法:
  python models.py --list
  python models.py --fetch harmonix-all
  python models.py --fetch-all
  python models.py --check-updates
  python models.py --status

环境变量:
  CHORDS_MODEL_DIR    缓存目录(默认 ~/.cache/chords-analysis/models)
  CHORDS_HF_MIRROR    HF 镜像(默认 https://hf-mirror.com);设 none 可强制官方源
"""
import argparse, hashlib, json, os, sys, time, urllib.error, urllib.request
from pathlib import Path
from typing import Optional

SKILL_DIR = Path(__file__).resolve().parent.parent
MANIFEST = SKILL_DIR / "models" / "manifest.json"
DEFAULT_CACHE = Path(os.environ.get("CHORDS_MODEL_DIR",
                                    Path.home() / ".cache" / "chords-analysis" / "models"))
STATE_FILE = "state.json"


def cache_dir() -> Path:
    d = DEFAULT_CACHE
    d.mkdir(parents=True, exist_ok=True)
    return d


def dest_for(artifact: dict, model: dict) -> Path:
    """权重必须落在实际加载位置:allin1 读 torch hub 缓存,其它读本地缓存目录。"""
    sub = model.get("cache_subdir")
    if sub == "torch/hub/checkpoints":
        try:
            import torch
            d = Path(torch.hub.get_dir()) / "checkpoints"
        except Exception:
            d = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
        d.mkdir(parents=True, exist_ok=True)
        return d / artifact["file"]
    return cache_dir() / artifact["file"]


def load_manifest():
    if not MANIFEST.exists():
        return {"schema": 1, "models": {}}
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def mirror_base() -> Optional[str]:
    v = os.environ.get("CHORDS_HF_MIRROR") or os.environ.get("HF_ENDPOINT")
    if v and v.lower() == "none":
        return None
    return v or "https://hf-mirror.com"


def candidate_urls(url: str) -> list:
    """官方源 + 镜像源(HF 链接可映射到镜像)。"""
    urls = []
    mb = mirror_base()
    if mb and "huggingface.co" in url:
        urls.append(url.replace("https://huggingface.co", mb.rstrip("/")))
    urls.append(url)
    if mb and "huggingface.co" not in url:
        pass
    return urls


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(urls, dest: Path, expect_sha=None, expect_size=None,
          timeout=120, retries=3) -> bool:
    """下载到 dest(断点续传 + 校验 + 原子重命名)。全失败返回 False。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if expect_sha and sha256_of(dest) == expect_sha:
            return True
        if not expect_sha:
            return True
        dest.unlink()          # 校验失败,重下

    part = dest.with_suffix(dest.suffix + ".part")
    for url in urls:
        for attempt in range(1, retries + 1):
            try:
                pos = part.stat().st_size if part.exists() else 0
                req = urllib.request.Request(url)
                if pos:
                    req.add_header("Range", f"bytes={pos}-")
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    mode = "ab" if pos and r.status == 206 else "wb"
                    with part.open(mode) as f:
                        while True:
                            chunk = r.read(1 << 20)
                            if not chunk:
                                break
                            f.write(chunk)
                if expect_size and part.stat().st_size != expect_size:
                    raise ValueError(f"size mismatch {part.stat().st_size} != {expect_size}")
                if expect_sha:
                    got = sha256_of(part)
                    if got != expect_sha:
                        part.unlink(missing_ok=True)
                        raise ValueError(f"sha256 mismatch {got[:12]}...")
                part.replace(dest)
                return True
            except Exception as e:
                print(f"  [retry {attempt}/{retries}] {url}: {type(e).__name__}: {e}",
                      file=sys.stderr)
                time.sleep(min(2 ** attempt, 10))
    return False


def artifacts(model: dict) -> list:
    return [a for a in model.get("artifacts", [])
            if a.get("kind") in (None, "checkpoint", "hf_snapshot")]


def fetch_hf_snapshot(entry: dict, model: dict) -> bool:
    """HF 仓库快照(需要代码+权重一起拉,如 trust_remote_code 模型)。"""
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        print(f"  huggingface_hub unavailable: {e}", file=sys.stderr)
        return False
    dest = DEFAULT_CACHE / entry.get("file", model.get("cache_subdir", "snapshot"))
    rev = entry.get("revision") or os.environ.get("CHORDS_BTC_REVISION")
    kw = dict(repo_id=entry["repo"], revision=rev, local_dir=str(dest),
              allow_patterns=entry.get("allow_patterns") or ["*.py", "*.json", "*.pt"],
              max_workers=4)
    ep = mirror_base()
    if ep:
        kw["endpoint"] = ep
    try:
        snapshot_download(**kw)
        return True
    except Exception as e:
        print(f"  snapshot failed: {type(e).__name__}: {e}", file=sys.stderr)
        return False


def fetch_model(name: str, model: dict, quiet=False) -> bool:
    ok_all = True
    for a in artifacts(model):
        if a.get("kind") == "hf_snapshot":
            if not quiet:
                print(f"  snapshot {a['repo']} ...")
            if not fetch_hf_snapshot(a, model):
                ok_all = False
            continue
        dest = dest_for(a, model)
        if dest.exists() and (not a.get("sha256") or sha256_of(dest) == a["sha256"]):
            if not quiet:
                print(f"  cached: {a['file']}")
            continue
        if not quiet:
            print(f"  fetching {a['file']} ({a.get('size_mb', '?')} MB) ...")
        ok = fetch(candidate_urls(a["url"]), dest, a.get("sha256"), a.get("size"))
        if not ok:
            ok_all = False
            print(f"  FAILED: {a['file']}", file=sys.stderr)
    return ok_all


def state_path() -> Path:
    return cache_dir() / STATE_FILE


def load_state() -> dict:
    p = state_path()
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_state(st: dict):
    state_path().write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")


def check_updates(verbose=True):
    """HEAD 请求比对 ETag/Last-Modified;发现变化只提示,不自动下载。"""
    st = load_state()
    manifest = load_manifest()
    changes = []
    for name, model in manifest.get("models", {}).items():
        for a in artifacts(model):
            urls = candidate_urls(a["url"])
            head = None
            for url in urls:
                try:
                    req = urllib.request.Request(url, method="HEAD")
                    with urllib.request.urlopen(req, timeout=20) as r:
                        head = {"etag": r.headers.get("ETag"),
                                "last_modified": r.headers.get("Last-Modified"),
                                "size": r.headers.get("Content-Length")}
                    break
                except Exception:
                    continue
            if not head:
                changes.append((name, a["file"], "unreachable"))
                continue
            key = f"{name}/{a['file']}"
            prev = st.get(key, {})
            if head != {k: prev.get(k) for k in head}:
                if prev:
                    changes.append((name, a["file"], "changed"))
                st[key] = head
    save_state(st)
    if verbose:
        if not changes:
            print("up to date (no remote changes detected)")
        for name, f, why in changes:
            print(f"[{why}] {name}: {f}")
    return changes


def status():
    manifest = load_manifest()
    print(f"cache dir: {cache_dir()}")
    print(f"mirror:    {mirror_base() or '(official only)'}")
    for name, model in manifest.get("models", {}).items():
        arts = artifacts(model)
        present = []
        for a in arts:
            if a.get("kind") == "hf_snapshot":
                d = DEFAULT_CACHE / a.get("file", model.get("cache_subdir", "snapshot"))
                if d.exists() and any(d.iterdir()):
                    present.append(a)
            elif dest_for(a, model).exists():
                present.append(a)
        optional = " (optional)" if model.get("optional") else ""
        print(f"- {name}{optional}: {len(present)}/{len(arts)} artifacts"
              f" | {model.get('purpose', '')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--fetch")
    ap.add_argument("--fetch-all", action="store_true")
    ap.add_argument("--check-updates", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    manifest = load_manifest()

    if args.list:
        for name, m in manifest.get("models", {}).items():
            print(f"{name}: {m.get('purpose', '')}")
            for a in artifacts(m):
                print(f"   {a['file']}  {a.get('size_mb', '?')} MB  {a['url']}")
    elif args.fetch or args.fetch_all:
        names = [args.fetch] if args.fetch else list(manifest.get("models", {}))
        for n in names:
            m = manifest["models"].get(n)
            if not m:
                print(f"unknown model: {n}", file=sys.stderr)
                continue
            print(f"[{n}]")
            ok = fetch_model(n, m)
            print(f"  {'OK' if ok else 'INCOMPLETE'}")
    elif args.check_updates:
        check_updates()
    else:
        status()


if __name__ == "__main__":
    main()
