"""从 ModelScope 多线程分块下载模型（绕开单连接限速）。

用法：
    python tools/ms_download.py --repo Qwen/Qwen3.5-4B --out ~/models/Qwen3.5-4B

原理：
* ModelScope 的 repo 端点会 302 到带签名、易过期的 CDN 链接；
  因此每个分块都重新请求一次 repo 端点（自动跟随 302），拿到新鲜签名。
* 支持 HTTP Range，分块并发下载，断点续传（已完成的块会跳过）。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

FILES_API = "https://modelscope.cn/api/v1/models/{repo}/repo/files?Revision={rev}"
FILE_API = "https://modelscope.cn/api/v1/models/{repo}/repo?Revision={rev}&FilePath={path}"
SMALL_FILE_LIMIT = 64 * 1024 * 1024


def list_files(repo: str, rev: str) -> list[tuple[str, int]]:
    url = FILES_API.format(repo=repo, rev=rev)
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = json.load(resp)
    files = data["Data"]["Files"]
    return [(f["Name"], int(f.get("Size") or 0)) for f in files]


def _request(path: str, repo: str, rev: str, byte_range: str | None = None) -> urllib.request.Request:
    url = FILE_API.format(repo=repo, rev=rev, path=urllib.parse.quote(path))
    headers = {"User-Agent": "curl/8.0"}
    if byte_range:
        headers["Range"] = f"bytes={byte_range}"
    return urllib.request.Request(url, headers=headers)


def fetch_range(repo: str, rev: str, path: str, start: int, end: int, attempts: int = 6) -> bytes:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            req = _request(path, repo, rev, byte_range=f"{start}-{end - 1}")
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            if len(data) != end - start:
                raise IOError(f"短读：期望 {end - start}，得到 {len(data)}")
            return data
        except Exception as error:  # noqa: BLE001 - 下载器需要兜底重试
            last_error = error
            time.sleep(min(2 ** attempt, 15))
    raise RuntimeError(f"分块 {start}-{end} 下载失败：{last_error}")


def download_big(repo: str, rev: str, path: str, size: int, out: Path, workers: int, chunk_mb: int) -> None:
    parts_dir = out.with_suffix(out.suffix + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    chunk = chunk_mb * 1024 * 1024
    ranges = [(start, min(start + chunk, size)) for start in range(0, size, chunk)]

    pending = [r for r in ranges if not (parts_dir / f"{r[0]:012d}").exists()]
    if pending:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(fetch_range, repo, rev, path, start, end): (start, end)
                for start, end in pending
            }
            done = 0
            for future in as_completed(futures):
                start, end = futures[future]
                (parts_dir / f"{start:012d}").write_bytes(future.result())
                done += 1
                if done % 10 == 0 or done == len(pending):
                    print(f"    {path}: {done}/{len(pending)} 块完成", flush=True)

    with open(out, "wb") as target:
        for start, _ in ranges:
            with open(parts_dir / f"{start:012d}", "rb") as part:
                shutil.copyfileobj(part, target)
    shutil.rmtree(parts_dir)


def download_small(repo: str, rev: str, path: str, out: Path) -> None:
    req = _request(path, repo, rev)
    with urllib.request.urlopen(req, timeout=300) as resp, open(out, "wb") as target:
        shutil.copyfileobj(resp, target)


def download(repo: str, out_dir: Path, rev: str, workers: int, chunk_mb: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = list_files(repo, rev)
    for name, size in files:
        out = out_dir / os.path.basename(name)
        if out.exists() and out.stat().st_size == size and size > 0:
            print(f"[skip] {name}", flush=True)
            continue
        print(f"[get ] {name} ({size / 1024 / 1024:.1f} MB)", flush=True)
        if size > SMALL_FILE_LIMIT:
            download_big(repo, rev, name, size, out, workers, chunk_mb)
        else:
            download_small(repo, rev, name, out)
    print(f"[done] 已下载到 {out_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="ModelScope 多线程模型下载器")
    parser.add_argument("--repo", required=True, help="例如 Qwen/Qwen3.5-4B")
    parser.add_argument("--out", required=True, help="输出目录")
    parser.add_argument("--rev", default="master")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--chunk-mb", type=int, default=16)
    args = parser.parse_args()
    download(args.repo, Path(args.out).expanduser(), args.rev, args.workers, args.chunk_mb)


if __name__ == "__main__":
    main()
