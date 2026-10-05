# -*- coding: utf-8 -*-
"""用 GitHub Git Data API 把 gh-publish/ 全部文件推送到仓库（无需 git，带重试与实时日志）。

用法：python push_to_github.py [--repo justin-cen/text-represented-values] [--branch main]
"""
import argparse
import base64
import io
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
TOKEN_FILE = Path(r"E:\culture-difference\.secrets\gh-token-new.txt")
SRC = Path(r"E:\culture-difference\gh-publish")
API = "https://api.github.com"
SKIP_DIRS = {".git", "__pycache__"}
TIMEOUT = 180
RETRIES = 4


def token():
    return TOKEN_FILE.read_text(encoding="utf-8").strip()


def req(method, url, tok, body=None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    last = None
    for attempt in range(1, RETRIES + 1):
        r = urllib.request.Request(url, data=data, method=method, headers={
            "Authorization": f"Bearer {tok}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "dsh-agent",
            "Content-Type": "application/json",
        })
        try:
            with urllib.request.urlopen(r, timeout=TIMEOUT) as resp:
                raw = resp.read().decode("utf-8")
                return resp.status, (json.loads(raw) if raw else {})
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8") or "{}"
            if e.code >= 500 and attempt < RETRIES:
                last = f"HTTP {e.code}"
                time.sleep(2 * attempt)
                continue
            return e.code, json.loads(detail or "{}")
        except Exception as e:                      # 超时/连接错误 → 重试
            last = f"{type(e).__name__}"
            if attempt < RETRIES:
                print(f"    （{last}，第 {attempt} 次重试…）")
                time.sleep(3 * attempt)
                continue
            return 0, {"message": last}
    return 0, {"message": last}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="justin-cen/text-represented-values")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--message", default="Release: lexicon, pipeline, aggregate results, figures, manuscripts")
    a = ap.parse_args()
    tok = token()

    st, repo = req("GET", f"{API}/repos/{a.repo}", tok)
    if st != 200:
        print(f"✗ 无法访问 {a.repo}: HTTP {st} {repo.get('message')}")
        return 1
    print(f"✓ 仓库: {repo['full_name']} | push={repo.get('permissions',{}).get('push')}")

    files = [p for p in sorted(SRC.rglob("*"))
             if p.is_file() and not any(part in SKIP_DIRS for part in p.parts)]
    print(f"待推送文件: {len(files)} 个")

    # 空仓库先建初始提交
    st, _ = req("GET", f"{API}/repos/{a.repo}/git/ref/heads/{a.branch}", tok)
    if st != 200:
        seed = SRC / "README.md"
        print("  → 空仓库，创建初始提交…")
        st, out = req("PUT", f"{API}/repos/{a.repo}/contents/README.md", tok,
                      {"message": "Initialize repository",
                       "content": base64.b64encode(seed.read_bytes()).decode("ascii"),
                       "branch": a.branch})
        if st not in (200, 201):
            print(f"✗ 初始化失败: HTTP {st} {out.get('message')}")
            return 1
        print("  ✓ 初始提交完成")

    # 建 blob
    entries = []
    for i, p in enumerate(files, 1):
        rel = p.relative_to(SRC).as_posix()
        st, blob = req("POST", f"{API}/repos/{a.repo}/git/blobs", tok,
                       {"content": base64.b64encode(p.read_bytes()).decode("ascii"),
                        "encoding": "base64"})
        if st not in (200, 201):
            print(f"✗ blob 失败 {rel}: HTTP {st} {blob.get('message')}")
            return 1
        entries.append({"path": rel, "mode": "100644", "type": "blob", "sha": blob["sha"]})
        if i % 10 == 0 or i == len(files):
            print(f"  blobs {i}/{len(files)}")

    # tree → commit → ref
    st, tree = req("POST", f"{API}/repos/{a.repo}/git/trees", tok, {"tree": entries})
    if st not in (200, 201):
        print(f"✗ tree 失败: HTTP {st} {tree.get('message')}")
        return 1
    print(f"✓ tree {tree['sha'][:10]}")

    st, ref = req("GET", f"{API}/repos/{a.repo}/git/ref/heads/{a.branch}", tok)
    parents = [ref["object"]["sha"]] if st == 200 else []

    st, commit = req("POST", f"{API}/repos/{a.repo}/git/commits", tok,
                     {"message": a.message, "tree": tree["sha"], "parents": parents})
    if st not in (200, 201):
        print(f"✗ commit 失败: HTTP {st} {commit.get('message')}")
        return 1
    print(f"✓ commit {commit['sha'][:10]}")

    st, _ = req("PATCH", f"{API}/repos/{a.repo}/git/refs/heads/{a.branch}", tok,
                {"sha": commit["sha"], "force": True})
    if st not in (200, 201):
        st2, _ = req("POST", f"{API}/repos/{a.repo}/git/refs", tok,
                     {"ref": f"refs/heads/{a.branch}", "sha": commit["sha"]})
        if st2 not in (200, 201):
            print(f"✗ ref 失败: HTTP {st}/{st2}")
            return 1
    print(f"\n✅ 推送完成: https://github.com/{a.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
