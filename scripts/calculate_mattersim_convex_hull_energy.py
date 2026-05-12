#!/usr/bin/env python
# coding: utf-8

# # Amm2で提案された最適化構造を評価するノートブック

# In[1]:


from pathlib import Path
import os

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ["PMG_VASP_PSP_DIR"] = str(REPO_ROOT / "pmg_potcars")


# ## MPから構造取得と緩和のコード

# In[ ]:


# In[2]:


import os
import torch
import numpy as np
import pandas as pd
import sqlite3
import hashlib
import gzip
import copy
import multiprocessing as mp
import traceback
import typing
from typing import List, Tuple, Sequence, Optional, Union, Iterable
from tqdm.auto import tqdm

if not hasattr(typing, "NotRequired"):
    try:
        from typing_extensions import NotRequired as _NotRequired
    except ImportError as exc:
        raise ImportError(
            "mp_api/emmet requires typing.NotRequired on Python 3.10. "
            "Install typing_extensions or use Python >=3.11."
        ) from exc
    typing.NotRequired = _NotRequired

from mp_api.client import MPRester
from pymatgen.io.ase import AseAtomsAdaptor

# from ase.io import write
import ase
import ase.constraints

if not hasattr(ase.constraints, "Filter"):
    from ase.filters import Filter as _AseFilter

    ase.constraints.Filter = _AseFilter

from mattersim.forcefield import MatterSimCalculator
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility

from mattersim.applications.relax import Relaxer

from IPython.display import display, clear_output


# In[3]:


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 出力先フォルダ
OUTDIR = Path("mp_relaxed")
OUTDIR.mkdir(exist_ok=True)

# Repository-relative default data location (avoid hard-coded absolute paths)
DEFAULT_RELAX_DB_PATH = (
    REPO_ROOT / "data" / "mp_mattersim_relaxed" / "relax_results.json"
)
GLOBAL_RELAX_DIR = DEFAULT_RELAX_DB_PATH.parent
DEFAULT_CACHE_DIR = Path.cwd() / "cache"
FETCH_EHULL_DB_THRES = 0.3
CREATE_DFT_NEEDED_THRES = 0.1


def _resolve_relax_record_path(path_value: Union[str, Path]) -> Path:
    """Resolve stored POSCAR paths across different workspace roots."""
    path = Path(path_value)
    if path.exists():
        return path
    path_str = str(path_value)
    marker = f"{os.sep}SMOACS{os.sep}"
    if marker in path_str:
        suffix = path_str.split(marker, 1)[1]
        candidate = REPO_ROOT / suffix
        if candidate.exists():
            return candidate
    return path


from pymatgen.core import Lattice, Structure as PmgStructure


from itertools import combinations


def make_optimade_filter(
    chemsys: str,
    max_nelements: int | None = None,
    require_3d: bool = True,
    prefer_has_only: bool = True,
) -> str:
    """
    任意の化学系から、凸包計算などに使いやすい OPTIMADE filter を生成する。

    例:
      chemsys="Li-O"          -> 単体 Li, O と二元 Li-O を含める
      chemsys="Li-Fe-P-O"     -> 単体/二元/三元/四元 (要素は指定集合のみに限定)

    引数:
      chemsys: "Li-O", "Fe-S", "Li-Fe-P-O" など（区切りは '-' または ',' を許容）
      max_nelements: 取得する最大元素数（None のとき = 系の元素数）
      require_3d: True のとき `nperiodic_dimensions=3` を付与（バルクのみ）
      prefer_has_only: True なら `HAS ONLY` を使った短い filter を生成
                       False なら OR 展開で移植性の高い filter を生成

    戻り値:
      OPTIMADE の filter 文字列
    """
    # 系の元素リスト（重複除去・順序保持）
    elems = [e.strip() for e in chemsys.replace(",", "-").split("-") if e.strip()]
    elems = list(dict.fromkeys(elems))
    if not elems:
        raise ValueError("chemsys が空です。例: 'Li-O'")

    m = len(elems)
    if max_nelements is None or max_nelements > m:
        max_nelements = m

    # "Li","O","Fe" のようなクォート付きカンマ区切り
    def quote_list(items):
        return ",".join(f'"{x}"' for x in items)

    if prefer_has_only:
        # 短い表現（対応プロバイダ限定だが、Alexandria では利用可）
        core = f"(nelements<={max_nelements} AND elements HAS ONLY {quote_list(elems)})"
    else:
        # 移植性の高い OR 展開（HAS ONLY 非対応プロバイダ向け）
        disj = []
        for k in range(1, max_nelements + 1):
            for combo in combinations(elems, k):
                disj.append(f"(nelements={k} AND elements HAS ALL {quote_list(combo)})")
        core = "(" + " OR ".join(disj) + ")"

    if require_3d:
        return f"(nperiodic_dimensions=3) AND {core}"
    return core


# In[ ]:


# In[6]:


import itertools
import concurrent.futures as cf
from urllib.parse import urljoin
import requests
from urllib3.util.retry import Retry
import time

ALEX_RETRY = Retry(
    total=8,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
)

BASE = "https://alexandria.icams.rub.de/pbe/v1/"
STRUCTURES = urljoin(BASE, "structures")

# --- 1) 軽量フィールドのみ（凸包と簡易解析に必要な分だけ）
LIGHT_FIELDS = [
    "id",
    "immutable_id",
    "elements",
    "nelements",
    "chemical_formula_reduced",
    "_alexandria_formation_energy_per_atom",
    "_alexandria_hull_distance",
    "_alexandria_band_gap",
]

# --- 2) 後で構造が欲しいときにだけ取りに行く
STRUCTURE_FIELDS = [
    "id",
    "immutable_id",
    "lattice_vectors",
    "cartesian_site_positions",
    "species_at_sites",
]


def make_filters_for_chemsys(elems, require_3d=True, max_k=None, only_known=True):
    """単体～max_k 元までの OR クエリを、(nelements, 組合せ)ごとに分割して返す。"""
    elems = list(dict.fromkeys(e.strip() for e in elems))
    if max_k is None:
        max_k = len(elems)

    head = "(nperiodic_dimensions=3) AND " if require_3d else ""
    tail = " AND _alexandria_formation_energy_per_atom IS KNOWN" if only_known else ""

    filters = []
    for k in range(1, max_k + 1):
        for combo in itertools.combinations(elems, k):
            quoted = ",".join(f'"{x}"' for x in combo)
            f = f"{head}(nelements={k} AND elements HAS ALL {quoted}{tail})"
            filters.append(f)
    return filters


def fetch_all_pages(session, url, filter_str, fields, page_limit=200, timeout=(5, 60)):
    """filter に一致する全ページをたどって data を返す（軽量 & 確実版）。"""
    params = {
        "filter": filter_str,
        "response_fields": ",".join(fields),
        "page_limit": page_limit,
    }
    out = []
    next_url = url
    next_params = params
    while next_url:
        r = session.get(next_url, params=next_params, timeout=timeout)
        r.raise_for_status()
        js = r.json()
        out.extend(js.get("data", []))
        next_url = js.get("links", {}).get("next")
        next_params = None  # next は完全 URL なので params は不要
    return out


def gather_chemsys_light(chemsys: str, max_workers=8, page_limit=200):
    """軽量フィールドだけを並列収集（単体～系の元素数まで）。"""
    elems = chemsys.replace(",", "-").split("-")
    flts = make_filters_for_chemsys(elems, require_3d=True, max_k=None, only_known=True)

    # セッション & リトライ設定
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=max_workers,
        pool_maxsize=max_workers,
        max_retries=ALEX_RETRY,
    )
    session.mount("https://", adapter)

    results = []
    # 並列に各サブクエリを取得
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [
            ex.submit(fetch_all_pages, session, STRUCTURES, f, LIGHT_FIELDS, page_limit)
            for f in flts
        ]
        for fut in cf.as_completed(futs):
            results.extend(fut.result())

    # たまに重複（同一 id）が混ざるので除重
    dedup = {}
    for d in results:
        dedup[d["id"]] = d
    return list(dedup.values())


def fetch_structures_by_ids(ids, page_limit=200):
    """必要になった id だけをまとめて構造取得（IDごとに filter）。"""
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=10,
        pool_maxsize=10,
        max_retries=ALEX_RETRY,
    )
    session.mount("https://", adapter)

    out = []
    # 過剰リクエスト回避のため小分け
    chunk = 50
    for i in range(0, len(ids), chunk):
        sub = ids[i : i + chunk]
        # id IN (...) がないので OR でまとめる
        ors = " OR ".join(f'(id="{_id}")' for _id in sub)
        flt = f"({ors})"
        out.extend(
            fetch_all_pages(
                session, STRUCTURES, flt, STRUCTURE_FIELDS, page_limit=page_limit
            )
        )
    # id で辞書化
    return {d["id"]: d for d in out}


def entry_to_pmgstructure(struct_entry: dict) -> PmgStructure:
    """OPTIMADE structures エントリ -> pymatgen.Structure"""
    a = struct_entry["attributes"]
    lattice = Lattice(a["lattice_vectors"])
    coords = a["cartesian_site_positions"]
    species = a["species_at_sites"]  # 例: ["Li","Li","O",...]
    return PmgStructure(lattice, species, coords, coords_are_cartesian=True)


# In[7]:


from urllib.parse import urljoin

from pymatgen.core import Structure
import re


# ---------------------------
# 内部ユーティリティ
# ---------------------------
def _make_filters_for_chemsys(
    chemsys: Union[str, Sequence[str]],
    *,
    require_3d: bool = True,
    max_k: Optional[int] = None,
    only_known: bool = True,
) -> list[str]:
    """Build OPTIMADE filter strings for all subsets up to `max_k`.

    This function expands a chemical system (e.g. "Li-O-Fe-P") into a set of
    filters of the form ``(nelements=k AND elements HAS ALL "A","B",...)`` for
    each k-subset (k=1..max_k). It optionally restricts to 3D periodic entries
    and/or entries with known formation energy.

    Args:
        chemsys: Chemical system as a string like "Li-O" or "Li-Fe-P-O",
            or as a sequence like ["Li", "O"].
        require_3d: If True, prepend ``(nperiodic_dimensions=3) AND``.
        max_k: Maximum order of subsets to generate. If None, use the
            number of elements in `chemsys`.
        only_known: If True, append
            ``AND _alexandria_formation_energy_per_atom IS KNOWN``.

    Returns:
        A list of OPTIMADE filter strings, one per (k, subset).

    Raises:
        ValueError: If `chemsys` is empty or cannot be parsed.
    """
    # Normalize elements from string or sequence
    if isinstance(chemsys, str):
        raw_elems = [e.strip() for e in chemsys.replace(",", "-").split("-")]
    else:
        raw_elems = [str(e).strip() for e in chemsys]

    # Light validation for element symbols: "H", "He", "Li", ...
    elem_pat = re.compile(r"^[A-Z][a-z]?$")
    elems: list[str] = []
    seen = set()
    for e in raw_elems:
        if e and elem_pat.match(e) and e not in seen:
            elems.append(e)
            seen.add(e)

    if not elems:
        raise ValueError("Empty or unparsable `chemsys`. Example: 'Li-O'")

    if max_k is None or max_k > len(elems):
        max_k = len(elems)

    head = "(nperiodic_dimensions=3) AND " if require_3d else ""
    tail = " AND _alexandria_formation_energy_per_atom IS KNOWN" if only_known else ""

    filters: list[str] = []
    for k in range(1, max_k + 1):
        for combo in itertools.combinations(elems, k):
            quoted = ",".join(f'"{x}"' for x in combo)
            filters.append(f"{head}(nelements={k} AND elements HAS ALL {quoted}{tail})")
    return filters


def _fetch_all_pages(
    session: requests.Session,
    url: str,
    filter_str: str,
    fields: Sequence[str],
    *,
    page_limit: int = 200,
    timeout: tuple[float, float] = (5, 60),
) -> list[dict]:
    """filter に一致する全ページを辿って data を返す（確実な全件取得）。"""
    params = {
        "filter": filter_str,
        "response_fields": ",".join(fields),
        "page_limit": page_limit,
    }
    out: list[dict] = []
    next_url, next_params = url, params
    while next_url:
        r = session.get(next_url, params=next_params, timeout=timeout)
        r.raise_for_status()
        js = r.json()
        out.extend(js.get("data", []))
        next_url = js.get("links", {}).get("next")
        next_params = None  # next は完全URL
    return out


def _gather_chemsys_light(
    base_url: str,
    chemsys: Sequence[str] | str,
    *,
    require_3d: bool = True,
    only_known: bool = True,
    max_k: Optional[int] = None,
    max_workers: int = 8,
    page_limit: int = 200,
    max_n: Optional[int] = None,
    ehull_max: Optional[float] = None,
    include_structure: bool = True,
    use_lock: bool = True,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> list[dict]:
    """軽量フィールドのみを（単体～max_k 元）で分割・並列取得し、id で重複除去。"""
    # エンドポイント
    base_v1 = base_url.rstrip("/") + "/v1/"
    structures_url = urljoin(base_v1, "structures")

    # 軽量フィールド（凸包判定等に必要な分だけ）
    LIGHT_FIELDS = [
        "id",
        "immutable_id",
        "elements",
        "nelements",
        "chemical_formula_reduced",
        "_alexandria_formation_energy_per_atom",
        "_alexandria_hull_distance",
        "_alexandria_phase_separation_energy",
        "_alexandria_band_gap",
    ]
    if include_structure:
        LIGHT_FIELDS.extend(
            ["lattice_vectors", "cartesian_site_positions", "species_at_sites"]
        )

    # chemsys を正規化
    if isinstance(chemsys, str):
        elems = chemsys.replace(",", "-").split("-")
    else:
        elems = list(chemsys)

    flts = _make_filters_for_chemsys(
        elems, require_3d=require_3d, max_k=max_k, only_known=only_known
    )

    # セッション & リトライ
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=max_workers,
        pool_maxsize=max_workers,
        max_retries=ALEX_RETRY,
    )
    session.mount("https://", adapter)

    def _fetch_sorted(
        filter_str: str, max_n_limit: int
    ) -> tuple[list[dict], int, Optional[float], str]:
        """Sorted fetch with early stop on ehull_max or max_n_limit. Returns (data, pages, last_eh, status)."""
        params = {
            "filter": filter_str,
            "response_fields": ",".join(LIGHT_FIELDS),
            "page_limit": page_limit,
            "sort": "_alexandria_hull_distance",
        }
        out: list[dict] = []
        next_url: Optional[str] = structures_url
        pages = 0
        status = "ok"
        last_eh: Optional[float] = None
        while next_url and len(out) < max_n_limit:
            resp = session.get(next_url, params=params, timeout=(5, 60))
            resp.raise_for_status()
            js = resp.json()
            data = js.get("data", [])
            out.extend(data)
            params = None  # next は完全URL
            pages += 1
            if data:
                try:
                    last_eh = float(
                        data[-1].get("attributes", {}).get("_alexandria_hull_distance")
                    )
                except Exception:
                    last_eh = None
            if len(out) >= max_n_limit:
                status = "hit_max_n"
                break
            if last_eh is not None and ehull_max is not None and last_eh > ehull_max:
                status = "hit_ehull_max"
                break
            next_url = js.get("links", {}).get("next")
            if not next_url:
                status = "no_next"
        return out[:max_n_limit], pages, last_eh, status

    results: list[dict] = []
    ladder = [800, 1000, 1200, 1500]
    initial_max_n = max_n if max_n is not None else page_limit
    log_path = Path.cwd() / "logs" / "time_logs" / "stage3_light.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _log_attempt(
        filter_str: str,
        attempt: int,
        max_n_used: int,
        fetched: list[dict],
        elapsed: float,
        pages: int,
        status: str,
    ) -> None:
        ehulls: list[float] = []
        for d in fetched:
            try:
                eh = float(d.get("attributes", {}).get("_alexandria_hull_distance"))
                ehulls.append(eh)
            except Exception:
                continue
        last_eh = ehulls[-1] if ehulls else None
        rec = {
            "filter": filter_str,
            "attempt": attempt,
            "max_n": max_n_used,
            "page_limit": page_limit,
            "count": len(fetched),
            "pages": pages,
            "ehull_min": min(ehulls) if ehulls else None,
            "ehull_max_seen": max(ehulls) if ehulls else None,
            "last_eh": last_eh,
            "sorted": (
                all(ehulls[i] <= ehulls[i + 1] for i in range(len(ehulls) - 1))
                if len(ehulls) > 1
                else True
            ),
            "elapsed_s": elapsed,
            "status": status,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=True) + "\n")
                f.flush()
        except Exception:
            pass

    lock_path = _alexandria_lock_path(cache_dir)
    fetch_start = time.time()

    def _run_fetch() -> None:
        nonlocal page_limit
        for f in flts:
            current_max_n = initial_max_n
            attempt = 0
            while True:
                t0 = time.time()
                fetched, pages, last_eh_val, status = _fetch_sorted(f, current_max_n)
                elapsed = time.time() - t0
                _log_attempt(f, attempt, current_max_n, fetched, elapsed, pages, status)
                results.extend(fetched)
                if ehull_max is None or last_eh_val is None or last_eh_val > ehull_max:
                    break
                next_n = next((n for n in ladder if n > current_max_n), None)
                if next_n is None:
                    break
                page_limit = 200  # ladder 以降は固定
                current_max_n = next_n
                attempt += 1

    if use_lock:
        lock_wait_start = time.time()
        with _InterProcessFileLock(lock_path):
            lock_wait = time.time() - lock_wait_start
            _append_timing(
                "alex_lock_wait",
                lock_wait,
                stage="light",
                chemsys=",".join(elems),
                pid=os.getpid(),
            )
            _run_fetch()
    else:
        _run_fetch()

    _append_timing(
        "alex_light_fetch",
        time.time() - fetch_start,
        chemsys=",".join(elems),
        count=len(results),
        max_workers=max_workers,
        pid=os.getpid(),
    )

    dedup: dict[str, dict] = {}
    for d in results:
        dedup[d["id"]] = d
    return list(dedup.values())


def _fetch_structures_by_ids(
    base_url: str,
    ids: list[str],
    *,
    page_limit: int = 200,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> dict[str, dict]:
    """必要な id だけ構造（格子・座標）をまとめて取得して id→entry にして返す。"""
    base_v1 = base_url.rstrip("/") + "/v1/"
    structures_url = urljoin(base_v1, "structures")

    STRUCT_FIELDS = [
        "id",
        "immutable_id",
        "lattice_vectors",
        "cartesian_site_positions",
        "species_at_sites",
    ]

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=10,
        pool_maxsize=10,
        max_retries=ALEX_RETRY,
    )
    session.mount("https://", adapter)

    out: list[dict] = []
    log_path = Path.cwd() / "logs" / "time_logs" / "stage3_structure.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _alexandria_lock_path(cache_dir)
    lock_wait_start = time.time()
    with _InterProcessFileLock(lock_path):
        lock_wait = time.time() - lock_wait_start
        _append_timing(
            "alex_lock_wait",
            lock_wait,
            stage="structure",
            count=len(ids),
            pid=os.getpid(),
        )
        print(f"[Alexandria] Acquired lock for structure fetch: {lock_path}")
        fetch_start = time.time()
        chunk = 50  # まとめすぎるとURLが長くなるので適度に分割
        total_pages = 0
        for i in range(0, len(ids), chunk):
            sub = ids[i : i + chunk]
            ors = " OR ".join(f'(id="{_id}")' for _id in sub)
            flt = f"({ors})"
            out.extend(
                _fetch_all_pages(
                    session, structures_url, flt, STRUCT_FIELDS, page_limit=page_limit
                )
            )
            total_pages += 1
        _append_timing(
            "alex_structure_fetch",
            time.time() - fetch_start,
            count=len(out),
            pid=os.getpid(),
        )
    try:
        rec = {
            "input_ids": len(ids),
            "output_records": len(out),
            "chunk": chunk,
            "page_limit": page_limit,
            "pages": total_pages if ids else 0,
            "elapsed_s": time.time() - fetch_start,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")
    except Exception:
        pass

    return {d["id"]: d for d in out}


def _entry_to_pmgstructure(struct_entry: dict) -> Structure:
    """OPTIMADE structures エントリ -> pymatgen.Structure（依存最小の手組み変換）"""
    a = struct_entry["attributes"]
    lattice = Lattice(a["lattice_vectors"])
    coords = a["cartesian_site_positions"]
    species = a["species_at_sites"]
    return Structure(lattice, species, coords, coords_are_cartesian=True)


# ---------------------------
# 公開関数：fetch_alex_structures
# ---------------------------
def fetch_alex_structures(
    chemsys: Sequence[str],
    *,
    ehull_max: float = FETCH_EHULL_DB_THRES,
    max_n: Optional[int] = None,
    base_url: str = "https://alexandria.icams.rub.de/pbe",  # "…/pbesol" や "…/scan" も可
    require_3d: bool = True,
    only_known: bool = True,
    max_workers: int = 8,
    page_limit: int = 200,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> List[Tuple[str, Structure, float]]:
    """
    Alexandria (OPTIMADE) から (id, Structure) を取得する。
    2段階（軽量→構造）で高速化し、_alexandria_hull_distance<=ehull_max のみを返す。

    Parameters
    ----------
    chemsys : Sequence[str]
        例: ["Li","O","Fe","P"] など。単体～n元を自動で全探索。
    ehull_max : float
        eV/atom。_alexandria_hull_distance で判定。値が無い構造は除外。
    max_n : int | None
        返却件数の上限（e_hull 昇順後に適用）。None なら上限なし。
    base_url : str
        Alexandria データセットのベース（/pbe, /pbesol, /scan）。
    require_3d : bool
        True なら nperiodic_dimensions=3（バルクのみ）。
    only_known : bool
        True なら formation_energy_per_atom が IS KNOWN のものだけを最初に取得。
    max_workers : int
        軽量フェーズの並列度。
    page_limit : int
        1ページの件数（大きすぎるとプロバイダにより無視 or 400 のことあり）。

    Returns
    -------
    list[tuple[str, Structure, float]]
        (alexandria_id, pymatgen.Structure, e_hull) のリスト。e_hull 昇順。
    """
    # 1) 軽量＋構造を一段で取得（probe と同等の逐次・ロック無し）
    data_light = _gather_chemsys_light(
        base_url,
        chemsys,
        require_3d=require_3d,
        only_known=only_known,
        max_k=None,
        max_workers=max_workers,
        page_limit=page_limit,
        max_n=max_n,
        ehull_max=ehull_max,
        include_structure=True,
        use_lock=False,
        cache_dir=cache_dir,
    )

    # 2) hull_distance <= 閾値 の id を抽出（昇順ソート）
    def _get_ehull(attr: dict) -> Optional[float]:
        return attr.get("_alexandria_hull_distance")

    candidates = [
        d
        for d in data_light
        if _get_ehull(d.get("attributes", {})) is not None
        and _get_ehull(d.get("attributes", {})) <= ehull_max
    ]
    candidates.sort(key=lambda d: _get_ehull(d.get("attributes", {})))

    if max_n is not None:
        candidates = candidates[:max_n]

    # 3) (id, Structure) に整形（構造込みで取得済み）
    results: List[Tuple[str, Structure, float]] = []
    for entry in candidates:
        _id = entry["id"]
        attrs = entry.get("attributes", {})
        if not attrs:
            continue
        pmg = _entry_to_pmgstructure(entry)
        ehull_val = _get_ehull(attrs)
        if ehull_val is None:
            continue
        results.append((_id, pmg, float(ehull_val)))

    return results


# In[8]:


from pathlib import Path
from typing import List, Tuple, Sequence, Optional

from pymatgen.core.structure import Structure
from pymatgen.analysis.phase_diagram import PhaseDiagram

import random


def _chunks(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


RETRY_KEYS = (
    "status code 429",
    "status code 500",
    "status code 502",
    "status code 503",
    "status code 504",
    "Response ended prematurely",
)


def _retry(call, retries=6):
    for a in range(retries):
        try:
            return call()
        except Exception as e:
            s = str(e)
            if any(k in s for k in RETRY_KEYS):
                time.sleep(min(30, 2**a) + random.random())
                continue
            raise
    raise RuntimeError("MP API failed repeatedly")


def fetch_mp_structures(
    api_key: str,
    chemsys: Sequence[str],
    max_n: Optional[int] = None,
    ehull_max: float = FETCH_EHULL_DB_THRES,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> List[Tuple[str, Structure, float]]:
    """
    Fetch (material_id, Structure, e_hull) tuples from the Materials Project
    for a given chemical system, keeping only entries whose MP-reported energy
    above the convex hull is below ``ehull_max`` (default: 0.3 eV/atom).
    Uses MP's public energy_above_hull values (Summary docs) and includes
    lower-order subsets from the requested chemical system.

    Parameters
    ----------
    api_key : str
        Materials Project API key.
    chemsys : Sequence[str]
        Element symbols that define the chemical system, e.g. ``["Li", "O"]``.
    max_n : int | None, default None
        Maximum number of structures to return. ``None`` means no limit after
        filtering by ``ehull_max``.
    ehull_max : float, default 0.3
        Upper bound for the energy above hull in eV/atom.

    Returns
    -------
    list[tuple[str, Structure, float]]
        List of ``(material_id, Structure, e_hull)`` that satisfy the hull criterion.
    """
    def _summary_field(doc, key):
        if isinstance(doc, dict):
            return doc.get(key)
        return getattr(doc, key, None)

    with MPRester(api_key) as mpr:
        # Retrieve all thermo entries for the chemsys (includes subsets).
        fetch_start = time.time()
        entries = mpr.get_entries_in_chemsys(
            chemsys,
            additional_criteria={
                "thermo_types": ["GGA_GGA+U", "R2SCAN", "GGA_GGA+U_R2SCAN"],
            },
            inc_structure=False,
        )
        _append_timing(
            "mp_entries_fetch",
            time.time() - fetch_start,
            chemsys=",".join(chemsys),
            count=len(entries),
            pid=os.getpid(),
        )

        material_ids: List[str] = []
        for entry in entries:
            material_id = None
            if hasattr(entry, "data"):
                material_id = entry.data.get("material_id")
            if not material_id:
                material_id = getattr(entry, "entry_id", None)
            if material_id:
                material_ids.append(str(material_id))

        seen = set()
        unique_ids: List[str] = []
        for mid in material_ids:
            if mid in seen:
                continue
            seen.add(mid)
            unique_ids.append(mid)

        # Fetch MP summary docs to obtain public energy_above_hull and structure.
        summary_start = time.time()
        summary_docs = []
        for chunk in _chunks(unique_ids, 200):
            docs = _retry(
                lambda: mpr.materials.summary.search(
                    material_ids=chunk,
                    fields=["material_id", "structure", "energy_above_hull"],
                )
            )
            summary_docs.extend(docs)
        _append_timing(
            "mp_summary_fetch",
            time.time() - summary_start,
            chemsys=",".join(chemsys),
            count=len(summary_docs),
            pid=os.getpid(),
        )

        filtered = [
            d
            for d in summary_docs
            if _summary_field(d, "energy_above_hull") is not None
            and float(_summary_field(d, "energy_above_hull")) <= ehull_max
        ]
        filtered.sort(key=lambda d: float(_summary_field(d, "energy_above_hull")))

        # Respect the max_n limit (after E_hull filtering)
        if max_n is not None:
            filtered = filtered[:max_n]

        # Finally obtain the relaxed crystal structures
        # structures: List[Tuple[str, Structure]] = [
        #    (mid, mpr.get_structure_by_material_id(mid)) for mid in stable_ids
        # ]

        structures: List[Tuple[str, Structure, float]] = []
        struct_start = time.time()
        for d in filtered:
            mid = _summary_field(d, "material_id")
            if mid is None:
                continue
            struct = _summary_field(d, "structure")
            if struct is None:
                continue
            e_hull = _summary_field(d, "energy_above_hull")
            if e_hull is None:
                continue
            structures.append((str(mid), struct, float(e_hull)))

        _append_timing(
            "mp_structure_fetch",
            time.time() - struct_start,
            chemsys=",".join(chemsys),
            count=len(structures),
            pid=os.getpid(),
        )
    return structures


def relax_with_mattersim(pmg_struct, device=DEVICE, steps=400):
    """pymatgen Structure → ASE 変換後、位置＋セルを緩和"""
    atoms = AseAtomsAdaptor.get_atoms(pmg_struct)
    atoms.calc = MatterSimCalculator(load_path="MatterSim-v1.0.0-5M.pth", device=device)
    e0 = atoms.get_potential_energy()
    # cf = ExpCellFilter(atoms, constant_volume=False)
    # BFGS(cf, logfile=None).run(fmax=fmax, steps=steps)
    # e1 = atoms.get_potential_energy()
    relaxer = Relaxer(
        optimizer="BFGS",  # the optimization method
        filter="ExpCellFilter",  # filter to apply to the cell
        constrain_symmetry=True,  # whether to constrain the symmetry
    )

    _, relaxed_structure = relaxer.relax(atoms, steps=steps)
    e1 = relaxed_structure.get_potential_energy()
    return relaxed_structure, e0, e1


# # 実行ファイル群

# In[9]:


import json
from pathlib import Path
from typing import Dict, List

from pathlib import Path
import os
import tempfile
import fcntl


def _load_json_db(db_path: Path) -> dict:
    if db_path.exists():
        with db_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # {material_id: record} に正規化（あなたの既存仕様に合わせる）
        return {rec["material_id"]: rec for rec in data}
    return {}


def _source_from_material_id(material_id: str) -> Optional[str]:
    mid = str(material_id or "")
    if mid.startswith("mp-"):
        return "mp"
    if mid.startswith("agm"):
        return "alex"
    return None


def _assert_only_source_changed(
    before: List[dict], after: List[dict], context: str
) -> None:
    if len(before) != len(after):
        raise RuntimeError(
            f"{context}: record count changed ({len(before)} -> {len(after)})"
        )
    for idx, (b, a) in enumerate(zip(before, after)):
        b_wo = {k: v for k, v in b.items() if k != "source"}
        a_wo = {k: v for k, v in a.items() if k != "source"}
        if b_wo != a_wo:
            mid = b.get("material_id", a.get("material_id", f"index={idx}"))
            raise RuntimeError(
                f"{context}: non-source field changed for material_id={mid}"
            )


def _normalize_sources_in_relax_results_file(path: Path) -> List[dict]:
    if not path.exists():
        return []
    lock_path = path.parent / f"{path.name}.lock"
    with _InterProcessFileLock(lock_path):
        records = _load_json_gz(path)
        before = copy.deepcopy(records)
        unknown_material_ids: List[str] = []
        changed = 0
        for rec in records:
            src = rec.get("source")
            if src in {"mp", "alex"}:
                continue
            inferred = _source_from_material_id(str(rec.get("material_id", "")))
            if inferred is None:
                unknown_material_ids.append(
                    str(rec.get("material_id", "<missing_material_id>"))
                )
                continue
            rec["source"] = inferred
            changed += 1

        if unknown_material_ids:
            sample = ", ".join(unknown_material_ids[:10])
            raise RuntimeError(
                f"Unknown source in {path.name}: count={len(unknown_material_ids)} sample=[{sample}]"
            )

        _assert_only_source_changed(before, records, f"normalize_sources {path.name}")
        if changed > 0:
            _write_json_gz(path, records)
        return records


def _contains_corrupt_file_error(exc: BaseException) -> bool:
    """Return True when `exc` (or its cause/context chain) indicates file corruption."""
    chain: set[int] = set()
    cur: BaseException | None = exc
    corrupt_types = (
        json.JSONDecodeError,
        gzip.BadGzipFile,
        EOFError,
        UnicodeDecodeError,
    )
    while cur is not None and id(cur) not in chain:
        if isinstance(cur, corrupt_types):
            return True
        chain.add(id(cur))
        cur = cur.__cause__ or cur.__context__
    return False


def _raise_if_corrupt_file_error(exc: Exception, path: Path, context: str) -> None:
    if _contains_corrupt_file_error(exc):
        raise RuntimeError(f"{context}: detected corrupted file: {path}") from exc


def _load_relax_results_file(path: Path, normalize_source: bool = False) -> List[dict]:
    if not path.exists():
        return []
    try:
        if normalize_source:
            records = _normalize_sources_in_relax_results_file(path)
        else:
            records = _load_json_gz(path)
    except Exception as exc:
        if normalize_source:
            raise
        _raise_if_corrupt_file_error(exc, path, "load_relax_results_file")
        return []
    _backfill_collection_chemsys(records, f"load {path.name}")
    return records


class _InterProcessFileLock:
    """Advisory lock using fcntl.flock (Unix)."""

    def __init__(self, lock_path: Path):
        self.lock_path = lock_path
        self.fd = None

    def __enter__(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        # O_CREAT で確実にロック用ファイルを作る
        self.fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR)
        fcntl.flock(self.fd, fcntl.LOCK_EX)  # 排他ロック
        return self

    def __exit__(self, exc_type, exc, tb):
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        os.close(self.fd)


def _timing_log_path() -> Path:
    """Return timing log path under current workspace."""
    log_dir = Path.cwd() / "timing_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "timing.jsonl"


def _append_timing(event: str, elapsed_s: float, **fields) -> None:
    """Append timing data as JSONL; failures should not stop the run."""
    try:
        record = {
            "ts": time.time(),
            "event": event,
            "elapsed_s": float(elapsed_s),
        }
        record.update(fields)
        log_path = _timing_log_path()
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
    except Exception:
        pass


# ------------------------------------------------------------
# Progress logging (spec: logs/time_logs/stage*.log)
# ------------------------------------------------------------
def _progress_log_path(stage: str) -> Path:
    path = Path.cwd() / "logs" / "time_logs"
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{stage}.log"


def _append_progress(stage: str, record: dict) -> None:
    """Append stage progress to logs/time_logs/<stage>.log (JSONL)."""
    try:
        record = {"stage": stage, **record}
        log_path = _progress_log_path(stage)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
    except Exception:
        pass


def _load_json_gz(path: Path) -> list:
    with gzip.open(path, "rt") as fp:
        return json.load(fp)


def _write_json_gz(path: Path, data: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    ) as tf:
        with gzip.open(tf, "wt", encoding="utf-8") as fp:
            json.dump(data, fp)
        tf.flush()
        os.fsync(tf.fileno())
        tmp_name = tf.name
    os.replace(tmp_name, path)


def _read_json(path: Path) -> dict | list:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _write_json_atomic(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
        encoding="utf-8",
    ) as tf:
        json.dump(data, tf, ensure_ascii=True, indent=2)
        tf.flush()
        os.fsync(tf.fileno())
        tmp_name = tf.name
    os.replace(tmp_name, path)


def _read_jsonl(path: Path) -> List[dict]:
    records = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _has_cache_for_chemsys(cache_path: Path, target_chemsys: str) -> bool:
    """Return True if mp_cache has at least one entry for target_chemsys."""
    if not cache_path.exists():
        return False
    try:
        entries = _load_cached_entries(cache_path, "mp_entries", target_chemsys)
        return len(entries) > 0
    except Exception:
        return False


def _find_superset_done(target_chemsys: str, fetch_done: dict) -> List[str]:
    """
    Return list of chemsys keys in fetch_done whose element set is a superset of target.
    """
    target_set = set(_elements_from_chemsys(target_chemsys))
    supers: List[str] = []
    for cs in fetch_done.keys():
        try:
            sup_set = set(_elements_from_chemsys(cs))
        except Exception:
            continue
        if target_set.issubset(sup_set):
            supers.append(cs)
    return supers


def _write_jsonl_atomic(path: Path, records: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
        encoding="utf-8",
    ) as tf:
        for rec in records:
            tf.write(json.dumps(rec, ensure_ascii=True) + "\n")
        tf.flush()
        os.fsync(tf.fileno())
        tmp_name = tf.name
    os.replace(tmp_name, path)


def _alexandria_lock_path(cache_dir: Path) -> Path:
    """Lock file for Alexandria API access (scoped per cache_dir)."""
    return cache_dir / "alexandria_api.lock"


def _mp_lock_path(cache_dir: Path) -> Path:
    """Lock file for Materials Project API access (scoped per cache_dir)."""
    return cache_dir / "mp_api.lock"


def _material_lock_path(material_id: str) -> Path:
    return DEFAULT_RELAX_DB_PATH.parent / "locks" / f"{material_id}.lock"


# ------------------------------------------------------------
# Cache and manifest helpers for spec-compliant stages
# ------------------------------------------------------------
def _ensure_cache_table(db_path: Path, table: str) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                material_id TEXT PRIMARY KEY,
                entry_chemsys TEXT NOT NULL,
                source TEXT NOT NULL,
                data_json TEXT NOT NULL,
                e_hull REAL
            )
            """
        )
        conn.commit()


def _cache_entries(db_path: Path, table: str, entries: Sequence[dict]) -> int:
    if not entries:
        return 0
    _ensure_cache_table(db_path, table)
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        inserted = 0
        for rec in entries:
            cur.execute(
                f"""
                INSERT OR REPLACE INTO {table} (material_id, entry_chemsys, source, data_json, e_hull)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    rec["material_id"],
                    rec["entry_chemsys"],
                    rec.get("source", "mp"),
                    rec["data_json"],
                    rec.get("e_hull"),
                ),
            )
            inserted += 1
        conn.commit()
    return inserted


def _load_cached_entries(db_path: Path, table: str, target_chemsys: str) -> List[dict]:
    if not db_path.exists():
        return []
    target_set = set(_elements_from_chemsys(target_chemsys))
    out: List[dict] = []
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        for material_id, entry_chemsys, source, data_json, e_hull in cur.execute(
            f"SELECT material_id, entry_chemsys, source, data_json, e_hull FROM {table}"
        ):
            entry_set = set(_elements_from_chemsys(entry_chemsys))
            if not entry_set.issubset(target_set):
                continue
            out.append(
                {
                    "material_id": material_id,
                    "entry_chemsys": entry_chemsys,
                    "source": source,
                    "data_json": data_json,
                    "e_hull": e_hull,
                }
            )
    return out


def _load_fetch_done(path: Path) -> dict:
    if path.exists():
        try:
            return _read_json(path)
        except Exception as exc:
            _raise_if_corrupt_file_error(exc, path, "load_fetch_done")
            return {}
    return {}


def _save_fetch_done(path: Path, data: dict) -> None:
    _write_json_atomic(path, data)


def _query_signature_mp() -> str:
    payload = {
        "fetch_ehull_db_thres": FETCH_EHULL_DB_THRES,
        "thermo_types": ["GGA_GGA+U", "R2SCAN", "GGA_GGA+U_R2SCAN"],
        "v": 1,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _query_signature_alex() -> str:
    payload = {
        "fetch_ehull_db_thres": FETCH_EHULL_DB_THRES,
        "base_url": "https://alexandria.icams.rub.de/pbe",
        "v": 1,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _enumerate_subsystems(chemsys: str) -> List[str]:
    elems = _elements_from_chemsys(chemsys)
    out: List[str] = []
    for k in range(1, len(elems) + 1):
        for combo in combinations(elems, k):
            out.append(chemsys_by_atomic_number(combo))
    return sorted(set(out))


def _sort_chemsys_desc(cs_list: Iterable[str]) -> List[str]:
    """元素数の降順、次いで文字列昇順でソートした化学系リストを返す。"""
    return sorted(
        {_normalize_chemsys(cs) for cs in cs_list},
        key=lambda cs: (-len(_elements_from_chemsys(cs)), cs),
    )


def _save_json_db(db_path: Path, new_records: dict) -> None:
    """
    new_records: {material_id: record}
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = db_path.with_suffix(db_path.suffix + ".lock")

    with _InterProcessFileLock(lock_path):
        # 1) ロック中に最新のオンディスクを読み直してマージ
        on_disk = _load_json_db(db_path)
        on_disk.update(new_records)

        # 2) 一意な一時ファイル（同一ディレクトリ内）に書く
        with tempfile.NamedTemporaryFile(
            "w",
            dir=db_path.parent,
            prefix=db_path.name + ".",
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as tf:
            json.dump(list(on_disk.values()), tf, indent=2, ensure_ascii=False)
            tf.flush()
            os.fsync(tf.fileno())  # ディスクへフラッシュ
            tmp_name = tf.name

        # 3) 原子的に差し替え（同一FS & 同ディレクトリ内が必須）
        os.replace(tmp_name, db_path)


def _load_known_material_ids(db_path: Path) -> set[str]:
    if not db_path.exists():
        return set()
    return set(_load_json_db(db_path).keys())


def _record_has_agm(rec: dict) -> bool:
    file_path = rec.get("file")
    return bool(file_path and Path(file_path).name.startswith("agm"))


def _ensure_record_chemsys(rec: dict) -> Optional[str]:
    chem_sys = rec.get("chem_sys")
    if chem_sys:
        return chem_sys
    formula = rec.get("formula")
    if formula:
        from pymatgen.core.composition import Composition

        elems = [str(el) for el in Composition(formula).elements]
        if elems:
            return chemsys_by_atomic_number(elems)
    file_path = rec.get("file")
    if not file_path:
        return None
    path = _resolve_relax_record_path(file_path)
    if not path.exists():
        return None
    return chemsys_by_atomic_number(get_elements_from_poscar(path))


def _backfill_record_chemsys(rec: dict, label: str) -> bool:
    if rec.get("chem_sys"):
        return False
    inferred = _ensure_record_chemsys(rec)
    if not inferred:
        raise RuntimeError(
            f"Missing chem_sys in {label} "
            f"(material_id={rec.get('material_id')}, inferred={inferred}, "
            f"file={rec.get('file')})"
        )
    rec["chem_sys"] = inferred
    return True


def _backfill_collection_chemsys(records: Sequence[dict], label: str) -> None:
    for rec in records:
        _backfill_record_chemsys(rec, label)


def _missing_terminal_elements(
    target_elems: set[str],
    records: Sequence[dict],
) -> List[str]:
    terminal_elems: set[str] = set()
    for rec in records:
        chem_sys = rec.get("chem_sys")
        if not chem_sys:
            chem_sys = _ensure_record_chemsys(rec)
            if not chem_sys:
                raise RuntimeError(
                    "Missing chem_sys during terminal check "
                    f"(material_id={rec.get('material_id')}, file={rec.get('file')})"
                )
            chem_sys = _normalize_chemsys(chem_sys)
            rec["chem_sys"] = chem_sys
        elems = set(_elements_from_chemsys(chem_sys))
        if len(elems) == 1:
            terminal_elems.update(elems)
    missing = sorted(target_elems - terminal_elems, key=lambda e: Element(e).Z)
    return missing


def _update_subchemsys_files(
    records: List[Dict[str, dict]],
    base_dir: Path,
    target_chemsys: Optional[Sequence[str]] = None,
) -> Dict[str, int]:
    """
    Build relax_results_<chem_sys>.json.gz for each target chem_sys using
    records whose elements are subsets of the target (includes terminals).
    Deduplicate by material_id. Overwrite only if candidate has agm data and
    existing file does not.
    """
    record_infos: List[tuple[str, dict, frozenset[str], bool]] = []
    missing_poscar = 0

    for rec in records:
        chem_sys = _ensure_record_chemsys(rec)
        if not chem_sys:
            missing_poscar += 1
            raise RuntimeError(
                "Missing chem_sys for record in subchemsys update "
                f"(material_id={rec.get('material_id')}, file={rec.get('file')})"
            )
        chem_sys = _normalize_chemsys(chem_sys)
        rec["chem_sys"] = chem_sys
        key = rec.get("material_id") or rec.get("file")
        if not key:
            continue
        elems = frozenset(_elements_from_chemsys(chem_sys))
        record_infos.append((key, rec, elems, _record_has_agm(rec)))

    if target_chemsys is None:
        target_chemsys = sorted({info[1]["chem_sys"] for info in record_infos})
    else:
        target_chemsys = sorted({_normalize_chemsys(cs) for cs in target_chemsys})

    new_files = 0
    overwritten_files = 0

    for chem_sys in target_chemsys:
        target_elems = set(_elements_from_chemsys(chem_sys))
        if not target_elems:
            continue

        candidate_by_key: Dict[str, dict] = {}
        for key, rec, elems, has_agm in record_infos:
            if elems.issubset(target_elems):
                candidate_by_key[key] = rec

        candidate_records = list(candidate_by_key.values())
        missing_terms = _missing_terminal_elements(target_elems, candidate_records)
        if missing_terms:
            raise RuntimeError(
                "Missing terminal entries for elements "
                f"{missing_terms} (chemsys={chem_sys})"
            )

        out_path = base_dir / f"relax_results_{chem_sys}.json.gz"
        exists = out_path.exists()
        if exists:
            existing = _load_json_gz(out_path)
            # merge existing + candidate (candidate takes precedence)
            existing_by_key = {}
            for rec in existing:
                key = rec.get("material_id") or rec.get("file")
                if key:
                    existing_by_key[key] = rec
            for k, v in candidate_by_key.items():
                existing_by_key[k] = v
            candidate_records = list(existing_by_key.values())

        _write_json_gz(out_path, candidate_records)
        if exists:
            overwritten_files += 1
        else:
            new_files += 1

    return {
        "new_files": new_files,
        "overwritten_files": overwritten_files,
        "missing_poscar": missing_poscar,
    }


# In[10]:


def _failure_record(mp_id: str, error_msg: str) -> dict:
    return {
        "material_id": mp_id,
        "formula": None,
        "E_initial (eV)": None,
        "E_final (eV)": None,
        "ΔE (eV)": None,
        "file": None,
        "status": "failed",
        "error": error_msg,
    }


# In[11]:


from typing import Dict, Any
from pymatgen.io.vasp.inputs import Incar, Poscar


def extract_params_from_vasp_input(incar: Incar, elems: List[str]) -> Dict[str, Any]:
    """
    VaspInput (vasp_input.incar, vasp_input.potcar) から
    Materials Project 互換の params を返す関数。

    Args:
        vasp_input (VaspInput): Pymatgen VaspInput

    Returns:
        dict: {"run_type": str, "is_hubbard": bool, "hubbards": dict}
    """
    # --- run_type 判定 ---
    if incar.get("METAGGA") not in (None, "None"):
        run_type = str(incar.get("METAGGA"))
    elif incar.get("LHFCALC", False):
        run_type = "HYB"
    else:
        ldau = bool(incar.get("LDAU", False))
        ldauu = incar.get("LDAUU", [])
        if isinstance(ldauu, str):
            ldauu = [float(x) for x in ldauu.split()]
        has_positive_u = any(float(u) > 0 for u in ldauu)
        run_type = "GGA+U" if (ldau and has_positive_u) else "GGA"

    # --- Hubbard 辞書作成 ---
    is_hubbard = bool(incar.get("LDAU", False))
    hubbards: Dict[str, float] = {}
    if is_hubbard:
        ldaul = incar.get("LDAUL", [])
        ldauu = incar.get("LDAUU", [])
        # 文字列ならリスト化
        if isinstance(ldaul, str):
            ldaul = [int(x) for x in ldaul.split()]
        if isinstance(ldauu, str):
            ldauu = [float(x) for x in ldauu.split()]
        # 長さを POTCAR の元素数に合わせる
        n = len(elems)
        if len(ldaul) < n:
            ldaul += [-1] * (n - len(ldaul))
        if len(ldauu) < n:
            ldauu += [0.0] * (n - len(ldauu))

        for el, lval, uval in zip(elems, ldaul, ldauu):
            if int(lval) >= 0 and float(uval) > 0.0:
                hubbards[el] = float(uval)

    return {
        "run_type": run_type,
        "is_hubbard": is_hubbard,
        "hubbards": hubbards,
    }


# In[12]:


from pathlib import Path
from typing import List, Dict, Optional, Sequence

import gc
import torch
import shutil

from pymatgen.core import Structure, Composition
from pymatgen.entries.computed_entries import ComputedStructureEntry

try:
    from atomate2.vasp.flows.mp import MPGGADoubleRelaxStaticMaker
except ImportError:
    MPGGADoubleRelaxStaticMaker = None


_RUN_PARAM_INPUT_SET_GENERATOR = None


def _new_mp_gga_double_relax_static_maker():
    if MPGGADoubleRelaxStaticMaker is None:
        raise ImportError(
            "atomate2 is required to recompute VASP run_params. "
            "Install atomate2, or use --allow_run_params_fallback when existing "
            "relax_results records already contain usable run_params."
        )
    return MPGGADoubleRelaxStaticMaker()


def _get_run_param_input_set_generator():
    """Lazily initialize the MP input-set generator used for run_params."""
    global _RUN_PARAM_INPUT_SET_GENERATOR
    if _RUN_PARAM_INPUT_SET_GENERATOR is None:
        flow_maker = _new_mp_gga_double_relax_static_maker()
        _RUN_PARAM_INPUT_SET_GENERATOR = flow_maker.static_maker.input_set_generator
    return _RUN_PARAM_INPUT_SET_GENERATOR


def _infer_record_source(record: dict) -> str:
    src = record.get("source")
    if src in {"mp", "alex"}:
        return src
    mid = str(record.get("material_id", ""))
    if mid.startswith("mp-"):
        return "mp"
    if mid.startswith("agm"):
        return "alex"
    return ""


def _is_failed_relax_record(rec: dict) -> bool:
    if rec.get("status") == "failed":
        return True
    return str(rec.get("file", "")).endswith("_failed.vasp")


def _default_run_params_from_record(rec: dict) -> dict:
    return {
        "run_type": str(rec.get("run_type", "GGA")),
        "is_hubbard": bool(rec.get("is_hubbard", False)),
        "hubbards": dict(rec.get("hubbards", {}) or {}),
    }


def _build_run_params_from_structure(struct: Structure) -> dict:
    gen = _get_run_param_input_set_generator()
    vis = gen.get_input_set(struct, potcar_spec=True)
    incar = vis.incar
    elems = list(vis.poscar.site_symbols)
    return extract_params_from_vasp_input(incar, elems)


def _assign_run_params_to_record(
    rec: dict,
    *,
    allow_fallback: bool = False,
    context: str = "",
) -> tuple[bool, Optional[str]]:
    """
    Recompute and set run_params for a record from its structure file.

    Returns:
        (changed, error_message). error_message is None on success.
    """
    file_value = rec.get("file")
    if not file_value:
        msg = f"{context}: missing file path for material_id={rec.get('material_id')}"
        if allow_fallback:
            fallback = _default_run_params_from_record(rec)
            rec.update(fallback)
            return True, msg
        return False, msg

    struct_path = _resolve_relax_record_path(file_value)
    if not struct_path.exists():
        msg = (
            f"{context}: structure file not found for material_id={rec.get('material_id')}: "
            f"{struct_path}"
        )
        if allow_fallback:
            fallback = _default_run_params_from_record(rec)
            rec.update(fallback)
            return True, msg
        return False, msg

    try:
        struct = Structure.from_file(struct_path)
        run_params = _build_run_params_from_structure(struct)
    except Exception as exc:
        msg = f"{context}: failed to build run_params for material_id={rec.get('material_id')}: {exc}"
        if allow_fallback:
            fallback = _default_run_params_from_record(rec)
            before = _default_run_params_from_record(rec)
            rec.update(fallback)
            changed = any(
                before[k] != fallback[k] for k in ("run_type", "is_hubbard", "hubbards")
            )
            return changed, msg
        return False, msg

    before = _default_run_params_from_record(rec)
    rec.update(run_params)
    changed = any(
        before[k] != run_params[k] for k in ("run_type", "is_hubbard", "hubbards")
    )
    return changed, None


def _repair_mp_run_params_in_records(
    records: List[dict],
    *,
    allow_fallback: bool = False,
    context: str = "",
) -> Dict[str, Any]:
    """
    Repair run_params in-memory for non-failed reference records.
    """
    changed = 0
    processed = 0
    skipped = 0
    errors: List[str] = []
    for rec in records:
        if _infer_record_source(rec) not in {"mp", "alex"}:
            continue
        if _is_failed_relax_record(rec):
            skipped += 1
            continue
        processed += 1
        rec_changed, err = _assign_run_params_to_record(
            rec,
            allow_fallback=allow_fallback,
            context=context,
        )
        if err is not None:
            errors.append(err)
            if not allow_fallback:
                break
        if rec_changed:
            changed += 1

    if errors and not allow_fallback:
        raise RuntimeError(errors[0])

    return {
        "processed": processed,
        "changed": changed,
        "skipped": skipped,
        "errors": errors,
    }


def _collect_mp_compatibility_diagnostics(
    entries_mp: List[ComputedStructureEntry],
    proc_mp: List[ComputedStructureEntry],
    *,
    context: str,
) -> RuntimeError:
    proc_ids = {
        entry.entry_id for entry in proc_mp if str(entry.entry_id).startswith("mp-")
    }
    filtered_ids = [
        entry.entry_id
        for entry in entries_mp
        if str(entry.entry_id).startswith("mp-") and entry.entry_id not in proc_ids
    ]
    compat = MaterialsProject2020Compatibility(
        check_potcar_hash=False,
        check_potcar=False,
    )
    reasons: List[str] = []
    for entry in entries_mp:
        if entry.entry_id not in filtered_ids[:20]:
            continue
        try:
            compat.process_entry(entry, clean=True, on_error="raise")
        except Exception as exc:
            head = str(exc).split("\n")[0]
            reasons.append(f"{entry.entry_id}: {head}")
    filtered_preview = ", ".join(filtered_ids[:10]) if filtered_ids else "<none>"
    reason_preview = "; ".join(reasons[:5]) if reasons else "<no detailed reason>"
    msg = (
        f"{context}: mp_ok_count/proc_count mismatch; filtered MP entries detected. "
        f"filtered_sample=[{filtered_preview}] reasons=[{reason_preview}]"
    )
    return RuntimeError(msg)


def _assert_mp_ok_count_equals_proc_count(
    mp_records_ok: List[dict],
    proc_entries: List[ComputedStructureEntry],
    entries_mp: List[ComputedStructureEntry],
    *,
    context: str,
) -> None:
    mp_ok_count = len(mp_records_ok)
    expected_mp_entry_ids = set()
    for rec in mp_records_ok:
        mid = rec.get("material_id")
        if not mid:
            raise RuntimeError(
                f"{context}: missing material_id in MP record during count check"
            )
        expected_mp_entry_ids.add(f"{mid}_ms")

    proc_count = sum(
        1 for entry in proc_entries if str(entry.entry_id) in expected_mp_entry_ids
    )
    if mp_ok_count != proc_count:
        raise _collect_mp_compatibility_diagnostics(
            entries_mp, proc_entries, context=context
        )


def _collect_alex_compatibility_diagnostics(
    entries_refs: List[ComputedStructureEntry],
    proc_entries: List[ComputedStructureEntry],
    *,
    context: str,
) -> RuntimeError:
    proc_ids = {
        entry.entry_id
        for entry in proc_entries
        if str(entry.entry_id).startswith("agm")
    }
    filtered_ids = [
        entry.entry_id
        for entry in entries_refs
        if str(entry.entry_id).startswith("agm") and entry.entry_id not in proc_ids
    ]
    filtered_preview = ", ".join(filtered_ids[:10]) if filtered_ids else "<none>"
    msg = (
        f"{context}: alex_ok_count/proc_count mismatch; filtered Alexandria entries detected. "
        f"filtered_sample=[{filtered_preview}]"
    )
    return RuntimeError(msg)


def _assert_alex_ok_count_equals_proc_count(
    alex_records_ok: List[dict],
    proc_entries: List[ComputedStructureEntry],
    entries_refs: List[ComputedStructureEntry],
    *,
    context: str,
) -> None:
    alex_ok_count = len(alex_records_ok)
    proc_count = sum(
        1 for entry in proc_entries if str(entry.entry_id).startswith("agm")
    )
    if alex_ok_count != proc_count:
        raise _collect_alex_compatibility_diagnostics(
            entries_refs, proc_entries, context=context
        )


def _repair_run_params_in_relax_results_file(
    path: Path,
    *,
    allow_fallback: bool = False,
    verify_mp_compat: bool = False,
    context: str = "",
) -> Dict[str, Any]:
    """
    Repair run_params for non-failed reference records in
    relax_results_<chemsys>.json.gz.
    """
    if not path.exists():
        return {
            "records": [],
            "processed": 0,
            "changed": 0,
            "skipped": 0,
            "errors": [],
            "mp_ok_count": 0,
            "proc_count": 0,
        }

    # Normalize source fields first.
    _ = _load_relax_results_file(path, normalize_source=True)
    lock_path = path.parent / f"{path.name}.lock"
    with _InterProcessFileLock(lock_path):
        records = _load_json_gz(path)
        before_run_params_by_mid: Dict[str, dict] = {}
        for rec in records:
            mid = rec.get("material_id")
            if mid is None:
                continue
            before_run_params_by_mid[str(mid)] = _default_run_params_from_record(rec)
        _backfill_collection_chemsys(records, f"repair_run_params {path.name}")
        stats = _repair_mp_run_params_in_records(
            records,
            allow_fallback=allow_fallback,
            context=context or f"repair_run_params {path.name}",
        )
        if stats["changed"] > 0:
            _write_json_gz(path, records)
            changed_records_by_mid: Dict[str, dict] = {}
            for rec in records:
                mid = rec.get("material_id")
                if mid is None:
                    continue
                key = str(mid)
                before = before_run_params_by_mid.get(key)
                after = _default_run_params_from_record(rec)
                if before is None or any(
                    before[k] != after[k]
                    for k in ("run_type", "is_hubbard", "hubbards")
                ):
                    changed_records_by_mid[key] = rec
            if changed_records_by_mid:
                _save_json_db(DEFAULT_RELAX_DB_PATH, changed_records_by_mid)

    mp_ok_count = 0
    proc_count = 0
    if verify_mp_compat:
        mp_records_ok = [
            rec
            for rec in records
            if _infer_record_source(rec) == "mp" and not _is_failed_relax_record(rec)
        ]
        if mp_records_ok:
            entries_mp = _make_entries(mp_records_ok)
            compat = MaterialsProject2020Compatibility(
                check_potcar_hash=False,
                check_potcar=False,
            )
            proc_mp = compat.process_entries(entries_mp, clean=True)
            _assert_mp_ok_count_equals_proc_count(
                mp_records_ok,
                proc_mp,
                entries_mp,
                context=context or f"repair_run_params {path.name}",
            )
            mp_ok_count = len(mp_records_ok)
            proc_count = sum(
                1 for entry in proc_mp if str(entry.entry_id).startswith("mp-")
            )

    stats.update(
        {
            "records": records,
            "mp_ok_count": mp_ok_count,
            "proc_count": proc_count,
        }
    )
    return stats


# ----------------------------------------------------------------------
# ①  New helper function
# ----------------------------------------------------------------------
def relax_and_save(
    mp_id: str,
    struct: Structure,
    outdir_path: Path,
    steps_max: int = 400,
) -> Dict[str, str | float]:
    """
    Relax a single Materials Project structure with MatterSim, save the relaxed
    structure in VASP (POSCAR) format, and return a result record.

    Parameters
    ----------
    mp_id : str
        Materials Project material ID (e.g. ``"mp-12345"``).
    struct : Structure
        Pymatgen ``Structure`` object to be relaxed.
    outdir : Path
        Directory where the relaxed file will be written.
    steps_max : int, default 400
        Maximum number of relaxation steps for MatterSim.

    Returns
    -------
    dict
        A dictionary with keys::

            {
              "material_id": mp_id,
              "formula": struct.formula,
              "E_initial (eV)": e0_total,
              "E_final (eV)":   e1_total,
              "ΔE (eV)":        e1_total - e0_total,
              "file":           "<POSCAR path>",
            }

    Notes
    -----
    * The function **does not** catch exceptions; let the caller handle them.
    * GPU / host memory is cleared after each relaxation to avoid OOM issues.
    """
    # --- Relaxation -------------------------------------------------------
    atoms_rlx, e0_total, e1_total = relax_with_mattersim(struct, steps=steps_max)
    delta_e = float(e1_total - e0_total)

    # --- Save structure ---------------------------------------------------
    outdir_path.mkdir(parents=True, exist_ok=True)
    poscar_path = outdir_path / f"{mp_id}_relaxed.vasp"
    ase.io.write(poscar_path, atoms_rlx, format="vasp")

    # --- Free memory ------------------------------------------------------
    del atoms_rlx
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Assemble record --------------------------------------------------
    return {
        "material_id": mp_id,
        "formula": struct.formula,
        "E_initial (eV)": float(e0_total),
        "E_final (eV)": float(e1_total),
        "ΔE (eV)": delta_e,
        "file": str(poscar_path),
        "chem_sys": chemsys_by_atomic_number(struct.symbol_set),
    }


def _relax_and_save_with_retry(
    mp_id: str,
    struct: Structure,
    outdir_path: Path,
    steps_max: int = 400,
    attempts: int = 3,
) -> Dict[str, str | float]:
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return relax_and_save(mp_id, struct, outdir_path, steps_max=steps_max)
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            fatal_hint = "MatterSim" in msg or "load_path" in msg
            if fatal_hint:
                raise
            if attempt == attempts:
                break
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    # fallback failure record with magic 999 to continue pipeline
    failed_path = outdir_path / f"{mp_id}_failed.vasp"
    try:
        Poscar(struct).write_file(failed_path)
    except Exception:
        pass
    return {
        "material_id": mp_id,
        "formula": struct.formula,
        "E_initial (eV)": 999.0,
        "E_final (eV)": 999.0,
        "ΔE (eV)": 999.0,
        "file": str(failed_path),
        "chem_sys": chemsys_by_atomic_number(struct.symbol_set),
        "run_type": "GGA",
        "is_hubbard": False,
        "hubbards": {},
        "status": "failed",
        "error": str(last_exc) if last_exc else "unknown",
    }


def collect_mp_alex_entries(
    api_key: str,
    chemsys: Sequence[str],
    db_path: Path,
    max_n: Optional[int] = None,
    steps_max: int = 400,
    use_alexandria: bool = False,
    outdir: Path | str = "mp_relaxed",
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> List[Dict[str, str | float]]:

    outdir_path = Path(db_path.parent) / Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    # Hubbard U 等の情報を取得するための VASP 入力ジェネレータ
    flow_maker = _new_mp_gga_double_relax_static_maker()
    static_maker = flow_maker.static_maker
    gen = static_maker.input_set_generator  # VaspInputGenerator

    # ---------- 既存 DB をロード -------------------------------
    if db_path.exists():
        print(" Loading existing database...")
        cache: Dict[str, dict] = _load_json_db(db_path)
    else:
        cache = {}

    results: List[Dict[str, str | float]] = []
    failure_list: List[str] = []

    # Get structures from Materials Project and Alexandria Materials Database
    fetch_data = fetch_mp_structures(api_key, chemsys, max_n, cache_dir=cache_dir)
    if use_alexandria:
        fetch_alex_data = fetch_alex_structures(
            chemsys,
            ehull_max=1.0,
            max_n=max_n,
            base_url="https://alexandria.icams.rub.de/pbe",  # /pbesol, /scan に切替可
            cache_dir=cache_dir,
        )
        fetch_data = fetch_data + fetch_alex_data

    total_entries = len(fetch_data)
    start_time = time.time()
    for entry_i, entry in enumerate(
        tqdm(fetch_data, desc="Relaxing structures"), start=1
    ):
        mp_id, struct = entry[0], entry[1]
        public_ehull = entry[2] if len(entry) > 2 else None
        # ---------- 進捗表示 ---------------------------------------
        clear_output(wait=True)
        _log_prefetch_progress(
            "mp_collect",
            entry_i,
            total_entries,
            start_time,
            item_id=mp_id,
        )
        print(f"Processing entry {entry_i}/{len(fetch_data)}: {mp_id}")
        display(struct)
        # ---------- キャッシュ判定 -----------------------------
        if mp_id in cache:
            # 既に計算済み
            cached = cache[mp_id]
            if public_ehull is not None and cached.get("e_hull") is None:
                cached["e_hull"] = float(public_ehull)
                cache[mp_id] = cached
                _save_json_db(db_path, cache)
            results.append(cached)
            print(f"✓ Cache hit — skip {mp_id}")
            continue

        # ---------- 未計算なら通常フロー ------------------------
        try:
            record = relax_and_save(
                mp_id=mp_id,
                struct=struct,
                outdir_path=outdir_path,
                steps_max=steps_max,
            )
        except Exception as exc:
            print(f"✗ Error processing {mp_id}: {exc}")
            failure_list.append(mp_id)
            continue

        vis = gen.get_input_set(struct, potcar_spec=True)
        incar = vis.incar
        elems = Poscar(struct).site_symbols
        params = extract_params_from_vasp_input(incar, elems)
        record.update(params)
        if public_ehull is not None:
            record["e_hull"] = float(public_ehull)

        # DB・結果リストに追加
        cache[mp_id] = record
        results.append(record)
        _save_json_db(db_path, cache)

        # 予備用にdbをコピーしておく
        db_backup_path = db_path.parent / f"{db_path.stem}_backup.json"
        shutil.copy(db_path, db_backup_path)

        print(
            f"{record['formula']} ({mp_id}) "
            f"{record['E_final (eV)']:.4f} eV "
            f"(ΔE = {record['ΔE (eV)']:+.3f} eV)"
        )

    # ---------- DB を保存 -------------------------------------
    _save_json_db(db_path, cache)

    # 予備用にdbをコピーしておく
    db_backup_path = db_path.parent / f"{db_path.stem}_backup.json"
    shutil.copy(db_path, db_backup_path)

    if failure_list:
        print("Failed IDs:", failure_list)

    return results


def make_entries_from_results(results_list):
    """results (List[dict]) を ComputedStructureEntry のリストに変換"""
    entries = []
    for rec in results_list:

        # VASP ファイルから Structure を再ロード
        struct_path = _resolve_relax_record_path(rec["file"])
        struct = Structure.from_file(struct_path)
        # total energy [eV] を per atom に直す
        n_atoms = len(struct)
        energy_pa = rec["E_final (eV)"] / n_atoms
        run_params = {
            "run_type": rec["run_type"],
            "is_hubbard": rec["is_hubbard"],
            "hubbards": rec["hubbards"],
            "source": "mattersim",
        }

        entry = ComputedStructureEntry(
            structure=struct,
            energy=rec["E_final (eV)"],  # energy_pa,      # ← per atom
            parameters=run_params,
            entry_id=rec["material_id"] + "_ms",  # ID は一意なら何でも
        )
        entries.append(entry)

    return entries


# In[13]:


from pathlib import Path
from typing import List, Dict, Sequence, Optional

import torch
from pymatgen.core import Structure


SKIP_EHULL_REASON_TOO_CLOSE = "too_close_get_input_set"
SKIP_EHULL_REASON_PHASE_DIAGRAM_TIMEOUT = "phase_diagram_timeout"


class PhaseDiagramTimeoutError(TimeoutError):
    pass


def _phase_diagram_worker(
    proc_entries,
    target_id: str,
    child_conn,
) -> None:
    try:
        target_entry = next(
            (entry for entry in proc_entries if entry.entry_id == target_id), None
        )
        if target_entry is None:
            raise ValueError(
                f"Proposal entry missing in processed entries: {target_id}"
            )
        pd = PhaseDiagram(proc_entries)
        child_conn.send(("ok", float(pd.get_e_above_hull(target_entry))))
    except Exception:
        child_conn.send(("err", traceback.format_exc()))
    finally:
        child_conn.close()


def _compute_ehull_with_timeout(
    proc_entries,
    target_id: str,
    *,
    timeout_sec: Optional[float],
    context: str,
) -> float:
    if timeout_sec is None:
        target_entry = next(
            (entry for entry in proc_entries if entry.entry_id == target_id), None
        )
        if target_entry is None:
            raise ValueError(
                f"Proposal entry missing in processed entries: {target_id}"
            )
        pd = PhaseDiagram(proc_entries)
        return float(pd.get_e_above_hull(target_entry))

    timeout_value = float(timeout_sec)
    if timeout_value <= 0 or not np.isfinite(timeout_value):
        raise ValueError(
            f"phase diagram timeout must be positive finite seconds: {timeout_sec}"
        )

    ctx = mp.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_phase_diagram_worker,
        args=(proc_entries, target_id, child_conn),
    )
    proc.start()
    child_conn.close()
    proc.join(timeout_value)
    if proc.is_alive():
        proc.kill()
        proc.join()
        parent_conn.close()
        raise PhaseDiagramTimeoutError(
            f"PhaseDiagram timed out after {timeout_value} s ({context})"
        )
    if not parent_conn.poll():
        exitcode = proc.exitcode
        parent_conn.close()
        raise RuntimeError(
            f"PhaseDiagram worker exited without result ({context}), exitcode={exitcode}"
        )
    status, payload = parent_conn.recv()
    parent_conn.close()
    if status == "ok":
        return float(payload)
    raise RuntimeError(f"PhaseDiagram worker failed ({context}):\n{payload}")


def _is_too_close_get_input_set_error(exc: Exception) -> bool:
    return (
        exc.__class__.__name__ == "SymmetryUndeterminedError"
        and "too close distance between atoms" in str(exc).lower()
    )


def relax_from_poscars(
    poscar_paths: Sequence[str | Path],
    steps_max: int = 400,
    outdir: Path | str = "poscar_relaxed",
    allow_run_params_fallback: bool = False,
) -> List[Dict[str, str | float]]:
    """
    Relax a list of POSCAR/CONTCAR files with MatterSim and save the relaxed
    structures in a target directory.

    Parameters
    ----------
    poscar_paths : Sequence[str | Path]
        Iterable of POSCAR/CONTCAR file paths.
    steps_max : int, default 400
        Maximum number of relaxation steps in MatterSim.
    outdir : Path | str, default "poscar_relaxed"
        Directory where relaxed structures are saved.
        Each file will be named ``<stem>_relaxed.vasp``.
    Returns
    -------
    list[dict]
        Same schema as ``collect_mp_entries``::

            {
              "material_id": <file stem>,
              "formula":     <chemical formula>,
              "E_initial (eV)": <total E before relaxation>,
              "E_final   (eV)": <total E after relaxation>,
              "ΔE (eV)":       <energy change>,
              "file":          <path to relaxed POSCAR>
            }
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Hubbard U 等の情報を取得するための VASP 入力ジェネレータ
    try:
        flow_maker = _new_mp_gga_double_relax_static_maker()
        static_maker = flow_maker.static_maker
        gen = static_maker.input_set_generator  # VaspInputGenerator
    except ImportError:
        if not allow_run_params_fallback:
            raise
        gen = None

    results: List[Dict[str, str | float]] = []
    failure_list: List[str] = []

    total_start = time.time()
    for idx, poscar_path in enumerate(
        tqdm(poscar_paths, desc="Relaxing POSCARs"), start=1
    ):
        poscar_path = Path(poscar_path)
        mp_id = poscar_path.stem  # use file stem as an identifier

        struct = Structure.from_file(poscar_path)
        relax_start = time.time()
        try:
            # データに、Hubbard U などの情報を追加する
            if gen is None:
                run_params = {"run_type": "GGA", "is_hubbard": False, "hubbards": {}}
            else:
                vis = gen.get_input_set(struct, potcar_spec=True)
                incar = vis.incar
                elems = list(vis.poscar.site_symbols)
                run_params = extract_params_from_vasp_input(incar, elems)

            record = _relax_and_save_with_retry(
                mp_id=mp_id,
                struct=struct,
                outdir_path=outdir,
                steps_max=steps_max,
            )
            _append_timing(
                "mattersim_relax",
                time.time() - relax_start,
                material_id=mp_id,
                pid=os.getpid(),
            )
            record.update(run_params)
            results.append(record)

            print(
                f"{record['formula']} ({mp_id}) "
                f"{record['E_final (eV)']:.4f} eV "
                f"(ΔE = {record['ΔE (eV)']:+.3f} eV) → saved {Path(record['file']).name}"
            )

        except Exception as exc:
            print(f"Error processing {mp_id}: {exc}")
            failure_list.append(mp_id)
            skip_reason = (
                SKIP_EHULL_REASON_TOO_CLOSE
                if _is_too_close_get_input_set_error(exc)
                else None
            )
            failed_record = {
                "material_id": mp_id,
                "formula": struct.formula,
                "E_initial (eV)": 999.0,  # Magic value for failure
                "E_final (eV)": 999.0,  # Magic value for failure
                "ΔE (eV)": 999.0,  # Magic value for failure
                "file": str(poscar_path),
                "status": "failed",
                "error": str(exc),
            }
            if skip_reason is not None:
                failed_record["skip_ehull_mp_reason"] = skip_reason
            run_params = {"run_type": "GGA", "is_hubbard": False, "hubbards": {}}
            failed_record.update(run_params)
            results.append(failed_record)
            _append_timing(
                "mattersim_relax_fail",
                time.time() - relax_start,
                material_id=mp_id,
                pid=os.getpid(),
            )

        # Jupyter-friendly progress refresh
        clear_output(wait=True)

        # Extra GPU / CPU memory cleanup, in case `relax_and_save` is skipped
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    clear_output()
    print("Relaxation complete.")
    if failure_list:
        print("Failed to process:", failure_list)

    _append_timing(
        "mattersim_relax_batch",
        time.time() - total_start,
        count=len(poscar_paths),
        outdir=str(outdir),
        pid=os.getpid(),
    )
    return results


# In[14]:


from pymatgen.core import Element
from typing import Sequence


def chemsys_by_atomic_number(elements: Sequence[str]) -> str:
    """
    Return a hyphen-separated chemical system string with element symbols
    sorted in descending order of atomic number.

    Parameters
    ----------
    elements : Sequence[str]
        Iterable of element symbols, e.g. ``["Li", "O"]``.

    Returns
    -------
    str
        Hyphen-joined string such as ``"Li-O"``.
    """
    # Sort by atomic number (Element.Z) descending and join with hyphens
    return "-".join(sorted(elements, key=lambda e: Element(e).Z, reverse=True))


# Example


# In[15]:


from pathlib import Path
from typing import List, Sequence

from pymatgen.io.vasp import Poscar


def get_elements_from_poscar(path: Path | str) -> List[str]:
    """
    Read a VASP POSCAR/CONTCAR file and return a list of element symbols
    contained in the structure (unique, in order of appearance).

    Parameters
    ----------
    path : str or Path
        Path to a POSCAR/CONTCAR file.

    Returns
    -------
    list[str]
        Element symbols appearing in the file, e.g. ``["Li", "O"]``.

    Notes
    -----
    - The order is exactly the same as in the sixth line of POSCAR
      (the line with element symbols), which is also the order used by VASP.
    - If the POSCAR line has been edited to include *selective dynamics*
      flags or other custom fields, `pymatgen` still parses the symbols
      correctly.
    """
    poscar = Poscar.from_file(path)
    return [str(el) for el in poscar.site_symbols]


# In[16]:


from typing import List, Dict, Any


def _elements_from_chemsys(chemsys: str) -> List[str]:
    return [e.strip() for e in chemsys.replace(",", "-").split("-") if e.strip()]


def _normalize_chemsys(chemsys: str) -> str:
    return chemsys_by_atomic_number(_elements_from_chemsys(chemsys))


def _log_prefetch_progress(
    label: str,
    index: int,
    total: int,
    start_time: float,
    item_id: Optional[str] = None,
) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    elapsed = time.time() - start_time
    rate = elapsed / index if index else 0.0
    remaining = max(total - index, 0) if total else 0
    eta = rate * remaining if rate else 0.0
    total_label = f"{index}/{total}" if total else f"{index}/?"
    item_label = f" id={item_id}" if item_id else ""
    print(
        f"[{label}] {total_label} {ts} "
        f"elapsed={elapsed:.1f}s eta={eta:.1f}s{item_label}"
    )


def _log_stage_step(index: int, total: int, label: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{index}/{total}. {label}] {ts}")


def _chemsys_has_agm(chemsys: str, relax_result_db_path: Path) -> bool:
    saved_path = relax_result_db_path.parent / f"relax_results_{chemsys}.json.gz"
    if not saved_path.exists():
        return False
    try:
        existing = _load_json_gz(saved_path)
    except Exception:
        return False
    return any(_record_has_agm(rec) for rec in existing)


def attach_ehull(mp_alex_collection: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach energy above hull (e_hull) to each record and return the collection."""
    total_start = time.time()
    mp_entries = make_entries_from_results(mp_alex_collection)
    entries_elapsed = time.time() - total_start
    _append_timing(
        "attach_ehull_entries",
        entries_elapsed,
        count=len(mp_entries),
        pid=os.getpid(),
    )
    phase_start = time.time()
    compat = MaterialsProject2020Compatibility(
        check_potcar_hash=False, check_potcar=False
    )
    proc = compat.process_entries(mp_entries, clean=True)
    phsDgm = PhaseDiagram(proc)
    _append_timing(
        "attach_ehull_phase",
        time.time() - phase_start,
        count=len(proc),
        pid=os.getpid(),
    )

    collection_with_ehull = []
    for entry, collect_data in zip(mp_entries, mp_alex_collection):
        assert entry.entry_id == collect_data["material_id"] + "_ms"
        e_hull = phsDgm.get_e_above_hull(entry)
        collect_data["e_hull"] = e_hull
        collection_with_ehull.append(collect_data)
    _append_timing(
        "attach_ehull_total",
        time.time() - total_start,
        count=len(collection_with_ehull),
        pid=os.getpid(),
    )
    return collection_with_ehull


# In[17]:


def check_some_data_is_from_Alexandria(
    mp_alex_collection: List[Dict[str, Any]]
) -> bool:
    """
    Check if all entries in the collection are from Materials Project.

    Parameters
    ----------
    mp_alex_collection : List[Dict[str, Any]]
        The collection of material data.

    Returns
    -------
    bool
        True if all entries are from Materials Project, False otherwise.
    """
    alex_data: bool = False
    for mp_id in [data_dict["material_id"] for data_dict in mp_alex_collection]:
        if mp_id.startswith("agm"):
            alex_data = True
            break
    return alex_data


# In[18]:


# ## Convex Hull DFT計算に必要なPOSCARを作成するための関数群

# In[19]:


from pathlib import Path
from typing import List, Union, Any, Iterable
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher


def append_if_novel_structure(
    new_path: Union[str, Path],
    path_list: List[Union[str, Path]],
    *,
    anonymous: bool = False,
    **matcher_kwargs: Any,
) -> bool:
    """Append new_path to path_list if its structure is not equivalent to any in path_list.

    Args:
        new_path: Path to the new POSCAR/CONTCAR file.
        path_list: List of existing POSCAR/CONTCAR paths. Will be mutated in place.
        anonymous: If True, ignore species (geometry-only match) via `fit_anonymous`.
        **matcher_kwargs: Extra args for `StructureMatcher(...)`
            (e.g., ltol, stol, angle_tol, primitive_cell, scale, attempt_supercell).

    Returns:
        bool: True if appended (i.e., novel structure), False if a match was found and nothing was appended.
    """
    new_path = Path(new_path)
    s_new = Structure.from_file(str(new_path))

    sm = StructureMatcher(**matcher_kwargs)

    for p in path_list:
        s_old = Structure.from_file(str(p))
        is_match = sm.fit_anonymous(s_new, s_old) if anonymous else sm.fit(s_new, s_old)
        if is_match:
            return False  # 既存と同一 → appendしない

    # どれとも一致しなかった → append
    path_list.append(new_path)
    return True


def _composition_ratio(struct: Structure) -> dict[str, float]:
    comp = struct.composition.fractional_composition
    return {str(el): float(amount) for el, amount in comp.items()}


def _ratios_close(
    ratio_a: dict[str, float],
    ratio_b: dict[str, float],
    *,
    tol: float = 0.05,
) -> bool:
    keys = set(ratio_a) | set(ratio_b)
    for key in keys:
        if abs(ratio_a.get(key, 0.0) - ratio_b.get(key, 0.0)) > tol:
            return False
    return True


def _ratio_bucket_key(
    ratio: dict[str, float],
    elements: Sequence[str],
    *,
    bucket_size: float,
) -> tuple[int, ...]:
    return tuple(int(round(ratio.get(el, 0.0) / bucket_size)) for el in elements)


def _neighbor_bucket_keys(
    base: tuple[int, ...],
    *,
    delta: int = 1,
) -> Iterable[tuple[int, ...]]:
    if not base:
        return []
    keys = [()]
    for val in base:
        next_keys = []
        for offset in range(-delta, delta + 1):
            next_keys.extend([k + (val + offset,) for k in keys])
        keys = next_keys
    return keys


def append_if_novel_structure_ratio(
    new_path: Union[str, Path],
    path_list: List[Union[str, Path]],
    ratio_list: List[dict[str, float]],
    *,
    ratio_tol: float = 0.05,
    anonymous: bool = False,
    s_new: Optional[Structure] = None,
    **matcher_kwargs: Any,
) -> bool:
    """Append new_path if no matching structure within ratio_tol is found."""
    new_path = Path(new_path)
    s_new = s_new or Structure.from_file(str(new_path))
    ratio_new = _composition_ratio(s_new)

    sm = StructureMatcher(**matcher_kwargs)
    for p, ratio_old in zip(path_list, ratio_list):
        if not _ratios_close(ratio_new, ratio_old, tol=ratio_tol):
            continue
        s_old = Structure.from_file(str(p))
        is_match = sm.fit_anonymous(s_new, s_old) if anonymous else sm.fit(s_new, s_old)
        if is_match:
            return False

    path_list.append(new_path)
    ratio_list.append(ratio_new)
    return True


# In[21]:


from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from pymatgen.core import Structure


# In[27]:


import argparse


# ------------------------------------------------------------
# Spec-compliant Stage1–4 pipeline (legacy stages are removed)
# ------------------------------------------------------------
def _list_poscars(poscar_dir: Path) -> List[Path]:
    return sorted(Path(poscar_dir).glob("*.vasp"))


def _proposal_chemsys_map(poscar_paths: Sequence[Path]) -> dict:
    return {
        p: chemsys_by_atomic_number(get_elements_from_poscar(p)) for p in poscar_paths
    }


def _ensure_prop_relaxed(
    poscar_path: Path,
    prop_relaxed_dir: Path,
    steps_max: int,
    allow_run_params_fallback: bool = False,
) -> dict:
    records = relax_from_poscars(
        [poscar_path],
        steps_max=steps_max,
        outdir=prop_relaxed_dir,
        allow_run_params_fallback=allow_run_params_fallback,
    )
    return records[0] if records else {}


def _short_relaxed_name(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    try:
        return Path(str(value)).name
    except Exception:
        return str(value)


def _log_stage2_relax_event(
    proposal_fname: str,
    proposal_progress: Optional[str],
    relaxed_file: Optional[str],
    ref_progress: Optional[tuple[int, int]],
    relax_status: str,
    chem_sys: Optional[str] = None,
    p_dir: Optional[str] = None,
) -> None:
    record = {
        "fname": proposal_fname,
        "relaxed_file": _short_relaxed_name(relaxed_file),
        "relax_status": relax_status,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if proposal_progress is not None:
        record["proposal_progress"] = proposal_progress
    if ref_progress is not None:
        done, total = ref_progress
        record["ref_progress"] = f"{done}/{total}"
    if chem_sys is not None:
        record["chem_sys"] = chem_sys
    if p_dir is not None:
        record["p_dir"] = p_dir
    _append_progress("stage2", record)


def _log_stage4_relax_event(
    proposal_fname: str,
    proposal_progress: Optional[str],
    relaxed_file: Optional[str],
    ref_progress: Optional[tuple[int, int]],
    relax_status: str,
    chem_sys: Optional[str] = None,
    p_dir: Optional[str] = None,
) -> None:
    record = {
        "fname": proposal_fname,
        "relaxed_file": _short_relaxed_name(relaxed_file),
        "relax_status": relax_status,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if proposal_progress is not None:
        record["proposal_progress"] = proposal_progress
    if ref_progress is not None:
        done, total = ref_progress
        record["ref_progress"] = f"{done}/{total}"
    if chem_sys is not None:
        record["chem_sys"] = chem_sys
    if p_dir is not None:
        record["p_dir"] = p_dir
    _append_progress("stage4", record)


def _log_stage4_postprocess_event(
    step: str,
    elapsed_s: float,
    chemsys: Optional[str] = None,
    fname: Optional[str] = None,
) -> None:
    record = {
        "step": step,
        "elapsed_s": float(elapsed_s),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if chemsys is not None:
        record["chemsys"] = chemsys
    if fname is not None:
        record["fname"] = fname
    _append_progress("stage4_postprocess", record)


def _collect_records_for_phase_with_log(
    target_chemsys: str,
    mp_cache: Path,
    table: str,
    steps_max: int,
    proposal_fname: str,
    proposal_progress: Optional[str],
    known_ids: set[str],
    p_dir: Optional[str] = None,
) -> List[dict]:
    saved_path = GLOBAL_RELAX_DIR / f"relax_results_{target_chemsys}.json.gz"
    cached_file_records = _load_relax_results_file(saved_path)
    if cached_file_records:
        total = len(cached_file_records)
        for idx, rec in enumerate(cached_file_records, start=1):
            if rec.get("material_id"):
                known_ids.add(rec["material_id"])
            _log_stage2_relax_event(
                proposal_fname,
                proposal_progress,
                rec.get("file") or rec.get("material_id"),
                (idx, total),
                "skip_relax_results_gz",
                rec.get("chem_sys") or target_chemsys,
                p_dir=p_dir,
            )
        return cached_file_records

    cached = _records_for_chemsys_from_cache(mp_cache, table, target_chemsys)
    db_existing = _load_json_db(DEFAULT_RELAX_DB_PATH)
    relaxed: List[dict] = []
    total = len(cached)
    for idx, rec in enumerate(cached, start=1):
        mid = rec["material_id"]
        record = None
        status = None
        if mid in known_ids:
            record = db_existing.get(mid)
            if record is None:
                db_existing = _load_json_db(DEFAULT_RELAX_DB_PATH)
                record = db_existing.get(mid)
            if record is not None:
                status = "skipped_cache"
        if record is None:
            lock_path = _material_lock_path(mid)
            with _InterProcessFileLock(lock_path):
                db_existing = _load_json_db(DEFAULT_RELAX_DB_PATH)
                record = db_existing.get(mid)
                if record is not None:
                    status = "skipped_cache"
                else:
                    record = _relax_and_save_with_retry(
                        mp_id=mid,
                        struct=rec["structure"],
                        outdir_path=GLOBAL_RELAX_DIR / "mp_relaxed",
                        steps_max=steps_max,
                    )
                    _, err = _assign_run_params_to_record(
                        record,
                        allow_fallback=False,
                        context=f"stage2 relax ref {mid}",
                    )
                    if err is not None:
                        raise RuntimeError(err)
                    record["source"] = rec.get("source", "mp")
                    db_existing[mid] = record
                    # Persist after each record to avoid losing progress on interruption.
                    _save_json_db(DEFAULT_RELAX_DB_PATH, {mid: record})
                    status = "relaxed"
            known_ids.add(mid)
        if record.get("status") == "failed":
            status = "failed"
        elif _infer_record_source(record) == "mp":
            changed, err = _assign_run_params_to_record(
                record,
                allow_fallback=False,
                context=f"stage2 ensure run_params {mid}",
            )
            if err is not None:
                raise RuntimeError(err)
            if changed:
                _save_json_db(DEFAULT_RELAX_DB_PATH, {mid: record})
        _log_stage2_relax_event(
            proposal_fname,
            proposal_progress,
            record.get("file") or record.get("material_id"),
            (idx, total),
            status,
            record.get("chem_sys") or target_chemsys,
            p_dir=p_dir,
        )
        relaxed.append(record)
    _backfill_collection_chemsys(relaxed, "mp_phase1")
    return relaxed


def _records_for_chemsys_from_cache(
    cache_path: Path,
    table: str,
    target_chemsys: str,
) -> List[dict]:
    entries = _load_cached_entries(cache_path, table, target_chemsys)
    out: List[dict] = []
    for rec in entries:
        struct = Structure.from_dict(json.loads(rec["data_json"]))
        out.append(
            {
                "material_id": rec["material_id"],
                "formula": struct.formula,
                "E_initial (eV)": None,
                "E_final (eV)": None,
                "ΔE (eV)": None,
                "file": None,
                "chem_sys": chemsys_by_atomic_number(struct.symbol_set),
                "source": rec.get("source", "mp"),
                "structure": struct,
                "e_hull": rec.get("e_hull"),
            }
        )
    return out


def _relax_reference_records(
    records: List[dict],
    outdir: Path,
    steps_max: int,
    existing_db: dict,
) -> List[dict]:
    outdir.mkdir(parents=True, exist_ok=True)
    relaxed: List[dict] = []
    for rec in records:
        mid = rec["material_id"]
        if mid in existing_db:
            cached = existing_db[mid]
            if _infer_record_source(cached) == "mp" and not _is_failed_relax_record(
                cached
            ):
                changed, err = _assign_run_params_to_record(
                    cached,
                    allow_fallback=False,
                    context=f"relax_reference_records cached {mid}",
                )
                if err is not None:
                    raise RuntimeError(err)
                if changed:
                    _save_json_db(DEFAULT_RELAX_DB_PATH, {mid: cached})
            relaxed.append(cached)
            continue
        struct = rec["structure"]
        record = _relax_and_save_with_retry(
            mp_id=mid,
            struct=struct,
            outdir_path=outdir,
            steps_max=steps_max,
        )
        _, err = _assign_run_params_to_record(
            record,
            allow_fallback=False,
            context=f"relax_reference_records {mid}",
        )
        if err is not None:
            raise RuntimeError(err)
        record["source"] = rec.get("source", "mp")
        relaxed.append(record)
        existing_db[mid] = record
    return relaxed


def _collect_records_for_phase(
    target_chemsys: str,
    mp_cache: Path,
    table: str,
    steps_max: int,
) -> List[dict]:
    saved_path = GLOBAL_RELAX_DIR / f"relax_results_{target_chemsys}.json.gz"
    cached_file_records = _load_relax_results_file(saved_path)
    if cached_file_records:
        return cached_file_records
    cached = _records_for_chemsys_from_cache(mp_cache, table, target_chemsys)
    db_existing = _load_json_db(DEFAULT_RELAX_DB_PATH)
    relaxed = _relax_reference_records(
        cached,
        GLOBAL_RELAX_DIR / "mp_relaxed",
        steps_max,
        db_existing,
    )
    if relaxed:
        _save_json_db(DEFAULT_RELAX_DB_PATH, {r["material_id"]: r for r in relaxed})
    _backfill_collection_chemsys(relaxed, "mp_phase1")
    return relaxed


def _count_ref_candidates(
    chemsys: str, mp_cache: Path, alex_cache: Optional[Path] = None
) -> int:
    total = 0
    for rec in _records_for_chemsys_from_cache(mp_cache, "mp_entries", chemsys):
        total += 1
    if alex_cache and alex_cache.exists():
        for rec in _records_for_chemsys_from_cache(alex_cache, "alex_entries", chemsys):
            total += 1
    saved_path = GLOBAL_RELAX_DIR / f"relax_results_{chemsys}.json.gz"
    if saved_path.exists():
        try:
            total = max(total, len(_load_json_gz(saved_path)))
        except Exception:
            pass
    return total


def _write_relax_results_for_chemsys(records: List[dict], chemsys: str) -> None:
    """Write relax_results for a chemsys, including failed (999) records."""
    if not records:
        return
    # 親 + 下位系すべての relax_results_<sub>.json.gz をまとめて生成する
    target_list = _enumerate_subsystems(chemsys)
    _update_subchemsys_files(records, GLOBAL_RELAX_DIR, target_chemsys=target_list)


def _compute_ehull_for_proposal(
    mp_records: List[dict],
    proposal_record: dict,
    *,
    timeout_sec: Optional[float] = None,
) -> float:
    mp_records_ok = [rec for rec in mp_records if not _is_failed_relax_record(rec)]
    if not mp_records_ok:
        raise RuntimeError(
            "No valid MP reference records remain after excluding failed relaxations."
        )
    return _compute_ehull_with_references(
        mp_records_ok,
        proposal_record,
        context="stage2_mp",
        timeout_sec=timeout_sec,
    )


def _compute_ehull_with_references(
    reference_records: List[dict],
    proposal_record: dict,
    *,
    context: str,
    timeout_sec: Optional[float] = None,
) -> float:
    entries = _make_entries(reference_records + [proposal_record])
    compat = MaterialsProject2020Compatibility(
        check_potcar_hash=False, check_potcar=False
    )
    proc = compat.process_entries(entries, clean=True)
    mp_refs = [rec for rec in reference_records if _infer_record_source(rec) == "mp"]
    if mp_refs:
        _assert_mp_ok_count_equals_proc_count(
            mp_refs,
            proc,
            entries[:-1],
            context=context,
        )
    alex_refs = [
        rec for rec in reference_records if _infer_record_source(rec) == "alex"
    ]
    if alex_refs:
        _assert_alex_ok_count_equals_proc_count(
            alex_refs,
            proc,
            entries[:-1],
            context=context,
        )
    target_id = entries[-1].entry_id
    assert (not target_id.startswith("mp-")) and (
        not target_id.startswith("agm")
    ), f"target_id must be proposal-derived, but got reference-like id: {target_id}"
    target_entry = next((entry for entry in proc if entry.entry_id == target_id), None)
    if target_entry is None:
        raise ValueError(f"Proposal entry was filtered by compatibility: {target_id}")
    return _compute_ehull_with_timeout(
        proc,
        target_id,
        timeout_sec=timeout_sec,
        context=context,
    )


def stage_prefetch_mp_spec(args) -> None:
    cache_dir = Path(args.cache_dir or DEFAULT_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    mp_cache_path = cache_dir / "mp_requests_cache.sqlite"
    fetch_done_path = cache_dir / "mp_fetch_done.json"
    fetch_done = _load_fetch_done(fetch_done_path)
    query_sig = _query_signature_mp()

    poscars = _list_poscars(Path(args.p_dir))
    if not poscars:
        print("No POSCAR files to prefetch.")
        return
    chemsys_map = _proposal_chemsys_map(poscars)
    total_props = len(poscars)
    done = 0
    ordered_cs = _sort_chemsys_desc(chemsys_map.values())
    for cs in ordered_cs:
        cs_norm = _normalize_chemsys(cs)
        saved_path = GLOBAL_RELAX_DIR / f"relax_results_{cs_norm}.json.gz"
        if saved_path.exists():
            done += 1
            _append_progress(
                "stage1",
                {
                    "chemsys": cs_norm,
                    "progress": f"{done}/{total_props}",
                    "status": "skip_json_gz",
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
            )
            continue
        # Superset skip: if a superset chemsys was fetched and cache has entries
        supers = _find_superset_done(cs_norm, fetch_done)
        if supers and _has_cache_for_chemsys(mp_cache_path, cs_norm):
            subs = _enumerate_subsystems(cs_norm)
            fetch_done[cs_norm] = {"subsystems": subs, "query_signature": query_sig}
            _save_fetch_done(fetch_done_path, fetch_done)
            done += 1
            _append_progress(
                "stage1",
                {
                    "chemsys": cs_norm,
                    "progress": f"{done}/{total_props}",
                    "status": "skip_superset_done",
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
            )
            continue
        manifest = fetch_done.get(cs_norm)
        if manifest and manifest.get("query_signature") == query_sig:
            done += 1
            _append_progress(
                "stage1",
                {
                    "chemsys": cs_norm,
                    "progress": f"{done}/{total_props}",
                    "status": "skip_fetch_done",
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
            )
            continue
        to_cache: List[dict] = []
        subs = _enumerate_subsystems(cs_norm)
        with _InterProcessFileLock(_mp_lock_path(cache_dir)):
            structs = fetch_mp_structures(
                args.mp_api_key,
                _elements_from_chemsys(cs_norm),
                ehull_max=FETCH_EHULL_DB_THRES,
                cache_dir=cache_dir,
            )
        for mid, struct, e_hull in structs:
            to_cache.append(
                {
                    "material_id": mid,
                    "entry_chemsys": chemsys_by_atomic_number(struct.symbol_set),
                    "source": "mp",
                    "data_json": json.dumps(struct.as_dict(), ensure_ascii=True),
                    "e_hull": float(e_hull) if e_hull is not None else None,
                }
            )
        if to_cache:
            inserted = _cache_entries(mp_cache_path, "mp_entries", to_cache)
        else:
            inserted = 0
        fetch_done[cs_norm] = {"subsystems": subs, "query_signature": query_sig}
        _save_fetch_done(fetch_done_path, fetch_done)
        print(f"[Stage1] {cs_norm}: cached {inserted} MP entries")
        done += 1
        _append_progress(
            "stage1",
            {
                "chemsys": cs_norm,
                "progress": f"{done}/{total_props}",
                "status": "fetched",
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )


def stage_phase1_spec(args) -> None:
    cache_dir = Path(args.cache_dir or DEFAULT_CACHE_DIR)
    mp_cache_path = cache_dir / "mp_requests_cache.sqlite"
    fetch_done = _load_fetch_done(cache_dir / "mp_fetch_done.json")
    poscars = _list_poscars(Path(args.p_dir))
    if not poscars:
        print("No POSCAR files for Phase1.")
        return
    prop_relaxed_dir = Path(args.r_dir)
    prop_relaxed_dir.mkdir(parents=True, exist_ok=True)
    screening_path = cache_dir / "screening_results.jsonl"
    proposal_relax_records_path = cache_dir / "proposal_relax_records.jsonl"
    chemsys_to_fetch_path = cache_dir / "chemsys_to_fetch.json"
    records_screen: List[dict] = []
    proposal_relax_records: List[dict] = []
    chemsys_to_fetch: set[str] = set()
    hull_done = 0
    total_props = len(poscars)
    known_ids = _load_known_material_ids(DEFAULT_RELAX_DB_PATH)
    p_dir_label = Path(args.p_dir).name
    for idx, poscar in enumerate(poscars, start=1):
        cs = chemsys_by_atomic_number(get_elements_from_poscar(poscar))
        cs_norm = _normalize_chemsys(cs)
        # proposal relax
        proposal_record = _ensure_prop_relaxed(
            poscar,
            prop_relaxed_dir,
            args.steps_max,
            allow_run_params_fallback=args.allow_run_params_fallback,
        )
        prop_status = "relaxed"
        if not proposal_record or proposal_record.get("status") == "failed":
            prop_status = "failed"
        proposal_progress = f"{idx}/{total_props}"
        _log_stage2_relax_event(
            poscar.stem,
            proposal_progress,
            proposal_record.get("file") if proposal_record else str(poscar),
            None,
            prop_status,
            proposal_record.get("chem_sys") if proposal_record else cs_norm,
            p_dir=p_dir_label,
        )
        fname_for_screen = f"{poscar.parent.name}/{poscar.name}"
        proposal_relax_records.append(
            {
                "fname": fname_for_screen,
                "proposal_id": poscar.stem,
                "chemsys": proposal_record.get("chem_sys", cs_norm),
                "formula": proposal_record.get("formula"),
                "E_initial (eV)": proposal_record.get("E_initial (eV)"),
                "E_final (eV)": proposal_record.get("E_final (eV)"),
                "ΔE (eV)": proposal_record.get("ΔE (eV)"),
                "file": proposal_record.get("file"),
                "run_type": proposal_record.get("run_type", "GGA"),
                "is_hubbard": bool(proposal_record.get("is_hubbard", False)),
                "hubbards": dict(proposal_record.get("hubbards", {}) or {}),
                "status": proposal_record.get("status", "relaxed"),
                "error": proposal_record.get("error"),
                "skip_ehull_mp_reason": proposal_record.get("skip_ehull_mp_reason"),
            }
        )
        skip_ehull_too_close = (
            proposal_record.get("skip_ehull_mp_reason") == SKIP_EHULL_REASON_TOO_CLOSE
        )
        # 参照構造 (MP) の取得
        saved_path = GLOBAL_RELAX_DIR / f"relax_results_{cs_norm}.json.gz"
        ref_records: List[dict] = []
        ref_total = 0
        ref_status = "relaxed_refs"

        if saved_path.exists():
            try:
                repair = _repair_run_params_in_relax_results_file(
                    saved_path,
                    allow_fallback=args.allow_run_params_fallback,
                    verify_mp_compat=False,
                    context=f"stage2 load refs {cs_norm}",
                )
                ref_records = [r for r in repair["records"] if not _record_has_agm(r)]
            except Exception as exc:
                _raise_if_corrupt_file_error(
                    exc, saved_path, f"stage2 load refs {cs_norm}"
                )
                ref_records = []
            for r in ref_records:
                mid = r.get("material_id")
                if mid:
                    known_ids.add(mid)
            ref_total = len(ref_records)
            ref_status = "skip_relax_results_gz"

        if not ref_records:
            manifest = fetch_done.get(cs_norm)
            if not manifest:
                raise RuntimeError(
                    f"Missing MP fetch_done for {cs_norm}. Run Stage1 first or provide relax_results_{cs_norm}.json.gz."
                )
            ref_total = _count_ref_candidates(cs_norm, mp_cache_path)
            ref_records = _collect_records_for_phase_with_log(
                cs_norm,
                mp_cache_path,
                "mp_entries",
                args.steps_max,
                poscar.stem,
                proposal_progress,
                known_ids,
                p_dir=p_dir_label,
            )
            if not ref_records:
                raise RuntimeError(
                    f"No MP records available for {cs_norm}. Ensure Stage1 fetched data or relax_results exists."
                )
            _write_relax_results_for_chemsys(ref_records, cs_norm)

        ref_done = len(ref_records)
        if skip_ehull_too_close:
            records_screen.append(
                {
                    "fname": fname_for_screen,
                    "chemsys": cs_norm,
                    "e_hull_mp": float("nan"),
                }
            )
            print(
                f"[Stage2] {poscar.name}: e_hull_mp=nan "
                f"(skip_reason={SKIP_EHULL_REASON_TOO_CLOSE})"
            )
            hull_done += 1
            _append_progress(
                "stage2",
                {
                    "fname": poscar.stem,
                    "progress": f"{hull_done}/{total_props}",
                    "ref_progress": f"{ref_done}/{ref_total}",
                    "status": "skip_ehull_nan_too_close",
                    "p_dir": p_dir_label,
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
            )
            continue
        try:
            e_hull_mp = _compute_ehull_for_proposal(
                ref_records,
                proposal_record,
                timeout_sec=args.phase_diagram_timeout_sec,
            )
        except PhaseDiagramTimeoutError:
            proposal_relax_records[-1][
                "skip_ehull_mp_reason"
            ] = SKIP_EHULL_REASON_PHASE_DIAGRAM_TIMEOUT
            records_screen.append(
                {
                    "fname": fname_for_screen,
                    "chemsys": cs_norm,
                    "e_hull_mp": float("nan"),
                    "skip_ehull_mp_reason": SKIP_EHULL_REASON_PHASE_DIAGRAM_TIMEOUT,
                }
            )
            print(
                f"[Stage2] {poscar.name}: e_hull_mp=nan "
                f"(skip_reason={SKIP_EHULL_REASON_PHASE_DIAGRAM_TIMEOUT}, "
                f"timeout_sec={args.phase_diagram_timeout_sec})"
            )
            hull_done += 1
            _append_progress(
                "stage2",
                {
                    "fname": poscar.stem,
                    "progress": f"{hull_done}/{total_props}",
                    "ref_progress": f"{ref_done}/{ref_total}",
                    "status": "skip_ehull_nan_phase_diagram_timeout",
                    "p_dir": p_dir_label,
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
            )
            continue
        records_screen.append(
            {"fname": fname_for_screen, "chemsys": cs_norm, "e_hull_mp": e_hull_mp}
        )
        if e_hull_mp < CREATE_DFT_NEEDED_THRES:
            chemsys_to_fetch.add(cs_norm)
        print(f"[Stage2] {poscar.name}: e_hull_mp={e_hull_mp:.4f} eV/atom")
        hull_done += 1
        _append_progress(
            "stage2",
            {
                "fname": poscar.stem,
                "progress": f"{hull_done}/{total_props}",
                "ref_progress": f"{ref_done}/{ref_total}",
                "status": ref_status if ref_records else "missing_refs",
                "p_dir": p_dir_label,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )

    # Merge screening_results.jsonl and chemsys_to_fetch.json with a lock
    lock_path = cache_dir / "screening_results.lock"
    with _InterProcessFileLock(lock_path):
        existing_chemsys_to_fetch = (
            set(_read_json(chemsys_to_fetch_path))
            if chemsys_to_fetch_path.exists()
            else set()
        )
        chemsys_to_fetch |= existing_chemsys_to_fetch

        existing_records = (
            _read_jsonl(screening_path) if screening_path.exists() else []
        )
        merged: Dict[str, dict] = {}
        for rec in existing_records + records_screen:
            key = rec.get("fname")
            if key is None:
                continue
            merged[key] = rec
        _write_jsonl_atomic(screening_path, merged.values())

        existing_prop_records = (
            _read_jsonl(proposal_relax_records_path)
            if proposal_relax_records_path.exists()
            else []
        )
        merged_prop: Dict[str, dict] = {}
        for rec in existing_prop_records + proposal_relax_records:
            key = rec.get("fname")
            if key is None:
                continue
            merged_prop[key] = rec
        _write_jsonl_atomic(proposal_relax_records_path, merged_prop.values())

        _write_json_atomic(chemsys_to_fetch_path, sorted(chemsys_to_fetch))
    print(
        f"[Stage2] screening_results -> {screening_path}, "
        f"proposal_relax_records -> {proposal_relax_records_path}, "
        f"chemsys_to_fetch -> {chemsys_to_fetch_path}"
    )


def stage_prefetch_alex_spec(args) -> None:
    cache_dir = Path(args.cache_dir or DEFAULT_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    alex_cache_path = cache_dir / "alex_requests_cache.sqlite"
    fetch_done_path = cache_dir / "alex_fetch_done.json"
    fetch_done = _load_fetch_done(fetch_done_path)
    query_sig = _query_signature_alex()
    chemsys_to_fetch_path = cache_dir / "chemsys_to_fetch.json"
    if not chemsys_to_fetch_path.exists():
        raise RuntimeError(
            "chemsys_to_fetch.json not found. Run Stage2 (MatterSim_phase1) first."
        )
    chemsys_list = _sort_chemsys_desc(_read_json(chemsys_to_fetch_path))
    print(f"[Stage3] start prefetched chemsys count={len(chemsys_list)}", flush=True)
    log_dir = Path.cwd() / "logs" / "time_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "stage3_light.log").touch(exist_ok=True)
    (log_dir / "stage3_structure.log").touch(exist_ok=True)
    total = len(chemsys_list)
    done = 0
    for cs in chemsys_list:
        cs_norm = _normalize_chemsys(cs)
        print(f"[Stage3] start chemsys={cs_norm}", flush=True)
        saved_path = GLOBAL_RELAX_DIR / f"relax_results_{cs_norm}.json.gz"
        # 前提: Stage2 で生成済みであること
        if not saved_path.exists():
            raise RuntimeError(
                f"relax_results_{cs_norm}.json.gz not found; run Stage2 first"
            )

        has_agm = False
        try:
            existing = _load_json_gz(saved_path)
            has_agm = any(_record_has_agm(r) for r in existing)
        except Exception as exc:
            _raise_if_corrupt_file_error(exc, saved_path, f"stage3 check agm {cs_norm}")
            has_agm = False

        manifest = fetch_done.get(cs_norm)
        if has_agm:
            done += 1
            _append_progress(
                "stage3",
                {
                    "chemsys": cs_norm,
                    "progress": f"{done}/{total}",
                    "status": "skip_json_gz",
                    "skip_reason": "has_agm_in_saved",
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
            )
            continue
        if manifest:
            # query_signature に関係なく、一度取得済みならスキップ
            done += 1
            _append_progress(
                "stage3",
                {
                    "chemsys": cs_norm,
                    "progress": f"{done}/{total}",
                    "status": "skip_fetch_done",
                    "skip_reason": "fetch_done_exists",
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
            )
            continue
        subs = sorted(set(_enumerate_subsystems(cs_norm)))
        to_cache: List[dict] = []
        for sub in subs:
            sub_norm = _normalize_chemsys(sub)
            sub_manifest = fetch_done.get(sub_norm)
            if sub_manifest:
                # 既に取得済みのサブ系はシグネチャに関わらずスキップ
                _append_progress(
                    "stage3",
                    {
                        "chemsys": cs_norm,
                        "sub": sub_norm,
                        "status": "skip_sub_fetch_done",
                        "skip_reason": "sub_fetch_done_exists",
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    },
                )
                continue
            _append_progress(
                "stage3",
                {
                    "chemsys": cs_norm,
                    "sub": sub_norm,
                    "status": "fetch_sub",
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
            )
            print(f"[Stage3] fetch sub={sub_norm} (include_structure=True)", flush=True)
            structs = fetch_alex_structures(
                _elements_from_chemsys(sub_norm),
                ehull_max=FETCH_EHULL_DB_THRES,
                base_url=args.alexandria_base_url,
                cache_dir=cache_dir,
            )
            for mid, struct, e_hull in structs:
                to_cache.append(
                    {
                        "material_id": mid,
                        "entry_chemsys": chemsys_by_atomic_number(struct.symbol_set),
                        "source": "alex",
                        "data_json": json.dumps(struct.as_dict(), ensure_ascii=True),
                        "e_hull": float(e_hull) if e_hull is not None else None,
                    }
                )
        if to_cache:
            inserted = _cache_entries(alex_cache_path, "alex_entries", to_cache)
        else:
            inserted = 0
        _append_progress(
            "stage3",
            {
                "chemsys": cs_norm,
                "status": "sub_fetch_summary",
                "subs_total": len(subs),
                "inserted": inserted,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )
        # 上位 chemsys だけでなく、取得したサブ系も同じ query_signature で登録しておく。
        # これにより、後続でサブ系だけを指定して Stage3 を実行しても skip_fetch_done 判定が働く。
        fetch_done[cs_norm] = {"subsystems": subs, "query_signature": query_sig}
        for sub in subs:
            # 既に同一シグネチャで登録済みなら上書き不要
            if fetch_done.get(sub, {}).get("query_signature") == query_sig:
                continue
            fetch_done[sub] = {"subsystems": [sub], "query_signature": query_sig}
        _save_fetch_done(fetch_done_path, fetch_done)
        print(f"[Stage3] {cs_norm}: cached {inserted} Alexandria entries")
        done += 1
        _append_progress(
            "stage3",
            {
                "chemsys": cs_norm,
                "progress": f"{done}/{total}",
                "status": "fetched",
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )


def _gather_relax_records_for_chemsys(
    chemsys: str,
    mp_cache: Path,
    alex_cache: Path,
    steps_max: int,
    allow_run_params_fallback: bool,
    known_ids: set[str],
    proposal_fname: str,
    proposal_progress: Optional[str],
    p_dir: Optional[str] = None,
) -> List[dict]:
    saved_path = GLOBAL_RELAX_DIR / f"relax_results_{chemsys}.json.gz"
    existing_records: List[dict] = []
    has_agm = False
    if saved_path.exists():
        repair = _repair_run_params_in_relax_results_file(
            saved_path,
            allow_fallback=allow_run_params_fallback,
            verify_mp_compat=False,
            context=f"stage4 gather {chemsys}",
        )
        existing_records = repair["records"]
        has_agm = any(_record_has_agm(r) for r in existing_records)
        if has_agm:
            total = len(existing_records)
            for idx, rec in enumerate(existing_records, start=1):
                mid = rec.get("material_id")
                if mid:
                    known_ids.add(mid)
                _log_stage4_relax_event(
                    proposal_fname,
                    proposal_progress,
                    rec.get("file") or rec.get("material_id"),
                    (idx, total),
                    "skip_relax_results_gz",
                    rec.get("chem_sys") or chemsys,
                    p_dir=p_dir,
                )
            return existing_records

    # 既存 (MP-only) をベースに、キャッシュから不足分を補う
    for rec in existing_records:
        mid = rec.get("material_id")
        if mid:
            known_ids.add(mid)

    mp_records = _records_for_chemsys_from_cache(mp_cache, "mp_entries", chemsys)
    alex_records = _records_for_chemsys_from_cache(alex_cache, "alex_entries", chemsys)

    # 重複を除いた候補
    def _dedup(records: List[dict]) -> List[dict]:
        uniq: List[dict] = []
        seen: set[str] = set()
        for r in records:
            mid = r["material_id"]
            if mid in seen:
                continue
            seen.add(mid)
            uniq.append(r)
        return uniq

    mp_records = _dedup(
        [
            r
            for r in mp_records
            if r["material_id"]
            not in {rec.get("material_id") for rec in existing_records}
        ]
    )
    alex_records = _dedup(
        [
            r
            for r in alex_records
            if r["material_id"]
            not in {rec.get("material_id") for rec in existing_records}
        ]
    )

    existing_db = _load_json_db(DEFAULT_RELAX_DB_PATH)
    merged: List[dict] = list(existing_records)
    all_records = mp_records + alex_records
    total = len(all_records) + len(existing_records)

    for idx, rec in enumerate(all_records, start=1):
        mid = rec["material_id"]
        record = None
        status = None
        if mid in known_ids:
            record = existing_db.get(mid)
            if record is None:
                existing_db = _load_json_db(DEFAULT_RELAX_DB_PATH)
                record = existing_db.get(mid)
            if record is not None:
                status = "skipped_cache"
        if record is None:
            lock_path = _material_lock_path(mid)
            with _InterProcessFileLock(lock_path):
                existing_db = _load_json_db(DEFAULT_RELAX_DB_PATH)
                record = existing_db.get(mid)
                if record is not None:
                    status = "skipped_cache"
                else:
                    record = _relax_and_save_with_retry(
                        mp_id=mid,
                        struct=rec["structure"],
                        outdir_path=GLOBAL_RELAX_DIR / "mp_relaxed",
                        steps_max=steps_max,
                    )
                    _, err = _assign_run_params_to_record(
                        record,
                        allow_fallback=False,
                        context=f"stage4 relax ref {mid}",
                    )
                    if err is not None:
                        raise RuntimeError(err)
                    record["source"] = rec.get("source", "mp")
                    existing_db[mid] = record
                    _save_json_db(DEFAULT_RELAX_DB_PATH, {mid: record})
                    status = "relaxed"
            known_ids.add(mid)
        if record.get("status") == "failed":
            status = "failed"
        elif _infer_record_source(record) in {"mp", "alex"}:
            changed, err = _assign_run_params_to_record(
                record,
                allow_fallback=False,
                context=f"stage4 ensure run_params {mid}",
            )
            if err is not None:
                raise RuntimeError(err)
            if changed:
                _save_json_db(DEFAULT_RELAX_DB_PATH, {mid: record})
        _log_stage4_relax_event(
            proposal_fname,
            proposal_progress,
            record.get("file") or record.get("material_id"),
            (len(existing_records) + idx, total),
            status or "skipped_cache",
            record.get("chem_sys") or chemsys,
            p_dir=p_dir,
        )
        merged.append(record)

    _backfill_collection_chemsys(merged, "phase2_merge")
    if saved_path.exists():
        _write_json_gz(saved_path, merged)
    return merged


def _filter_records(records: List[dict], source: Optional[str]) -> List[dict]:
    if source is None:
        return records
    out = []
    for r in records:
        mid = r.get("material_id", "")
        # source 欠落時は material_id プレフィックスから推定し、レコードに書き戻す
        inferred = "alex" if mid.startswith("agm") else "mp"
        src = r.get("source")
        if src not in ("mp", "alex"):
            r["source"] = inferred
            src = inferred
        if src == source:
            out.append(r)
    return out


def _make_entries(records: List[dict]) -> List[ComputedStructureEntry]:
    # Ensure minimal run params exist
    for rec in records:
        rec.setdefault("run_type", "GGA")
        rec.setdefault("is_hubbard", False)
        rec.setdefault("hubbards", {})
    return make_entries_from_results(records)


def _compute_ehulls_phase2(
    mp_records: List[dict],
    alex_records: List[dict],
    proposal_record: dict,
    e_hull_mp_from_stage2: Optional[float] = None,
) -> Tuple[float, float]:
    mp_records_ok = [rec for rec in mp_records if not _is_failed_relax_record(rec)]
    alex_records_ok = [rec for rec in alex_records if not _is_failed_relax_record(rec)]

    if not mp_records_ok:
        raise RuntimeError(
            "No valid MP reference records remain after excluding failed relaxations."
        )

    if e_hull_mp_from_stage2 is None:
        e_hull_mp = _compute_ehull_with_references(
            mp_records_ok,
            proposal_record,
            context="stage4_mp",
        )
    else:
        e_hull_mp = float(e_hull_mp_from_stage2)
        entries_mp = _make_entries(mp_records_ok)
        compat = MaterialsProject2020Compatibility(
            check_potcar_hash=False,
            check_potcar=False,
        )
        proc_mp = compat.process_entries(entries_mp, clean=True)
        _assert_mp_ok_count_equals_proc_count(
            mp_records_ok,
            proc_mp,
            entries_mp,
            context="stage4_mp_refs_only",
        )

    e_hull_mp_alex = float("nan")
    if alex_records_ok:
        e_hull_mp_alex = _compute_ehull_with_references(
            mp_records_ok + alex_records_ok,
            proposal_record,
            context="stage4_mp_alex",
        )
    return e_hull_mp, e_hull_mp_alex


def stage_export_mp_only(args) -> None:
    """
    MP のみを用いて Stage4 相当の出力を行う簡易モード。
    前提: Stage1/Stage2 完了済み（screening_results.jsonl, relax_results_<chemsys>.json.gz が存在）
    Alexandria は使わない。
    """
    cache_dir = Path(args.cache_dir or DEFAULT_CACHE_DIR)
    poscars = _list_poscars(Path(args.p_dir))
    if not poscars:
        print("No POSCAR files for MP-only export.")
        return
    p_dir_label = Path(args.p_dir).name

    screening_path = cache_dir / "screening_results.jsonl"
    screening_map = (
        {rec.get("fname"): rec for rec in _read_jsonl(screening_path)}
        if screening_path.exists()
        else {}
    )

    prop_relaxed_dir = Path(args.r_dir)
    prop_relaxed_dir.mkdir(parents=True, exist_ok=True)
    result_dir = Path(args.rr_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    dft_out_dir = Path(args.s_dir)
    dft_out_dir.mkdir(parents=True, exist_ok=True)

    # Stage2 の結果を再利用。無ければ missing として NaN で埋める。
    def _load_cached_proposal_record(poscar: Path) -> dict | None:
        relaxed_path = prop_relaxed_dir / f"{poscar.stem}_relaxed.vasp"
        failed_path = prop_relaxed_dir / f"{poscar.stem}_failed.vasp"
        target_path = (
            relaxed_path
            if relaxed_path.exists()
            else failed_path if failed_path.exists() else None
        )
        if target_path is None:
            return None
        try:
            struct = Structure.from_file(target_path)
        except Exception:
            return None
        status = "failed" if target_path.name.endswith("_failed.vasp") else "relaxed"
        return {
            "material_id": poscar.stem,
            "formula": struct.formula,
            "E_initial (eV)": float("nan"),
            "E_final (eV)": float("nan"),
            "ΔE (eV)": float("nan"),
            "file": str(target_path),
            "chem_sys": chemsys_by_atomic_number(struct.symbol_set),
            "run_type": "GGA",
            "is_hubbard": False,
            "hubbards": {},
            "status": status,
        }

    proposal_relaxed_map = {}
    for poscar in poscars:
        rec = _load_cached_proposal_record(poscar)
        if rec is None:
            rec = {
                "material_id": poscar.stem,
                "formula": None,
                "E_initial (eV)": float("nan"),
                "E_final (eV)": float("nan"),
                "ΔE (eV)": float("nan"),
                "file": None,
                "chem_sys": None,
                "run_type": "GGA",
                "is_hubbard": False,
                "hubbards": {},
                "status": "missing_stage2",
            }
        proposal_relaxed_map[poscar] = rec

    def _chemsys_for_poscar(poscar: Path) -> str:
        rec = proposal_relaxed_map[poscar]
        if rec.get("chem_sys"):
            return rec["chem_sys"]
        return chemsys_by_atomic_number(get_elements_from_poscar(poscar))

    def _infer_source(record: dict) -> str:
        return _infer_record_source(record)

    def _is_failed_relax_record_local(record: dict) -> bool:
        return _is_failed_relax_record(record)

    def _load_structure(path_like: Optional[str]) -> Optional[Structure]:
        if not path_like:
            return None
        try:
            p = Path(str(path_like))
        except Exception:
            return None
        if not p.exists():
            return None
        try:
            return Structure.from_file(p)
        except Exception:
            return None

    def _reduced_formula_from_record(record: dict) -> str:
        formula = record.get("formula")
        if formula:
            try:
                return Composition(formula).reduced_formula
            except Exception:
                pass
        s_ref = _load_structure(record.get("file"))
        if s_ref is None:
            return ""
        return Composition(s_ref.composition).reduced_formula

    # per-chemsys で same_formula の MP 参照構造リストを持つ
    mp_formula_structs_by_cs: dict[str, dict[str, list[Structure]]] = {}
    mp_reference_rows_by_cs: dict[str, list[dict]] = {}
    reference_csv_columns = ["参照材料名", "chemsys", "e_hull_mp", "reduced_formula"]
    all_chemsys = sorted({_normalize_chemsys(_chemsys_for_poscar(p)) for p in poscars})
    total_chemsys = len(all_chemsys)
    _append_progress(
        "stage4",
        {
            "status": "export_start",
            "p_dir": p_dir_label,
            "num_poscars": len(poscars),
            "num_chemsys": total_chemsys,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )
    for prep_idx, cs in enumerate(all_chemsys, start=1):
        cs_norm = _normalize_chemsys(cs)
        _append_progress(
            "stage4",
            {
                "status": "chemsys_prepare_start",
                "p_dir": p_dir_label,
                "chemsys": cs_norm,
                "progress": f"{prep_idx}/{total_chemsys}",
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )
        saved_path = GLOBAL_RELAX_DIR / f"relax_results_{cs_norm}.json.gz"
        if not saved_path.exists():
            raise RuntimeError(
                f"Missing relax_results_{cs_norm}.json.gz. Run Stage1/2 first."
            )
        repair = _repair_run_params_in_relax_results_file(
            saved_path,
            allow_fallback=args.allow_run_params_fallback,
            verify_mp_compat=False,
            context=f"export_mp_only {cs_norm}",
        )
        ref_records = repair["records"]
        formula_map: dict[str, list[Structure]] = {}
        mp_records_ok: list[dict] = []
        for r in ref_records:
            if _infer_source(r) != "mp":
                continue
            if _is_failed_relax_record_local(r):
                continue
            mp_records_ok.append(r)
            s_ref = _load_structure(r.get("file"))
            if s_ref is None:
                continue
            rf = Composition(s_ref.composition).reduced_formula
            formula_map.setdefault(rf, []).append(s_ref)
        if not mp_records_ok:
            raise RuntimeError(
                f"No valid MP reference records remain for {cs_norm} after excluding failed relaxations."
            )
        entries_mp = _make_entries(mp_records_ok)
        compat = MaterialsProject2020Compatibility(
            check_potcar_hash=False, check_potcar=False
        )
        proc_mp = compat.process_entries(entries_mp, clean=True)
        _assert_mp_ok_count_equals_proc_count(
            mp_records_ok,
            proc_mp,
            entries_mp,
            context=f"export_mp_only {cs_norm}",
        )
        pd_mp = PhaseDiagram(proc_mp)
        proc_map = {entry.entry_id: entry for entry in proc_mp}
        rows: list[dict] = []
        for rec in mp_records_ok:
            material_id = str(rec.get("material_id", ""))
            if not material_id:
                continue
            entry_id = f"{material_id}_ms"
            entry = proc_map.get(entry_id)
            if entry is None:
                raise RuntimeError(
                    f"Reference entry was filtered by compatibility: {entry_id} ({cs_norm})"
                )
            ref_chemsys = _ensure_record_chemsys(rec)
            if not ref_chemsys:
                raise RuntimeError(
                    f"Missing chem_sys for reference record: {material_id} ({cs_norm})"
                )
            rows.append(
                {
                    "参照材料名": material_id,
                    "chemsys": _normalize_chemsys(ref_chemsys),
                    "e_hull_mp": float(pd_mp.get_e_above_hull(entry)),
                    "reduced_formula": _reduced_formula_from_record(rec),
                }
            )
        rows.sort(key=lambda row: (str(row["reduced_formula"]), str(row["参照材料名"])))
        mp_reference_rows_by_cs[cs_norm] = rows
        mp_formula_structs_by_cs[cs_norm] = formula_map
        _append_progress(
            "stage4",
            {
                "status": "chemsys_prepare_done",
                "p_dir": p_dir_label,
                "chemsys": cs_norm,
                "progress": f"{prep_idx}/{total_chemsys}",
                "mp_ref_count": len(rows),
                "mp_formula_groups": len(formula_map),
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )

    matcher = StructureMatcher(
        ltol=0.2,
        stol=0.3,
        angle_tol=5,
        primitive_cell=True,
        scale=True,
        attempt_supercell=True,
    )

    records_csv: List[dict] = []
    proposal_struct_meta: list[tuple[str, Optional[Structure], Optional[str]]] = []
    for poscar in poscars:
        proposal_record = proposal_relaxed_map[poscar]
        cs_norm = _normalize_chemsys(_chemsys_for_poscar(poscar))
        ref_formula_map = mp_formula_structs_by_cs.get(cs_norm, {})
        screen_key = f"{p_dir_label}/{poscar.name}"
        screen_rec = screening_map.get(screen_key, {})

        prop_path = proposal_record.get("file")
        prop_struct = _load_structure(prop_path)
        if prop_struct is None:
            # fallback: 提案元構造はほぼ必ず読める
            try:
                prop_struct = Structure.from_file(poscar)
            except Exception:
                prop_struct = None

        prop_failed = (
            proposal_record.get("status") in {"failed", "missing_stage2"}
            or proposal_record.get("E_final (eV)") == 999
            or str(proposal_record.get("file", "")).endswith("_failed.vasp")
        )

        e_hull_mp_raw = screen_rec.get("e_hull_mp")
        e_hull_mp = float(e_hull_mp_raw) if e_hull_mp_raw is not None else float("nan")

        if prop_struct is None or prop_failed:
            proposed_formula = float("nan")
            structural_novelty = float("nan")
            proposal_struct_meta.append((poscar.stem, None, None))
        else:
            proposed_formula = Composition(prop_struct.composition).reduced_formula
            same_formula_list = ref_formula_map.get(proposed_formula, [])
            if not same_formula_list:
                structural_novelty = True
            else:
                is_novel = True
                for s_ref in same_formula_list:
                    if matcher.fit(prop_struct, s_ref):
                        is_novel = False
                        break
                structural_novelty = is_novel
            proposal_struct_meta.append((poscar.stem, prop_struct, proposed_formula))

        records_csv.append(
            {
                "fname": poscar.stem,
                "chemsys": cs_norm,
                "e_hull_mp": e_hull_mp,
                "e_hull_mp_alex": float("nan"),
                "structural_novelty": structural_novelty,
                "num_dft_needed": float("nan"),
                "unique_structure": float("nan"),
                "reduced_formula": proposed_formula,
            }
        )

    # unique_structure: 同一フォルダ内提案で同一構造があれば False、それ以外は True
    fname_to_unique: dict[str, object] = {}
    by_formula: dict[str, list[tuple[str, Structure]]] = {}
    for fname, struct_obj, formula in proposal_struct_meta:
        if struct_obj is None or formula is None:
            fname_to_unique[fname] = float("nan")
            continue
        by_formula.setdefault(formula, []).append((fname, struct_obj))

    for _, items in by_formula.items():
        clusters: list[list[tuple[str, Structure]]] = []
        for fname, s_obj in items:
            placed = False
            for cluster in clusters:
                if matcher.fit(s_obj, cluster[0][1]):
                    cluster.append((fname, s_obj))
                    placed = True
                    break
            if not placed:
                clusters.append([(fname, s_obj)])
        for cluster in clusters:
            is_unique = len(cluster) == 1
            for fname, _ in cluster:
                fname_to_unique[fname] = is_unique

    for rec in records_csv:
        rec["unique_structure"] = fname_to_unique.get(rec["fname"], float("nan"))

    for poscar in poscars:
        cs_norm = _normalize_chemsys(_chemsys_for_poscar(poscar))
        rows = mp_reference_rows_by_cs.get(cs_norm, [])
        ref_df = pd.DataFrame(rows, columns=reference_csv_columns)
        ref_csv_path = dft_out_dir / f"{poscar.stem}_reference.csv"
        ref_df.to_csv(ref_csv_path, index=False)

    df = pd.DataFrame(records_csv)
    df.to_csv(result_dir / "relax_results.csv", index=False)
    print(f"[MP-only] relax_results.csv -> {result_dir / 'relax_results.csv'}")
    _append_progress(
        "stage4",
        {
            "status": "export_done",
            "p_dir": p_dir_label,
            "num_poscars": len(poscars),
            "num_rows": len(records_csv),
            "num_chemsys": total_chemsys,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )


def stage_repair_run_params(args) -> None:
    """Repair run_params in relax_results_*.json.gz records."""
    if args.repair_scope == "all":
        target_files = sorted(GLOBAL_RELAX_DIR.glob("relax_results_*.json.gz"))
    else:
        poscars = _list_poscars(Path(args.p_dir))
        if not poscars:
            raise RuntimeError(
                "No POSCAR files for repair_run_params with repair_scope=p_dir."
            )
        target_cs = sorted(
            {
                _normalize_chemsys(
                    chemsys_by_atomic_number(get_elements_from_poscar(p))
                )
                for p in poscars
            }
        )
        target_subs: set[str] = set()
        for cs in target_cs:
            target_subs.update(_enumerate_subsystems(cs))
        target_files = [
            GLOBAL_RELAX_DIR / f"relax_results_{cs}.json.gz"
            for cs in sorted(target_subs)
            if (GLOBAL_RELAX_DIR / f"relax_results_{cs}.json.gz").exists()
        ]

    if not target_files:
        print("[StageRepair] no target files found.")
        return

    total = len(target_files)
    changed_files = 0
    changed_records = 0
    processed_records = 0
    skipped_records = 0
    for idx, path in enumerate(target_files, start=1):
        stats = _repair_run_params_in_relax_results_file(
            path,
            allow_fallback=args.allow_run_params_fallback,
            verify_mp_compat=True,
            context=f"stage_repair {path.name}",
        )
        if stats["changed"] > 0:
            changed_files += 1
        changed_records += int(stats["changed"])
        processed_records += int(stats["processed"])
        skipped_records += int(stats["skipped"])
        _append_progress(
            "stage_repair",
            {
                "chemsys": path.name.replace("relax_results_", "").replace(
                    ".json.gz", ""
                ),
                "progress": f"{idx}/{total}",
                "updated_records": int(stats["changed"]),
                "processed_records": int(stats["processed"]),
                "skipped_records": int(stats["skipped"]),
                "mp_ok_count": int(stats["mp_ok_count"]),
                "proc_count": int(stats["proc_count"]),
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )

    print(
        "[StageRepair] completed: "
        f"files={total}, changed_files={changed_files}, "
        f"processed_records={processed_records}, changed_records={changed_records}, "
        f"skipped_records={skipped_records}"
    )


def _structural_novelty(proposal_relaxed: Path, refs: List[Path]) -> float:
    sm = StructureMatcher()
    if not refs:
        return 1.0
    s_prop = Structure.from_file(proposal_relaxed)
    for ref in refs:
        s_ref = Structure.from_file(ref)
        if sm.fit(s_prop, s_ref):
            return 0.0
    return 1.0


def stage_phase2_spec(args) -> None:
    cache_dir = Path(args.cache_dir or DEFAULT_CACHE_DIR)
    mp_cache_path = cache_dir / "mp_requests_cache.sqlite"
    alex_cache_path = cache_dir / "alex_requests_cache.sqlite"
    chemsys_to_fetch_path = cache_dir / "chemsys_to_fetch.json"
    _ = _read_json(chemsys_to_fetch_path) if chemsys_to_fetch_path.exists() else []
    alex_fetch_done_path = cache_dir / "alex_fetch_done.json"
    alex_fetch_done = (
        _load_fetch_done(alex_fetch_done_path) if alex_fetch_done_path.exists() else {}
    )
    screening_path = cache_dir / "screening_results.jsonl"
    if not screening_path.exists():
        raise RuntimeError(
            "screening_results.jsonl not found. Run Stage2 (MatterSim_phase1) first."
        )

    screening_records = _read_jsonl(screening_path)
    screening_map: Dict[str, dict] = {}
    for rec in screening_records:
        fname = rec.get("fname")
        if fname is None:
            continue
        screening_map[str(fname)] = rec
    proposal_relax_records_path = cache_dir / "proposal_relax_records.jsonl"
    if not proposal_relax_records_path.exists():
        raise RuntimeError(
            "proposal_relax_records.jsonl not found. "
            "Run Stage2 (MatterSim_phase1) first."
        )
    proposal_relax_map: Dict[str, dict] = {}
    for rec in _read_jsonl(proposal_relax_records_path):
        fname = rec.get("fname")
        if fname is None:
            continue
        proposal_relax_map[str(fname)] = rec

    poscars = _list_poscars(Path(args.p_dir))
    if not poscars:
        print("No POSCAR files for Phase2.")
        return
    prop_relaxed_dir = Path(args.r_dir)
    prop_relaxed_dir.mkdir(parents=True, exist_ok=True)
    result_dir = Path(args.rr_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    dft_out_dir = Path(args.s_dir)
    dft_out_dir.mkdir(parents=True, exist_ok=True)
    p_dir_label = Path(args.p_dir).name

    target_poscars: List[Path] = []
    for poscar in poscars:
        screen_key = f"{p_dir_label}/{poscar.name}"
        screen_rec = screening_map.get(screen_key)
        if screen_rec is None:
            raise RuntimeError(
                f"Missing screening result: {screen_key}. Run Stage2 (MatterSim_phase1) first."
            )
        e_hull_mp_raw = screen_rec.get("e_hull_mp")
        if e_hull_mp_raw is None:
            raise RuntimeError(f"Missing e_hull_mp in screening result: {screen_key}")
        if float(e_hull_mp_raw) < CREATE_DFT_NEEDED_THRES:
            target_poscars.append(poscar)

    if not target_poscars:
        pd.DataFrame(
            columns=[
                "fname",
                "chemsys",
                "e_hull_mp",
                "e_hull_mp_alex",
                "structural_novelty",
                "num_dft_needed",
                "unique_structure",
            ]
        ).to_csv(result_dir / "relax_results.csv", index=False)
        print(
            f"[Stage4] no proposals with e_hull_mp < {CREATE_DFT_NEEDED_THRES:.3f} in {p_dir_label}; wrote empty relax_results.csv"
        )
        return

    records_csv: List[dict] = []
    skipped_ehull_fnames: set[str] = set()
    known_ids = _load_known_material_ids(DEFAULT_RELAX_DB_PATH)
    proposal_progress_map = {
        p: f"{idx}/{len(target_poscars)}"
        for idx, p in enumerate(target_poscars, start=1)
    }

    # Load Stage2 proposal records (do not re-relax proposals in Stage4)
    proposal_relaxed_map = {}
    for poscar in target_poscars:
        screen_key = f"{p_dir_label}/{poscar.name}"
        rec = proposal_relax_map.get(screen_key)
        if rec is None:
            raise RuntimeError(
                f"Missing proposal record in proposal_relax_records.jsonl: {screen_key}"
            )
        rec_file = rec.get("file")
        if not rec_file:
            raise RuntimeError(
                f"Missing proposal relaxed file path in record: {screen_key}"
            )
        resolved_file = _resolve_relax_record_path(rec_file)
        if not resolved_file.exists():
            raise RuntimeError(
                f"Missing relaxed proposal file from Stage2: {resolved_file} ({screen_key})"
            )
        proposal_rec = {
            "material_id": rec.get("proposal_id", poscar.stem),
            "formula": rec.get("formula"),
            "E_initial (eV)": rec.get("E_initial (eV)"),
            "E_final (eV)": rec.get("E_final (eV)"),
            "ΔE (eV)": rec.get("ΔE (eV)"),
            "file": str(resolved_file),
            "chem_sys": rec.get("chemsys"),
            "run_type": rec.get("run_type", "GGA"),
            "is_hubbard": bool(rec.get("is_hubbard", False)),
            "hubbards": dict(rec.get("hubbards", {}) or {}),
            "status": rec.get("status", "relaxed"),
            "error": rec.get("error"),
            "skip_ehull_mp_reason": rec.get("skip_ehull_mp_reason"),
        }
        if proposal_rec.get("E_final (eV)") is None:
            raise RuntimeError(
                f"Missing E_final (eV) in proposal_relax_records.jsonl for {screen_key}"
            )
        if proposal_rec.get("chem_sys") is None:
            proposal_rec["chem_sys"] = chemsys_by_atomic_number(
                Structure.from_file(resolved_file).symbol_set
            )
        proposal_relaxed_map[poscar] = proposal_rec
        prop_status = "reuse_stage2"
        if proposal_rec.get("status") == "failed":
            prop_status = "failed"
        _log_stage4_relax_event(
            poscar.stem,
            proposal_progress_map[poscar],
            proposal_rec.get("file"),
            None,
            prop_status,
            proposal_rec.get("chem_sys"),
        )

    all_chemsys = sorted(
        {
            _normalize_chemsys(proposal_relaxed_map[p]["chem_sys"])
            for p in target_poscars
        }
    )

    hull_done = 0
    total_props = len(target_poscars)
    for cs in all_chemsys:
        cs_norm = _normalize_chemsys(cs)
        # gather relevant proposals
        target_props = [
            p
            for p in target_poscars
            if _normalize_chemsys(proposal_relaxed_map[p]["chem_sys"]) == cs_norm
        ]
        if not target_props:
            continue

        saved_path = GLOBAL_RELAX_DIR / f"relax_results_{cs_norm}.json.gz"
        has_agm = False
        if saved_path.exists():
            try:
                has_agm = any(
                    _record_has_agm(r) for r in _load_relax_results_file(saved_path)
                )
            except Exception as exc:
                _raise_if_corrupt_file_error(
                    exc, saved_path, f"stage4 check agm {cs_norm}"
                )
                has_agm = False
        if not has_agm and not alex_fetch_done_path.exists():
            raise RuntimeError(
                f"alex_fetch_done.json is missing but {cs_norm} requires Alexandria refs. Run Stage3 (prefetch_Alex) first."
            )

        # relax references (mp + alex)
        ref_context = target_props[0].stem
        t_start = time.monotonic()
        ref_records = _gather_relax_records_for_chemsys(
            cs_norm,
            mp_cache_path,
            alex_cache_path,
            args.steps_max,
            args.allow_run_params_fallback,
            known_ids,
            ref_context,
            proposal_progress_map[target_props[0]],
            p_dir=p_dir_label,
        )
        _log_stage4_postprocess_event(
            "gather_relax_records",
            time.monotonic() - t_start,
            chemsys=cs_norm,
            fname=ref_context,
        )
        t_start = time.monotonic()
        _write_relax_results_for_chemsys(ref_records, cs_norm)
        _log_stage4_postprocess_event(
            "write_relax_results_gz",
            time.monotonic() - t_start,
            chemsys=cs_norm,
            fname=ref_context,
        )
        mp_ref = _filter_records(ref_records, "mp")
        alex_ref = _filter_records(ref_records, "alex")
        if not mp_ref:
            raise RuntimeError(
                f"Missing MP relaxed records for {cs_norm}. Run Stage1/2 first."
            )
        # build DFT_POSCARS reference set (unique)  — 親＋下位系をまとめて reduced_formula グループでユニーク化
        dft_chemsys_dir = GLOBAL_RELAX_DIR / "DFT_POSCARS" / cs_norm
        dft_chemsys_dir.mkdir(parents=True, exist_ok=True)
        t_start = time.monotonic()

        filtered_csv = dft_chemsys_dir / "filtered_poscars.csv"
        ref_unique_paths: list[Path] | None = None

        # 既存 filtered_poscars を再利用するかチェック
        if filtered_csv.exists():
            df_existing = pd.read_csv(filtered_csv)
            sources = set(df_existing["source"].dropna().unique())
            if not {"mp", "alex"}.issubset(sources):
                raise RuntimeError(
                    f"filtered_poscars.csv for {cs_norm} exists but missing mp/alex refs: {sorted(sources)}"
                )
            paths = [Path(p) for p in df_existing["path"]]
            if all(p.exists() for p in paths):
                ref_unique_paths = paths
                _log_stage4_postprocess_event(
                    "reuse_filtered_poscars",
                    time.monotonic() - t_start,
                    chemsys=cs_norm,
                    fname=ref_context,
                )
            else:
                # パス欠如時は再計算にフォールバック
                pass

        if ref_unique_paths is None:
            # 参照レコードを構造付きで展開
            ref_struct_records: list[dict] = []
            for r in ref_records:
                p = Path(r.get("file", ""))
                if not p.exists():
                    continue
                try:
                    s_ref = Structure.from_file(str(p))
                except Exception:
                    continue
                ref_cs = _normalize_chemsys(chemsys_by_atomic_number(s_ref.symbol_set))
                ref_struct_records.append(
                    {
                        "path": p,
                        "mp_id": p.name.replace("_relaxed.vasp", "").replace(
                            ".vasp", ""
                        ),
                        "source": r.get("source")
                        or ("alex" if p.name.startswith("agm") else "mp"),
                        "chem_sys": ref_cs,
                        "structure": s_ref,
                        "reduced_formula": Composition(
                            s_ref.composition
                        ).reduced_formula,
                    }
                )

            # reduced_formula でグループ化し、グループ内で StructureMatcher でユニーク抽出
            groups: dict[str, list[dict]] = {}
            for rec in ref_struct_records:
                groups.setdefault(rec["reduced_formula"], []).append(rec)

            sm = StructureMatcher(
                ltol=0.2,
                stol=0.3,
                angle_tol=5,
                primitive_cell=True,
                scale=True,
                attempt_supercell=True,
            )

            ref_unique: list[dict] = []
            for _, recs in groups.items():
                # 代表決定: エネルギー情報があれば最小、無ければ先頭
                chosen: list[dict] = []
                for rec in recs:
                    placed = False
                    for idx, ref_rec in enumerate(chosen):
                        if sm.fit(rec["structure"], ref_rec["structure"]):
                            placed = True
                            # エネルギーが両方あれば小さい方を残す
                            e_new = (
                                rec["structure"].composition.energy
                                if hasattr(rec["structure"].composition, "energy")
                                else None
                            )
                            e_old = (
                                ref_rec["structure"].composition.energy
                                if hasattr(ref_rec["structure"].composition, "energy")
                                else None
                            )
                            if (
                                e_new is not None
                                and e_old is not None
                                and e_new < e_old
                            ):
                                chosen[idx] = rec
                            break
                    if not placed:
                        chosen.append(rec)
                ref_unique.extend(chosen)

            # filtered_poscars.csv を再生成（chem_sys と reduced_formula を明示的に含める）
            df_f = pd.DataFrame(
                [
                    {
                        "path": str(rec["path"]),
                        "mp_id": rec["mp_id"],
                        "source": rec["source"],
                        "chem_sys": rec["chem_sys"],
                        "reduced_formula": rec["reduced_formula"],
                    }
                    for rec in ref_unique
                ]
            )
            df_f.to_csv(filtered_csv, index=False)

            # 以降で使う参照パス（ユニーク代表）のリスト
            ref_unique_paths = [Path(rec["path"]) for rec in ref_unique]

            # 親 DFT_POSCARS ディレクトリには参照のみを配置（提案は含めない）
            for p in ref_unique_paths:
                dest = dft_chemsys_dir / p.name
                if not dest.exists():
                    shutil.copy(p, dest)

        _log_stage4_postprocess_event(
            "build_dft_poscars",
            time.monotonic() - t_start,
            chemsys=cs_norm,
            fname=ref_context,
        )

        for poscar in target_props:
            proposal_record = proposal_relaxed_map[poscar]
            if (
                proposal_record.get("skip_ehull_mp_reason")
                == SKIP_EHULL_REASON_TOO_CLOSE
            ):
                records_csv.append(
                    {
                        "fname": poscar.stem,
                        "chemsys": cs_norm,
                        "e_hull_mp": float("nan"),
                        "e_hull_mp_alex": float("nan"),
                        "structural_novelty": float("nan"),
                        "num_dft_needed": float("nan"),
                        "unique_structure": float("nan"),
                    }
                )
                skipped_ehull_fnames.add(poscar.stem)
                hull_done += 1
                _append_progress(
                    "stage4",
                    {
                        "fname": poscar.stem,
                        "progress": f"{hull_done}/{total_props}",
                        "ref_progress": f"{len(ref_records)}/{_count_ref_candidates(cs_norm, mp_cache_path, alex_cache_path)}",
                        "status": "skip_ehull_nan_too_close",
                        "p_dir": p_dir_label,
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    },
                )
                continue
            screen_key = f"{p_dir_label}/{poscar.name}"
            screen_rec = screening_map.get(screen_key)
            if screen_rec is None or screen_rec.get("e_hull_mp") is None:
                raise RuntimeError(
                    f"Missing e_hull_mp in screening result: {screen_key}"
                )
            e_hull_mp_from_stage2 = float(screen_rec["e_hull_mp"])
            ref_total = _count_ref_candidates(cs_norm, mp_cache_path, alex_cache_path)
            ref_done = len(ref_records)
            t_start = time.monotonic()
            e_hull_mp, e_hull_mp_alex = _compute_ehulls_phase2(
                mp_ref,
                alex_ref,
                proposal_record,
                e_hull_mp_from_stage2=e_hull_mp_from_stage2,
            )
            _log_stage4_postprocess_event(
                "compute_ehulls",
                time.monotonic() - t_start,
                chemsys=cs_norm,
                fname=poscar.stem,
            )
            e_hull_decision = (
                e_hull_mp_alex if not np.isnan(e_hull_mp_alex) else e_hull_mp
            )

            # structural novelty
            prop_relaxed_path = Path(proposal_record.get("file"))
            t_start = time.monotonic()
            s_novel = (
                _structural_novelty(prop_relaxed_path, ref_unique_paths)
                if ref_unique_paths
                else 1.0
            )
            _log_stage4_postprocess_event(
                "structural_novelty",
                time.monotonic() - t_start,
                chemsys=cs_norm,
                fname=poscar.stem,
            )
            num_dft_needed = len(ref_unique_paths) + 2
            # DFT output for selected proposals
            if e_hull_decision < CREATE_DFT_NEEDED_THRES:
                t_start = time.monotonic()
                save_dir = dft_out_dir / poscar.stem
                shutil.rmtree(save_dir, ignore_errors=True)
                save_dir.mkdir(parents=True, exist_ok=True)
                # 参照 + 提案構造を proposal ごとの DFT_OUT ディレクトリに配置
                out_refs = []
                for p in ref_unique_paths:
                    dest = save_dir / p.name
                    shutil.copy(p, dest)
                    out_refs.append(
                        {
                            "path": str(dest),
                            "mp_id": p.name.replace("_relaxed.vasp", "").replace(
                                ".vasp", ""
                            ),
                            "source": "alex" if p.name.startswith("agm") else "mp",
                            "chem_sys": _normalize_chemsys(cs_norm),
                            "reduced_formula": Composition(
                                Structure.from_file(dest).composition
                            ).reduced_formula,
                        }
                    )
                prop_dest = save_dir / prop_relaxed_path.name
                shutil.copy(prop_relaxed_path, prop_dest)
                out_summary = out_refs + [
                    {
                        "path": str(prop_dest),
                        "mp_id": poscar.stem,
                        "source": "proposal",
                        "chem_sys": _normalize_chemsys(cs_norm),
                        "reduced_formula": Composition(
                            Structure.from_file(prop_dest).composition
                        ).reduced_formula,
                    }
                ]
                pd.DataFrame(out_summary).to_csv(
                    save_dir / "material_summary.csv", index=False
                )
                shutil.copy(poscar, save_dir / poscar.name)
                _log_stage4_postprocess_event(
                    "write_dft_out",
                    time.monotonic() - t_start,
                    chemsys=cs_norm,
                    fname=poscar.stem,
                )

            records_csv.append(
                {
                    "fname": poscar.stem,
                    "chemsys": cs_norm,
                    "e_hull_mp": e_hull_mp,
                    "e_hull_mp_alex": e_hull_mp_alex,
                    "structural_novelty": (
                        s_novel
                        if e_hull_decision < CREATE_DFT_NEEDED_THRES
                        else float("nan")
                    ),
                    "num_dft_needed": (
                        num_dft_needed
                        if e_hull_decision < CREATE_DFT_NEEDED_THRES
                        else float("nan")
                    ),
                    "unique_structure": False,  # filled later
                }
            )
            hull_done += 1
            _append_progress(
                "stage4",
                {
                    "fname": poscar.stem,
                    "progress": f"{hull_done}/{total_props}",
                    "ref_progress": f"{ref_done}/{ref_total}",
                    "status": "relaxed_refs" if ref_done else "skip_relax_results_gz",
                    "p_dir": p_dir_label,
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
            )

        # 下位系の DFT_POSCARS も生成（既存があれば流用）
        parent_df = (
            pd.read_csv(filtered_csv) if filtered_csv.exists() else pd.DataFrame()
        )
        for sub in _enumerate_subsystems(cs_norm):
            if sub == cs_norm:
                continue
            sub_dir = GLOBAL_RELAX_DIR / "DFT_POSCARS" / sub
            if sub_dir.exists() and any(sub_dir.glob("*.vasp")):
                continue  # スキップ条件
            if parent_df.empty:
                continue
            # chem_sys 列を使用してサブ系を抽出
            rows = [
                row
                for _, row in parent_df.iterrows()
                if set(_elements_from_chemsys(row["chem_sys"])).issubset(
                    set(_elements_from_chemsys(sub))
                )
            ]
            if not rows:
                continue
            sub_dir.mkdir(parents=True, exist_ok=True)
            sub_filtered = sub_dir / "filtered_poscars.csv"
            df_sub = pd.DataFrame(rows)
            for p_str in df_sub.path.values:
                p = Path(p_str)
                if p.exists():
                    shutil.copy(p, sub_dir / p.name)
            df_sub.to_csv(sub_filtered, index=False)

    # unique_structure across proposals (StructureMatcher on relaxed proposals)
    t_start = time.monotonic()
    matcher = StructureMatcher(
        ltol=0.2,
        stol=0.3,
        angle_tol=5,
        primitive_cell=True,
        scale=True,
        attempt_supercell=True,
    )
    relaxed_structs = [
        (
            Path(proposal_relaxed_map[p]["file"]),
            proposal_relaxed_map[p]["E_final (eV)"] or 0.0,
        )
        for p in target_poscars
    ]
    groups: List[List[int]] = []
    for i, (path_i, _) in enumerate(relaxed_structs):
        placed = False
        for grp in groups:
            ref_idx = grp[0]
            if matcher.fit(
                Structure.from_file(relaxed_structs[ref_idx][0]),
                Structure.from_file(path_i),
            ):
                grp.append(i)
                placed = True
                break
        if not placed:
            groups.append([i])
    unique_flags = [False] * len(relaxed_structs)
    for grp in groups:
        best = min(grp, key=lambda idx: relaxed_structs[idx][1])
        unique_flags[best] = True
    fname_to_unique = {
        target_poscars[i].stem: unique_flags[i] for i in range(len(target_poscars))
    }
    for rec in records_csv:
        if rec["fname"] in skipped_ehull_fnames:
            continue
        rec["unique_structure"] = fname_to_unique.get(rec["fname"], False)
    _log_stage4_postprocess_event(
        "unique_structure",
        time.monotonic() - t_start,
    )

    df = pd.DataFrame(records_csv)
    t_start = time.monotonic()
    df.to_csv(result_dir / "relax_results.csv", index=False)
    _log_stage4_postprocess_event(
        "write_relax_results_csv",
        time.monotonic() - t_start,
    )
    print(f"[Stage4] relax_results.csv -> {result_dir / 'relax_results.csv'}")


def main() -> None:
    global GLOBAL_RELAX_DIR, DEFAULT_RELAX_DB_PATH

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p_dir", "--p_dir", required=True, help="Proposal POSCAR directory"
    )
    parser.add_argument(
        "-s_dir",
        "--s_dir",
        default="./DFT_poscars",
        help="DFT output directory per proposal",
    )
    parser.add_argument(
        "-r_dir",
        "--r_dir",
        default="./prop_relaxed_poscars",
        help="Proposal relaxed directory",
    )
    parser.add_argument(
        "-rr_dir",
        "--rr_dir",
        default="./result_dir",
        help="Result directory for relax_results.csv",
    )
    parser.add_argument(
        "--cache_dir",
        default=str(DEFAULT_CACHE_DIR),
        help="Cache directory (default: ./cache)",
    )
    parser.add_argument(
        "--global_relax_dir",
        default=str(GLOBAL_RELAX_DIR),
        help="Override GLOBAL_RELAX_DIR (default: data/mp_mattersim_relaxed)",
    )
    parser.add_argument(
        "--mp_api_key",
        default=None,
        help="MP API key (required for stages that run prefetch_MP)",
    )
    parser.add_argument(
        "--alexandria_base_url",
        default="https://alexandria.icams.rub.de/pbe",
        help="Alexandria base URL",
    )
    parser.add_argument("--steps_max", type=int, default=400)
    parser.add_argument(
        "--allow_run_params_fallback",
        action="store_true",
        help="Allow fallback to existing run_params when recalculation fails.",
    )
    parser.add_argument(
        "--phase_diagram_timeout_sec",
        type=float,
        default=None,
        help=(
            "Optional timeout for PhaseDiagram/get_e_above_hull in stage2. "
            "Timeouts are recorded as e_hull_mp=NaN and skipped."
        ),
    )
    parser.add_argument(
        "--repair_scope",
        choices=["p_dir", "all"],
        default="p_dir",
        help="Target scope for repair_run_params stage.",
    )
    parser.add_argument(
        "-stage",
        "--stage",
        choices=[
            "prefetch_MP",
            "MatterSim_phase1",
            "prefetch_Alex",
            "MatterSim_phase2",
            "all",
            "all_MP_only",
            "export_MP_only",
            "repair_run_params",
        ],
        default="all",
    )
    args = parser.parse_args()

    stages_requiring_mp_api_key = {"prefetch_MP", "all", "all_MP_only"}
    if args.stage in stages_requiring_mp_api_key and not args.mp_api_key:
        parser.error(
            "--mp_api_key is required for stages: prefetch_MP, all, all_MP_only"
        )

    # Override GLOBAL_RELAX_DIR / DEFAULT_RELAX_DB_PATH if requested
    GLOBAL_RELAX_DIR = Path(args.global_relax_dir).expanduser().resolve()
    GLOBAL_RELAX_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_RELAX_DB_PATH = GLOBAL_RELAX_DIR / "relax_results.json"

    if args.stage == "prefetch_MP":
        stage_prefetch_mp_spec(args)
    elif args.stage == "MatterSim_phase1":
        stage_phase1_spec(args)
    elif args.stage == "prefetch_Alex":
        stage_prefetch_alex_spec(args)
    elif args.stage == "MatterSim_phase2":
        stage_phase2_spec(args)
    elif args.stage == "all":
        stage_prefetch_mp_spec(args)
        stage_phase1_spec(args)
        stage_prefetch_alex_spec(args)
        stage_phase2_spec(args)
    elif args.stage == "all_MP_only":
        stage_prefetch_mp_spec(args)
        stage_phase1_spec(args)
        stage_export_mp_only(args)
    elif args.stage == "export_MP_only":
        stage_export_mp_only(args)
    elif args.stage == "repair_run_params":
        stage_repair_run_params(args)


if __name__ == "__main__":
    main()
