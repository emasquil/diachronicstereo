#!/usr/bin/env python3
"""
Build per-TIFF JSON metadata for DFC2019 Track-3 imagery (TIFF-first strategy).

Field priority: RPC from embedded TIFF tags, else matching RPB; dimensions from
TIFF, else IMD numRows/numColumns; acquisition from NITF_IDATIM, else IMD
firstLineTime; footprint from NITF_IGEOLO, else IMD BAND_* corner coords (with
HAE min/max if present); sun angles only from IMD when available.

Usage examples:
  python process_dfc_metadata.py --tif Track3-RGB-1/JAX_070_013_RGB.tif \
      --metadata-root Track3-Metadata --write-near-tif
  python process_dfc_metadata.py --images-dir Track3-RGB-1 \
      --metadata-root Track3-Metadata --out-dir out_json
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import rasterio  # pip install rasterio

# ---------- TIFF helpers (RPC, size, tags) ----------


def _read_rpc_from_tif(tif_path: Path) -> Optional[Dict[str, Any]]:
    with rasterio.open(tif_path) as ds:
        tags = ds.tags(ns="RPC")
        if not tags:
            return None

        def _f(k: str) -> Optional[float]:
            v = tags.get(k)
            return float(v) if v is not None else None

        def _l(k: str) -> Optional[List[float]]:
            v = tags.get(k)
            if v is None:
                return None
            arr = [float(x) for x in v.split()]
            if len(arr) != 20:
                raise ValueError(
                    f"{k} has {len(arr)} coeffs (expected 20) in {tif_path.name}"
                )
            return arr

        rpc = {
            "row_offset": _f("LINE_OFF"),
            "col_offset": _f("SAMP_OFF"),
            "lat_offset": _f("LAT_OFF"),
            "lon_offset": _f("LONG_OFF"),
            "alt_offset": _f("HEIGHT_OFF"),
            "row_scale": _f("LINE_SCALE"),
            "col_scale": _f("SAMP_SCALE"),
            "lat_scale": _f("LAT_SCALE"),
            "lon_scale": _f("LONG_SCALE"),
            "alt_scale": _f("HEIGHT_SCALE"),
            "row_num": _l("LINE_NUM_COEFF"),
            "row_den": _l("LINE_DEN_COEFF"),
            "col_num": _l("SAMP_NUM_COEFF"),
            "col_den": _l("SAMP_DEN_COEFF"),
        }
        if any(v is None for v in rpc.values()):
            return None
        return rpc


def _tif_size_tags(tif_path: Path) -> Tuple[int, int, Dict[str, str]]:
    with rasterio.open(tif_path) as ds:
        return ds.height, ds.width, ds.tags()


# ---------- Minimal RPB/IMD parsers ----------

_KV = re.compile(r"^\s*([A-Za-z0-9_]+)\s*=\s*(.+?)\s*;?\s*$")
_LIST_BEG = re.compile(r"^\s*([A-Za-z0-9_]+)\s*=\s*\(\s*$")
_LIST_END = re.compile(r"^\s*\)\s*;?\s*$")
_GBEG = re.compile(r"^\s*BEGIN_GROUP\s*=\s*([A-Za-z0-9_]+)\s*$")
_GEND = re.compile(r"^\s*END_GROUP\s*=\s*([A-Za-z0-9_]+)\s*$")


def _tokenize_kv(text: str):
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        i += 1
        if not line:
            continue
        m = _GBEG.match(line)
        m2 = _GEND.match(line)
        if m:
            yield ("gbeg", m.group(1))
            continue
        if m2:
            yield ("gend", m2.group(1))
            continue
        m = _LIST_BEG.match(line)
        if m:
            key = m.group(1)
            buf = []
            while i < len(lines):
                ln = lines[i].rstrip()
                i += 1
                if _LIST_END.match(ln):
                    break
                buf.append(ln)
            yield ("list", key, "(" + "\n".join(buf) + ")")
            continue
        m = _KV.match(line)
        if m:
            yield ("kv", m.group(1), m.group(2))
            continue


def _parse_number_list(text: str) -> List[float]:
    s = text.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    toks = [t.strip().rstrip(";") for t in s.replace("\n", " ").split(",") if t.strip()]
    return [float(t.strip('"')) for t in toks]


def _read_rpb(rpb_path: Path) -> Optional[Dict[str, Any]]:
    if not rpb_path or not rpb_path.exists():
        return None
    kv: Dict[str, Any] = {}
    for tok in _tokenize_kv(rpb_path.read_text()):
        if tok[0] in ("kv", "list"):
            kv[tok[1]] = tok[2]

    def _f(k):
        v = kv.get(k)
        return float(str(v).strip().rstrip(";").strip('"')) if v is not None else None

    def _l(k):
        v = kv.get(k)
        if v is None:
            return None
        arr = _parse_number_list(v)
        if len(arr) != 20:
            raise ValueError(
                f"{k} has {len(arr)} coeffs (expected 20) in {rpb_path.name}"
            )
        return arr

    rpc = {
        "row_offset": _f("lineOffset"),
        "col_offset": _f("sampOffset"),
        "lat_offset": _f("latOffset"),
        "lon_offset": _f("longOffset"),
        "alt_offset": _f("heightOffset"),
        "row_scale": _f("lineScale"),
        "col_scale": _f("sampScale"),
        "lat_scale": _f("latScale"),
        "lon_scale": _f("longScale"),
        "alt_scale": _f("heightScale"),
        "row_num": _l("lineNumCoef"),
        "row_den": _l("lineDenCoef"),
        "col_num": _l("sampNumCoef"),
        "col_den": _l("sampDenCoef"),
    }
    return rpc if all(v is not None for v in rpc.values()) else None


def _read_imd(imd_path: Path) -> Dict[str, Any]:
    if not imd_path or not imd_path.exists():
        return {}
    info: Dict[str, Any] = {"band_blocks": {}}
    stack: List[str] = []
    cur = None
    blk: Dict[str, str] = {}
    for tok in _tokenize_kv(imd_path.read_text()):
        if tok[0] == "gbeg":
            cur = tok[1]
            stack.append(cur)
            blk = {}
        elif tok[0] == "gend":
            if stack and stack[-1] == tok[1]:
                stack.pop()
            if cur == tok[1] and cur and cur.startswith("BAND_"):
                info["band_blocks"][cur] = blk.copy()
            cur = None
            blk = {}
        elif tok[0] == "kv":
            k, v = tok[1], tok[2].strip().rstrip(";").strip('"')
            if not stack:
                if k in (
                    "numRows",
                    "numColumns",
                    "meanSunEl",
                    "meanSunAz",
                    "firstLineTime",
                ):
                    info[k] = v
            else:
                blk[k] = v
    # coerce
    if "numRows" in info:
        info["numRows"] = int(float(info["numRows"]))
    if "numColumns" in info:
        info["numColumns"] = int(float(info["numColumns"]))
    if "meanSunEl" in info:
        info["meanSunEl"] = float(info["meanSunEl"])
    if "meanSunAz" in info:
        info["meanSunAz"] = float(info["meanSunAz"])
    return info


# ---------- NITF helpers ----------

_DMS = re.compile(r"(\d{2})(\d{2})(\d{2})([NS])(\d{3})(\d{2})(\d{2})([EW])")


def _dms_to_deg(d, m, s, hem):
    val = d + m / 60 + s / 3600
    return -val if hem in ("S", "W") else val


def _polygon_from_igeolo(igeolo: str) -> Optional[Dict[str, Any]]:
    pts: List[List[float]] = []
    pos = 0
    for _ in range(4):
        m = _DMS.search(igeolo, pos)
        if not m:
            return None
        lat = _dms_to_deg(int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4))
        lon = _dms_to_deg(int(m.group(5)), int(m.group(6)), int(m.group(7)), m.group(8))
        pts.append([lon, lat])
        pos = m.end()
    lon_c = sum(p[0] for p in pts) / 4.0
    lat_c = sum(p[1] for p in pts) / 4.0
    return {"type": "Polygon", "coordinates": [pts], "center": [lon_c, lat_c]}


def _imd_time_to_compact(t: str) -> str:
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})", t)
    return "".join(m.groups()) if m else t


# ---------- metadata file lookup ----------


def _find_metadata_files(
    metadata_root: Path, city: str, index: int
) -> Tuple[Optional[Path], Optional[Path]]:
    city_dir = metadata_root / city
    if not city_dir.exists():
        return None, None
    for pad in (0, 2, 3):
        s = f"{index:0{pad}d}" if pad else str(index)
        imd = city_dir / f"{s}.IMD"
        rpb = city_dir / f"{s}.RPB"
        if imd.exists() or rpb.exists():
            return (imd if imd.exists() else None), (rpb if rpb.exists() else None)
    return None, None


# ---------- core assembly (TIFF first) ----------


def build_json_tiff_first(
    tif_path: Path, metadata_root: Optional[Path]
) -> Dict[str, Any]:
    # Always start from the TIFF
    height, width, tags = _tif_size_tags(tif_path)
    out: Dict[str, Any] = {
        "img": tif_path.name,
        "height": int(height),
        "width": int(width),
    }

    # RPC: TIFF -> RPB
    rpc = _read_rpc_from_tif(tif_path)

    # Parse filename to locate metadata
    m = re.match(r"^([A-Za-z]+)_[^_]*_(\d+)_.*\.tif$", tif_path.name)
    city = m.group(1) if m else None
    idx = int(m.group(2)) if m else None
    imd_info: Dict[str, Any] = {}
    imd_path = rpb_path = None
    if metadata_root and city and idx is not None:
        imd_path, rpb_path = _find_metadata_files(metadata_root, city, idx)
        if imd_path:
            imd_info = _read_imd(imd_path)
        if rpc is None and rpb_path:
            rpc = _read_rpb(rpb_path)

    if rpc is None:
        raise RuntimeError(
            f"No RPC available for {tif_path.name} (no embedded RPC and no RPB)."
        )
    out["rpc"] = rpc

    # acquisition_date: TIFF first, then IMD
    if "NITF_IDATIM" in tags:
        out["acquisition_date"] = tags["NITF_IDATIM"].strip()
    elif "firstLineTime" in imd_info:
        out["acquisition_date"] = _imd_time_to_compact(imd_info["firstLineTime"])

    # geojson: TIFF first (NITF_IGEOLO), then IMD corners
    if "NITF_IGEOLO" in tags:
        poly = _polygon_from_igeolo(tags["NITF_IGEOLO"])
        if poly:
            out["geojson"] = {
                "type": "Polygon",
                "coordinates": [poly["coordinates"]],
                "center": poly["center"],
            }
    if "geojson" not in out and imd_info.get("band_blocks"):
        blk = imd_info["band_blocks"].get("BAND_P") or next(
            iter(imd_info["band_blocks"].values())
        )
        need = {"ULLon", "ULLat", "URLon", "URLat", "LRLon", "LRLat", "LLLon", "LLLat"}
        if need.issubset(blk.keys()):
            pts = [
                [float(blk["ULLon"]), float(blk["ULLat"])],
                [float(blk["URLon"]), float(blk["URLat"])],
                [float(blk["LRLon"]), float(blk["LRLat"])],
                [float(blk["LLLon"]), float(blk["LLLat"])],
            ]
            lon_c = sum(p[0] for p in pts) / 4.0
            lat_c = sum(p[1] for p in pts) / 4.0
            out["geojson"] = {
                "type": "Polygon",
                "coordinates": [pts],
                "center": [lon_c, lat_c],
            }
            # Optional alt range
            haes = [blk.get(k) for k in ("ULHAE", "URHAE", "LRHAE", "LLHAE")]
            haes = [float(h) for h in haes if h is not None]
            if haes:
                out["min_alt"] = min(haes)
                out["max_alt"] = max(haes)

    # sun angles: IMD only
    if "meanSunEl" in imd_info:
        out["sun_elevation"] = f"{imd_info['meanSunEl']:+g}"
    if "meanSunAz" in imd_info:
        out["sun_azimuth"] = f"{imd_info['meanSunAz']:g}"

    # height/width already from TIFF; if missing (shouldn't happen), try IMD
    if "height" not in out and "numRows" in imd_info:
        out["height"] = int(imd_info["numRows"])
    if "width" not in out and "numColumns" in imd_info:
        out["width"] = int(imd_info["numColumns"])

    return out


# ---------- CLI / batch ----------


def main():
    ap = argparse.ArgumentParser(
        description="Build DFC2019 Track-3 JSONs (TIFF-first; fallback to RPB/IMD)."
    )
    ap.add_argument("--tif", type=Path, help="Single chip .tif")
    ap.add_argument(
        "--images-dir", type=Path, help="Directory containing chip .tif files"
    )
    ap.add_argument(
        "--metadata-root",
        type=Path,
        required=True,
        help="Track3-Metadata root (contains JAX/, etc.)",
    )
    ap.add_argument("--out", type=Path, help="Output JSON for single file")
    ap.add_argument("--out-dir", type=Path, help="Directory to write batch JSONs")
    ap.add_argument(
        "--write-near-tif",
        action="store_true",
        help="Write JSON next to each TIFF (single or batch)",
    )
    args = ap.parse_args()

    if bool(args.tif) == bool(args.images_dir):
        raise SystemExit("Provide either --tif or --images-dir (exclusively).")

    if args.tif:
        js = build_json_tiff_first(args.tif, args.metadata_root)
        out_path = args.out or (
            args.tif.with_suffix(".json") if args.write_near_tif else None
        )
        if out_path is None:
            raise SystemExit("For single --tif, provide --out or --write-near-tif.")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(js, indent=2))
        print(f"Wrote {out_path}")
        return

    # batch
    tifs = sorted(
        p for p in args.images_dir.iterdir() if p.suffix.lower() in (".tif", ".tiff")
    )
    if not tifs:
        raise SystemExit(f"No TIFFs found in {args.images_dir}")

    for tif_path in tifs:
        try:
            js = build_json_tiff_first(tif_path, args.metadata_root)
        except Exception as e:
            print(f"[WARN] Skipping {tif_path.name}: {e}")
            continue
        if args.write_near_tif:
            out_path = tif_path.with_suffix(".json")
        else:
            out_dir = args.out_dir or (args.images_dir / "json")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / (tif_path.stem + ".json")
        out_path.write_text(json.dumps(js, indent=2))
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
