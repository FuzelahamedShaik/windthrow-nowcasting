# windthrow_nowcasting/agents/cap_ingest.py

"""
Usage examples:
  # 1) Last 7 days
  python agents/data_ingestion/cap_ingest.py --feed en --days 7

  # Specific timeframe
  python agents/data_ingestion/cap_ingest.py --feed en --start "2025-12-25T00:00:00Z" --end "2025-12-29T18:00:00Z"

  # Timeframe + bbox filter
  python agents/data_ingestion/cap_ingest.py --feed en --days 7 --bbox 24,59.5,26.5,60.7
"""

import os
import time
import json
import argparse
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from urllib.parse import urljoin
from datetime import datetime, timezone, timedelta

# Optional (bbox filtering using CAP polygons)
try:
    from shapely.geometry import Polygon, box
    HAS_SHAPELY = True
except Exception:
    HAS_SHAPELY = False


CAP_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "cap": "urn:oasis:names:tc:emergency:cap:1.2",
}

FEEDS = {
    "fi": "https://alerts.fmi.fi/cap/feed/atom_fi-FI.xml",
    "sv": "https://alerts.fmi.fi/cap/feed/atom_sv-FI.xml",
    "en": "https://alerts.fmi.fi/cap/feed/atom_en-GB.xml",
}

DEFAULT_BASE = "https://alerts.fmi.fi/"


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _get(url: str, timeout=30) -> bytes:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "windthrow_nowcasting/0.1"})
    r.raise_for_status()
    return r.content


def _load_state(state_path: str):
    if os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"seen_xml_urls": []}


def _save_state(state_path: str, state):
    ensure_dir(os.path.dirname(state_path))
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def parse_iso_utc(s: str) -> datetime:
    """
    Accepts:
      - 2025-12-29T00:00:00Z
      - 2025-12-29T00:00:00+00:00
      - 2025-12-29 00:00:00 (assumed UTC)
    Returns timezone-aware UTC datetime.
    """
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        # try space-separated
        dt = datetime.fromisoformat(s.replace(" ", "T"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_bbox(s: str):
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be 'minLon,minLat,maxLon,maxLat'")
    minlon, minlat, maxlon, maxlat = map(float, parts)
    return minlon, minlat, maxlon, maxlat


def extract_cap_xml_urls_from_atom(atom_bytes: bytes, base_url: str = DEFAULT_BASE) -> list[str]:
    root = ET.fromstring(atom_bytes)
    urls = []

    for entry in root.findall("atom:entry", CAP_NS):
        for link in entry.findall("atom:link", CAP_NS):
            href = link.attrib.get("href")
            if not href:
                continue
            full = urljoin(base_url, href)
            if "/cap/" in full and full.endswith(".xml"):
                urls.append(full)

    # de-dup preserve order
    out = []
    seen = set()
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _cap_text(root: ET.Element, path: str):
    el = root.find(path, CAP_NS)
    return el.text.strip() if el is not None and el.text else None


def _info_text(info: ET.Element, tag: str):
    el = info.find(f"cap:{tag}", CAP_NS)
    return el.text.strip() if el is not None and el.text else None


def _collect_polygons_from_cap(root: ET.Element) -> list[str]:
    polys = []
    for poly_el in root.findall(".//cap:info/cap:area/cap:polygon", CAP_NS):
        if poly_el is not None and poly_el.text:
            polys.append(poly_el.text.strip())
    return polys


def _polygon_string_to_shapely(poly_str: str):
    """
    CAP polygon format: "lat,lon lat,lon lat,lon ..."
    Returns shapely Polygon in lon/lat order.
    """
    # Example: "60.1,24.9 60.2,25.0 60.1,25.1 60.1,24.9"
    pts = []
    for pair in poly_str.split():
        lat_s, lon_s = pair.split(",")
        lat = float(lat_s)
        lon = float(lon_s)
        pts.append((lon, lat))  # shapely expects (x=lon, y=lat)
    if len(pts) < 3:
        return None
    return Polygon(pts)


def parse_cap_xml(xml_bytes: bytes) -> dict:
    root = ET.fromstring(xml_bytes)

    identifier = _cap_text(root, "cap:identifier")
    sent = _cap_text(root, "cap:sent")
    sender = _cap_text(root, "cap:sender")
    status = _cap_text(root, "cap:status")
    msg_type = _cap_text(root, "cap:msgType")
    scope = _cap_text(root, "cap:scope")

    infos = root.findall("cap:info", CAP_NS)

    # first info block for key structured fields
    language = event = severity = urgency = certainty = onset = expires = effective = None
    area_desc = None

    if infos:
        info0 = infos[0]
        language = _info_text(info0, "language")
        event = _info_text(info0, "event")
        severity = _info_text(info0, "severity")
        urgency = _info_text(info0, "urgency")
        certainty = _info_text(info0, "certainty")
        onset = _info_text(info0, "onset")
        expires = _info_text(info0, "expires")
        effective = _info_text(info0, "effective")

        area0 = info0.find("cap:area", CAP_NS)
        if area0 is not None:
            ad = area0.find("cap:areaDesc", CAP_NS)
            area_desc = ad.text.strip() if ad is not None and ad.text else None

    # concatenate texts across all infos (multi-language safe)
    headline_all, desc_all, instr_all = [], [], []
    for info in infos:
        h = _info_text(info, "headline")
        d = _info_text(info, "description")
        ins = _info_text(info, "instruction")
        if h: headline_all.append(h)
        if d: desc_all.append(d)
        if ins: instr_all.append(ins)

    polygons = _collect_polygons_from_cap(root)

    return {
        "identifier": identifier,
        "sent": sent,
        "sender": sender,
        "status": status,
        "msg_type": msg_type,
        "scope": scope,
        "language_first": language,
        "event_first": event,
        "severity_first": severity,
        "urgency_first": urgency,
        "certainty_first": certainty,
        "effective_first": effective,
        "onset_first": onset,
        "expires_first": expires,
        "headline_all": " | ".join(headline_all) if headline_all else None,
        "description_all": " | ".join(desc_all) if desc_all else None,
        "instruction_all": " | ".join(instr_all) if instr_all else None,
        "area_desc_first": area_desc,
        "polygons": polygons,  # list[str]
    }


def _time_in_window(sent_iso: str | None, start: datetime | None, end: datetime | None) -> bool:
    if start is None and end is None:
        return True
    if not sent_iso:
        return True  # if missing, keep (or you can drop; but keep is safer)

    try:
        sent_dt = parse_iso_utc(sent_iso)
    except Exception:
        return True

    if start and sent_dt < start:
        return False
    if end and sent_dt > end:
        return False
    return True


def _bbox_match(polygons: list[str], bbox_tuple) -> bool | None:
    """
    Returns:
      True  -> intersects bbox
      False -> does not intersect bbox
      None  -> cannot evaluate (no shapely or no polygons)
    """
    if bbox_tuple is None:
        return None
    if not HAS_SHAPELY:
        return None
    if not polygons:
        return None

    minlon, minlat, maxlon, maxlat = bbox_tuple
    bb = box(minlon, minlat, maxlon, maxlat)

    # If any polygon intersects, treat as match
    for pstr in polygons:
        try:
            poly = _polygon_string_to_shapely(pstr)
            if poly is not None and poly.intersects(bb):
                return True
        except Exception:
            continue

    return False

def _parse_dt(s: str | None):
    if not s:
        return None
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def _overlaps(start, end, onset, expires):
    # If no filtering requested
    if start is None and end is None:
        return True
    # If CAP doesn't have hazard times, keep it (or drop if you prefer strict)
    if onset is None and expires is None:
        return True

    # Use sent as fallback if onset/expires missing
    o = onset or expires
    x = expires or onset
    if o is None or x is None:
        return True

    if start and x < start:
        return False
    if end and o > end:
        return False
    return True

def run(
    feed_key: str,
    max_new: int,
    raw_dir: str,
    out_dir: str,
    state_path: str,
    start: datetime | None,
    end: datetime | None,
    bbox_tuple,
    timeout: int = 30,
    *args
) -> pd.DataFrame:
    ensure_dir(raw_dir)
    ensure_dir(out_dir)

    state = _load_state(state_path)
    seen = set(state.get("seen_xml_urls", []))

    feed_url = FEEDS[feed_key]
    atom = _get(feed_url, timeout=timeout)
    xml_urls = extract_cap_xml_urls_from_atom(atom)

    if args.reprocess:
        new_urls = xml_urls[:max_new]
    else:
        new_urls = [u for u in xml_urls if u not in seen][:max_new]
    print(f"[cap] feed={feed_key} urls_in_feed={len(xml_urls)} new_candidates={len(new_urls)}")

    rows = []
    kept = 0
    for i, url in enumerate(new_urls, 1):
        xml = _get(url, timeout=timeout)
        rec = parse_cap_xml(xml)
        rec["source_url"] = url

        # Time filter (uses sent)
        if not _time_in_window(rec.get("sent"), start, end):
            seen.add(url)
            continue

        # BBox filter (best-effort using CAP polygons)
        match = _bbox_match(rec.get("polygons") or [], bbox_tuple)
        rec["bbox_match"] = match
        if bbox_tuple is not None:
            # If we can evaluate and it's false, drop it.
            # If match is None (no polygons), keep it but bbox_match=None.
            if match is False:
                seen.add(url)
                continue

        # save raw xml
        fname = (rec.get("identifier") or f"cap_{int(time.time())}_{i}").replace(":", "_").replace("/", "_")
        raw_path = os.path.join(raw_dir, f"{fname}.xml")
        with open(raw_path, "wb") as f:
            f.write(xml)
        rec["raw_path"] = raw_path

        rows.append(rec)
        kept += 1
        seen.add(url)

    state["seen_xml_urls"] = sorted(seen)
    _save_state(state_path, state)

    df = pd.DataFrame(rows)
    if not df.empty:
        # Store polygons as JSON string for parquet friendliness
        df["polygons_json"] = df["polygons"].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else None)
        df = df.drop(columns=["polygons"])

        out_path = os.path.join(out_dir, "cap_alerts_latest.parquet")
        df.to_parquet(out_path, index=False)
        df.to_csv(os.path.join(out_dir, "cap_alerts_latest.csv"), index=False)

        print(f"[cap] kept={kept} saved -> {out_path}")
        if bbox_tuple is not None:
            print(f"[cap] bbox filtering enabled; shapely={HAS_SHAPELY}. bbox_match True/False/None counts:")
            print(df["bbox_match"].value_counts(dropna=False).to_string())
    else:
        print("[cap] no records kept (after filtering)")

    return df

def main():
    p = argparse.ArgumentParser("FMI CAP ingest (feed-based) with local timeframe+bbox filtering")
    p.add_argument("--feed", choices=["fi", "sv", "en"], default="en", help="Which language feed to ingest")
    p.add_argument("--max-new", type=int, default=200, help="Max new CAP items to fetch from feed in one run")
    p.add_argument("--raw-dir", type=str, default="data/raw/cap_warnings")
    p.add_argument("--out-dir", type=str, default="data/interim/cap_parsed")
    p.add_argument("--state-path", type=str, default="data/interim/cap_parsed/cap_state.json")
    p.add_argument("--timeout", type=int, default=30)

    # Timeframe options
    p.add_argument("--start", type=str, default=None, help="ISO UTC start (e.g. 2025-12-29T00:00:00Z)")
    p.add_argument("--end", type=str, default=None, help="ISO UTC end (e.g. 2025-12-29T18:00:00Z)")
    p.add_argument("--days", type=int, default=None, help="Convenience: keep only last N days (overrides --start/--end)")

    # Bbox option
    p.add_argument("--bbox", type=str, default=None, help="minLon,minLat,maxLon,maxLat (filters using CAP polygons if available)")

    p.add_argument("--reprocess", action="store_true",
                   help="Reprocess feed items even if already seen (ignores state for selection).")

    p.add_argument("--time-field", choices=["sent", "onset_expires", "effective_expires"],
                   default="onset_expires",
                   help="Which CAP time to filter on. 'onset_expires' checks overlap with hazard period.")

    args = p.parse_args()

    # Resolve timeframe
    start_dt = end_dt = None
    if args.days is not None:
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=int(args.days))
    else:
        if args.start:
            start_dt = parse_iso_utc(args.start)
        if args.end:
            end_dt = parse_iso_utc(args.end)

    bbox_tuple = parse_bbox(args.bbox) if args.bbox else None
    if bbox_tuple is not None and not HAS_SHAPELY:
        print("[cap] WARNING: shapely not installed; bbox filtering will be 'None' (cannot evaluate polygons).")

    run(
        feed_key=args.feed,
        max_new=args.max_new,
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        state_path=args.state_path,
        start=start_dt,
        end=end_dt,
        bbox_tuple=bbox_tuple,
        timeout=args.timeout,
    )

if __name__ == "__main__":
    main()