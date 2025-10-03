# app.py
import io
import re
import json
import tempfile
from datetime import date, timedelta, datetime
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# Google Drive API (service account)
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload

# DuckDB
import duckdb

# =========================
# Page + Global Config
# =========================
st.set_page_config(page_title="Center Shiftwise Web Dashboard", layout="wide")

# =========================
# Secrets (read once)
# =========================
SA_INFO = st.secrets.get("gcp_service_account", {})
DRIVE_FOLDER_ID = st.secrets.get("DRIVE_FOLDER_ID", "").strip()
DUCKDB_FILE_NAME = st.secrets.get("DUCKDB_FILE_NAME", "").strip() or "cmb_delta.duckdb"
DUCKDB_FILE_ID = st.secrets.get("DUCKDB_FILE_ID", "").strip()  # optional but best

# =========================
# Google Drive helpers (Shared-Drive aware + shortcut-safe)
# =========================
def _get_drive_service():
    if not SA_INFO:
        st.error("Missing st.secrets['gcp_service_account']. Add your service account JSON.")
        raise RuntimeError("No service account in secrets")
    scopes = ['https://www.googleapis.com/auth/drive']
    creds = service_account.Credentials.from_service_account_info(SA_INFO, scopes=scopes)
    return build('drive', 'v3', credentials=creds, cache_discovery=False)

def _drive_resolve_shortcut(service, file_obj):
    if file_obj.get("mimeType") == "application/vnd.google-apps.shortcut":
        target_id = file_obj.get("shortcutDetails", {}).get("targetId")
        if target_id:
            return service.files().get(fileId=target_id,
                                       fields="id,name,parents,mimeType,driveId",
                                       supportsAllDrives=True).execute()
    return file_obj

def _drive_get_file_by_id(service, file_id: str):
    return service.files().get(
        fileId=file_id,
        fields="id,name,parents,mimeType,driveId,shortcutDetails",
        supportsAllDrives=True
    ).execute()

def _drive_find_file_id(service, name: str, folder_id: Optional[str]) -> Optional[str]:
    """Find file by exact name (optionally inside folder). Works on Shared Drives and resolves shortcuts."""
    q = f"name = '{name}' and trashed = false"
    if folder_id:
        q = f"name = '{name}' and '{folder_id}' in parents and trashed = false"
    resp = service.files().list(
        q=q,
        fields="files(id,name,parents,mimeType,shortcutDetails,driveId)",
        pageSize=50,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
        corpora="allDrives",
    ).execute()
    files = resp.get("files", [])
    if not files:
        return None
    files = [_drive_resolve_shortcut(service, f) for f in files]
    if folder_id:
        for f in files:
            if folder_id in (f.get("parents") or []):
                return f["id"]
    return files[0]["id"]

def _drive_download_bytes(service, file_id: str) -> bytes:
    req = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buf.getvalue()

def _drive_upload_bytes(service, name: str, data: bytes, mime: str, folder_id: Optional[str], file_id: Optional[str] = None) -> str:
    media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mime, resumable=False)
    if file_id:
        file = service.files().update(fileId=file_id, media_body=media, supportsAllDrives=True).execute()
        return file["id"]
    meta = {"name": name}
    if folder_id:
        meta["parents"] = [folder_id]
    file = service.files().create(body=meta, media_body=media, fields="id", supportsAllDrives=True).execute()
    return file["id"]

def _drive_list_folder(service, folder_id: str) -> list[dict]:
    items = []
    page_token = None
    while True:
        resp = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="nextPageToken, files(id,name,mimeType,shortcutDetails)",
            pageSize=200,
            pageToken=page_token,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            corpora="allDrives"
        ).execute()
        for f in resp.get("files", []):
            items.append(_drive_resolve_shortcut(service, f))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return items

# ===== Parquet helpers (fallback store, if you ever want it) =====
@st.cache_data(ttl=300)
def load_parquet_from_drive(name: str) -> pd.DataFrame:
    service = _get_drive_service()
    folder_id = DRIVE_FOLDER_ID or None
    file_id = _drive_find_file_id(service, name, folder_id)
    if not file_id:
        return pd.DataFrame()
    raw = _drive_download_bytes(service, file_id)
    try:
        return pd.read_parquet(io.BytesIO(raw))
    except Exception:
        try:
            return pd.read_csv(io.BytesIO(raw))
        except Exception:
            return pd.DataFrame()

def save_parquet_to_drive(name: str, df: pd.DataFrame):
    service = _get_drive_service()
    folder_id = DRIVE_FOLDER_ID or None
    file_id = _drive_find_file_id(service, name, folder_id)
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    _drive_upload_bytes(service, name, buf.getvalue(), "application/vnd.apache.parquet", folder_id, file_id)
    load_parquet_from_drive.clear()

# =========================
# DuckDB Load/Save (from/to Drive)
# =========================
def _download_duckdb_rw(name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Download DuckDB file to a temp path; return (local_path, file_id, error).
    Prefers DUCKDB_FILE_ID. Otherwise searches by name in DRIVE_FOLDER_ID (or all drives visible).
    """
    try:
        service = _get_drive_service()

        # Prefer explicit file ID if present (bypasses all name/folder/shortcut issues)
        if DUCKDB_FILE_ID:
            try:
                f = _drive_get_file_by_id(service, DUCKDB_FILE_ID)
                f = _drive_resolve_shortcut(service, f)
                raw = _drive_download_bytes(service, f["id"])
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f"-{name}")
                tmp.write(raw); tmp.flush(); tmp.close()
                return tmp.name, f["id"], None
            except Exception as e:
                return None, None, f"DUCKDB_FILE_ID lookup failed: {e}"

        # Else search by name (optionally within folder)
        file_id = _drive_find_file_id(service, name, DRIVE_FOLDER_ID or None)
        if not file_id:
            where = DRIVE_FOLDER_ID or "all drives visible to the service account"
            return None, None, f"File '{name}' not found in {where}."
        raw = _drive_download_bytes(service, file_id)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f"-{name}")
        tmp.write(raw); tmp.flush(); tmp.close()
        return tmp.name, file_id, None
    except Exception as e:
        return None, None, f"DuckDB download error: {e}"

def _upload_duckdb_back(local_path: str, file_id: str, name: str):
    service = _get_drive_service()
    with open(local_path, "rb") as f:
        data = f.read()
    _drive_upload_bytes(service, name, data, "application/octet-stream", DRIVE_FOLDER_ID or None, file_id)

@st.cache_data(ttl=300)
def load_from_duckdb(db_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str,str]]:
    """
    Read DuckDB tables into DataFrames. Returns (records, roster_long, diagnostics)
    """
    diags = {
        "db_file": db_name,
        "found": "no",
        "records_rows": "0",
        "roster_rows": "0",
        "note": "",
        "folder_id": DRIVE_FOLDER_ID or "(empty)",
        "file_id": DUCKDB_FILE_ID or "(not set)",
        "sa_email": SA_INFO.get("client_email", "<unknown>"),
    }
    local_path, file_id, err = _download_duckdb_rw(db_name)
    if err:
        diags["note"] = err
        return pd.DataFrame(), pd.DataFrame(), diags

    diags["found"] = "yes"
    try:
        con = duckdb.connect(local_path, read_only=True)
    except Exception as e:
        diags["note"] = f"connect error: {e}"
        return pd.DataFrame(), pd.DataFrame(), diags

    # records
    try:
        df_records = con.execute("""
            SELECT Center, Language, Shift, Metric, Date, Value FROM records
        """).df()
    except Exception:
        try:
            df_records = con.execute("SELECT Center, Language, Shift, Date, Value FROM records").df()
            df_records["Metric"] = "Requested"
            df_records = df_records[["Center","Language","Shift","Metric","Date","Value"]]
        except Exception as e:
            df_records = pd.DataFrame()
            diags["note"] += f" | records read error: {e}"

    # roster_long
    try:
        df_roster = con.execute("""
            SELECT AgentID, EmpID, AgentName, TLName, Status, WorkMode,
                   Center, Location, Language, SecondaryLanguage, LOB, FTPT,
                   BaseShift, Date, Shift
            FROM roster_long
        """).df()
    except Exception as e:
        df_roster = pd.DataFrame()
        diags["note"] += f" | roster_long read error: {e}"

    con.close()
    diags["records_rows"] = str(0 if df_records is None or df_records.empty else len(df_records))
    diags["roster_rows"]  = str(0 if df_roster is None or df_roster.empty else len(df_roster))
    return df_records, df_roster, diags

# ============ DuckDB write helpers ============
def _ensure_duckdb_schema(con: duckdb.DuckDBPyConnection):
    con.execute("""
        CREATE TABLE IF NOT EXISTS records (
            Center VARCHAR, Language VARCHAR, Shift VARCHAR,
            Metric VARCHAR, Date DATE, Value DOUBLE
        );
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS roster_long (
            AgentID VARCHAR, EmpID VARCHAR, AgentName VARCHAR, TLName VARCHAR,
            Status VARCHAR, WorkMode VARCHAR, Center VARCHAR, Location VARCHAR,
            Language VARCHAR, SecondaryLanguage VARCHAR, LOB VARCHAR, FTPT VARCHAR,
            BaseShift VARCHAR, Date DATE, Shift VARCHAR
        );
    """)

def duckdb_upsert_records(local_db_path: str, file_id: str, name_in_drive: str, new_rows: pd.DataFrame) -> int:
    if new_rows.empty: return 0
    need_cols = ["Center","Language","Shift","Metric","Date","Value"]
    for c in need_cols:
        if c not in new_rows.columns:
            raise RuntimeError(f"Missing column in new Requested rows: {c}")
    new_rows = new_rows.copy()
    new_rows["Metric"] = new_rows["Metric"].fillna("Requested")
    new_rows["Date"] = pd.to_datetime(new_rows["Date"]).dt.date

    con = duckdb.connect(local_db_path, read_only=False)
    _ensure_duckdb_schema(con)
    con.register("new_records_df", new_rows)
    con.execute("""
        MERGE INTO records AS t
        USING new_records_df AS s
        ON t.Center = s.Center
           AND t.Language = s.Language
           AND t.Shift = s.Shift
           AND t.Date = s.Date
        WHEN MATCHED THEN UPDATE SET
            Metric = s.Metric, Value = s.Value
        WHEN NOT MATCHED THEN INSERT (Center, Language, Shift, Metric, Date, Value)
        VALUES (s.Center, s.Language, s.Shift, s.Metric, s.Date, s.Value);
    """)
    con.close()
    _upload_duckdb_back(local_db_path, file_id, name_in_drive)
    load_from_duckdb.clear()
    return len(new_rows)

def duckdb_replace_roster(local_db_path: str, file_id: str, name_in_drive: str, roster_df: pd.DataFrame) -> int:
    if roster_df.empty: return 0
    roster_df = roster_df.copy()
    roster_df["Date"] = pd.to_datetime(roster_df["Date"]).dt.date
    con = duckdb.connect(local_db_path, read_only=False)
    _ensure_duckdb_schema(con)
    con.register("new_roster_df", roster_df)
    con.execute("DROP TABLE IF EXISTS roster_long;")
    con.execute("CREATE TABLE roster_long AS SELECT * FROM new_roster_df;")
    inserted = con.execute("SELECT COUNT(*) FROM roster_long;").fetchone()[0]
    con.close()
    _upload_duckdb_back(local_db_path, file_id, name_in_drive)
    load_from_duckdb.clear()
    return int(inserted)

# =========================
# Parsing helpers (Requested workbook)
# =========================
def _is_date_like(x) -> bool:
    try:
        if pd.isna(x): return False
        pd.to_datetime(x); return True
    except Exception:
        return False

def _is_language_token(x) -> bool:
    if not isinstance(x, str): return False
    s = x.strip()
    if not s or s.lower().startswith("shift"): return False
    if any(ch.isdigit() for ch in s): return False
    for ch in s:
        if not (ch.isalpha() or ch.isspace() or ch in "-_+/()"):
            return False
    return True

def _find_language_header(df: pd.DataFrame, shift_row: int, shift_col: int, win: int = 16, rows_up: int = 3):
    ncols = df.shape[1]
    for up in range(1, rows_up + 1):
        r = shift_row - up
        if r < 0: break
        for c in range(shift_col, max(-1, shift_col - win), -1):
            v = df.iat[r, c]
            if _is_language_token(v): return r, c, str(v).strip()
        for c in range(shift_col + 1, min(ncols, shift_col + 1 + win)):
            v = df.iat[r, c]
            if _is_language_token(v): return r, c, str(v).strip()
    return None, None, None

def _collect_contiguous_dates_in_row(df: pd.DataFrame, row: int, start_col: int):
    dates, cols = [], []
    c = start_col
    while c < df.shape[1] and _is_date_like(df.iat[row, c]):
        d = pd.to_datetime(df.iat[row, c]).date()
        dates.append(d); cols.append(c); c += 1
    return dates, cols

SHIFT_SPLIT = re.compile(r"\s*/\s*")
WOCL_VARIANTS = re.compile(r"^\s*W\s*O\s*[\+&/ ]\s*C\s*L\s*$", re.I)
TIME_RANGE_RE = re.compile(r"""^\s*(?:(?:\d{1,2}[:.]?\d{2})\s*-\s*(?:\d{1,2}[:.]?\d{2}))
                                (?:\s*/\s*(?:\d{1,2}[:.]?\d{2})\s*-\s*(?:\d{1,2}[:.]?\d{2}))*\s*$""", re.X)

def _is_off_code(s: str) -> Optional[str]:
    t = re.sub(r"\s+", "", str(s)).upper()
    if t in {"WO","OFF"}: return "WO"
    if t in {"CL","CO","COMPOFF","COMP-OFF"}: return "CL"
    if WOCL_VARIANTS.fullmatch(str(s)): return "WO+CL"
    return None

def _is_valid_shift_label(s: str) -> bool:
    if _is_off_code(s): return True
    return bool(TIME_RANGE_RE.fullmatch(str(s)))

def _normalize_shift_label(s: str) -> Optional[str]:
    off = _is_off_code(s)
    if off: return off
    txt = str(s).strip()
    if not _is_valid_shift_label(txt): return None
    parts = SHIFT_SPLIT.split(txt)
    norm_parts = []
    for part in parts:
        part = part.strip().upper().replace("HRS","").replace("HR","").replace(".",":")
        m = re.fullmatch(r"(\d{1,2})[:]?(\d{2})\s*-\s*(\d{1,2})[:]?(\d{2})", part)
        if not m: return None
        sh, sm, eh, em = map(int, m.groups())
        sh = max(0, min(23, sh)); eh = max(0, min(23, eh))
        sm = max(0, min(59, sm)); em = max(0, min(59, em))
        norm_parts.append(f"{sh:02d}{sm:02d}-{eh:02d}{em:02d}")
    return "/".join(norm_parts)

def parse_center_sheet(xls: pd.ExcelFile, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(xls, sheet_name, header=None)
    if df.empty:
        return pd.DataFrame(columns=['Center','Language','Shift','Metric','Date','Value'])
    nrows, ncols = df.shape
    out = []
    for i in range(1, nrows):
        for j in range(ncols):
            cell = df.iat[i, j]
            if isinstance(cell, str) and re.search(r"\bshift(?:\s*name|s)?\b", cell, flags=re.IGNORECASE):
                _, _, lang = _find_language_header(df, i, j)
                if not lang: continue
                first_date_col = None
                for c in range(j + 1, ncols):
                    if _is_date_like(df.iat[i, c]): first_date_col = c; break
                if first_date_col is None: continue
                dates, date_cols = _collect_contiguous_dates_in_row(df, i, first_date_col)
                if not dates: continue
                r, blanks = i + 1, 0
                while r < nrows:
                    lbl = df.iat[r, j]
                    if pd.isna(lbl) or str(lbl).strip() == "":
                        blanks += 1
                        if blanks >= 2: break
                        r += 1; continue
                    blanks = 0
                    shift_label_norm = _normalize_shift_label(str(lbl).strip())
                    if not shift_label_norm:
                        r += 1; continue
                    for d, dc in zip(dates, date_cols):
                        v = pd.to_numeric(df.iat[r, dc], errors="coerce")
                        if pd.notna(v):
                            out.append((sheet_name, lang, shift_label_norm, "Requested", d, float(v)))
                    r += 1
    cols = ['Center','Language','Shift','Metric','Date','Value']
    return pd.DataFrame(out, columns=cols) if out else pd.DataFrame(columns=cols)

def parse_workbook_to_df(file_bytes: bytes) -> pd.DataFrame:
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    EXCLUDED = {"Settings","Roster","Overall"}
    centers = [s for s in xls.sheet_names if s.casefold() not in {x.casefold() for x in EXCLUDED}]
    parts = []
    for s in centers:
        try:
            d = parse_center_sheet(xls, s)
            if not d.empty: parts.append(d)
        except Exception as e:
            st.warning(f"Failed parsing '{s}': {e}")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=['Center','Language','Shift','Metric','Date','Value'])

def read_roster_sheet(file_bytes: bytes) -> pd.DataFrame:
    try:
        return pd.read_excel(io.BytesIO(file_bytes), sheet_name="Roster")
    except Exception as e:
        st.warning(f"Could not read 'Roster' sheet: {e}")
        return pd.DataFrame()

# =========================
# Data model selection (prefer DuckDB)
# =========================
RECORDS_FILE = "records.parquet"
ROSTER_FILE  = "roster_long.parquet"

def _centers_union(records: pd.DataFrame, roster: pd.DataFrame) -> List[str]:
    a = set(records["Center"].unique().tolist()) if not records.empty and "Center" in records else set()
    b = set(roster["Center"].unique().tolist())  if not roster.empty  and "Center" in roster  else set()
    cs = sorted([c for c in (a | b) if c])
    return ["Overall"] + cs

def _languages_union(records: pd.DataFrame, roster: pd.DataFrame, center: Optional[str]) -> List[str]:
    def unique_lang(df):
        return set(df["Language"].dropna().astype(str).str.strip().unique().tolist()) if not df.empty and "Language" in df else set()
    if center in (None, "", "Overall"):
        return sorted([l for l in (unique_lang(records) | unique_lang(roster)) if l])
    else:
        a = unique_lang(records[records["Center"]==center]) if "Center" in records else set()
        b = unique_lang(roster[roster["Center"]==center]) if "Center" in roster else set()
        return sorted([l for l in (a | b) if l])

def load_store() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str,str]]:
    diags = {}
    if DUCKDB_FILE_NAME:
        records, roster, d = load_from_duckdb(DUCKDB_FILE_NAME)
        diags.update({"source":"duckdb", **d})
    else:
        records = load_parquet_from_drive(RECORDS_FILE)
        roster  = load_parquet_from_drive(ROSTER_FILE)
        diags.update({"source":"parquet",
                      "records_rows":str(0 if records.empty else len(records)),
                      "roster_rows": str(0 if roster.empty else len(roster)),
                      "found":"n/a","db_file":"n/a","note":"",
                      "folder_id": DRIVE_FOLDER_ID or "(empty)",
                      "file_id": DUCKDB_FILE_ID or "(not set)",
                      "sa_email": SA_INFO.get("client_email","<unknown>")})

    if not records.empty:
        records["Date"] = pd.to_datetime(records["Date"]).dt.date
        if "Metric" not in records or records["Metric"].isna().all():
            records["Metric"] = "Requested"
        if (records["Metric"] == "Requested").any():
            records = records[records["Metric"] == "Requested"].copy()

    if not roster.empty:
        roster["Date"] = pd.to_datetime(roster["Date"]).dt.date
        for c in ["AgentID","Language","Center","LOB","Shift"]:
            if c in roster.columns:
                roster[c] = roster[c].astype(str).str.strip()
    return records, roster, diags

# =========================
# Aggregation + Views
# =========================
def _pretty_int(x: float) -> int:
    try: return int(round(float(x)))
    except Exception: return 0

_TIME_RE_SINGLE = re.compile(r"^\d{4}-\d{4}$")

def _minutes_of(label: str) -> Optional[int]:
    s = str(label).strip()
    if not _TIME_RE_SINGLE.fullmatch(s): return None
    sh, sm, eh, em = int(s[0:2]), int(s[2:4]), int(s[5:7]), int(s[7:9])
    start = sh*60 + sm; end = eh*60 + em
    return (end - start) if end > start else None

def dedup_requested_split_pairs(pivot_df: pd.DataFrame) -> pd.DataFrame:
    if pivot_df.empty: return pivot_df
    df = pivot_df.copy()
    idx = list(df.index)
    single_seg_minutes = {s: (None if "/" in str(s) else _minutes_of(str(s))) for s in idx}
    for col in df.columns:
        val_to_5h, val_to_4h = {}, {}
        col_values = df[col]
        for s in idx:
            mins = single_seg_minutes.get(s)
            if mins not in (300, 240): continue
            v = col_values.get(s)
            if pd.isna(v): continue
            try: vv = float(v)
            except Exception: continue
            (val_to_5h if mins==300 else val_to_4h).setdefault(vv, set()).add(s)
        for v in set(val_to_4h.keys()) & set(val_to_5h.keys()):
            for row_4h in val_to_4h[v]:
                df.at[row_4h, col] = 0.0
    return df

def transform_to_interval_view(shift_df: pd.DataFrame) -> pd.DataFrame:
    if shift_df.empty: return pd.DataFrame(index=range(24))
    cols_dates = [pd.to_datetime(c).date() for c in shift_df.columns]
    shift_df = shift_df.copy(); shift_df.columns = cols_dates
    interval_df = pd.DataFrame(0.0, index=range(24), columns=cols_dates)
    def _minutes_pair(text: str) -> Optional[tuple[int,int]]:
        t = str(text).upper().strip().replace("HRS","").replace("HR","").replace(".",":")
        m = re.fullmatch(r"\s*(\d{1,2})[:]?(\d{2})\s*-\s*(\d{1,2})[:]?(\d{2})\s*", t)
        if not m: return None
        sh, sm, eh, em = map(int, m.groups())
        if not (0<=sh<=23 and 0<=eh<=23 and 0<=sm<=59 and 0<=em<=59): return None
        return sh*60+sm, eh*60+em
    def _add_span(to_date, start_min, end_min, counts):
        if end_min <= start_min: return
        if to_date not in interval_df.columns: interval_df[to_date] = 0.0
        for h in range(24):
            dur = max(0, min(end_min, (h+1)*60) - max(start_min, h*60))
            if dur > 0:
                interval_df.at[h, to_date] += float(counts) * (dur / 60.0)
    for shift_label, counts_series in shift_df.iterrows():
        if str(shift_label).upper() in {"WO","CL","WO+CL"}: continue
        for part in str(shift_label).split("/"):
            part = part.strip()
            if re.fullmatch(r"\d{4}-\d{4}", part):
                smin = int(part[:2]) * 60 + int(part[2:4])
                emin = int(part[5:7]) * 60 + int(part[7:9])
            else:
                pair = _minutes_pair(part)
                if not pair: continue
                smin, emin = pair
            for day, count in counts_series.items():
                if pd.isna(count) or float(count) == 0.0: continue
                if emin > smin:
                    _add_span(day, smin, emin, float(count))
                else:
                    _add_span(day, smin, 24*60, float(count))
                    _add_span(day + timedelta(days=1), 0, emin, float(count))
    final_cols = sorted(set(cols_dates) | set(interval_df.columns))
    return interval_df.reindex(columns=final_cols, fill_value=0.0)

# =========================
# Query utilities
# =========================
def union_dates(records: pd.DataFrame, roster: pd.DataFrame, center: Optional[str]) -> List[date]:
    rs = set()
    if not records.empty:
        rs |= set(records["Date"].unique().tolist()) if center in (None, "", "Overall") else set(records.loc[records["Center"]==center, "Date"].unique().tolist())
    if not roster.empty:
        rs |= set(roster["Date"].unique().tolist()) if center in (None, "", "Overall") else set(roster.loc[roster["Center"]==center, "Date"].unique().tolist())
    return sorted(list(rs))

def pivot_requested(records: pd.DataFrame, center: str, langs: List[str], start: date, end: date) -> pd.DataFrame:
    if records.empty or not langs: return pd.DataFrame()
    df = records[(records["Date"].between(start, end)) & (records["Language"].isin(langs))].copy()
    if center != "Overall": df = df[df["Center"] == center]
    if df.empty: return pd.DataFrame()
    p = df.groupby(["Shift","Date"], as_index=False)["Value"].sum()
    p = p.pivot(index="Shift", columns="Date", values="Value").fillna(0.0)
    p = p.reindex(sorted(p.columns), axis=1)
    return p

def roster_lobs(roster: pd.DataFrame, center: Optional[str], start: date, end: date) -> List[str]:
    if roster.empty or "LOB" not in roster.columns: return []
    df = roster[roster["Date"].between(start, end)]
    if center and center != "Overall": df = df[df["Center"] == center]
    return sorted([x for x in df["LOB"].dropna().astype(str).str.strip().unique().tolist() if x])

def roster_languages_for_lob(roster: pd.DataFrame, center: Optional[str], lob: Optional[str], start: date, end: date) -> List[str]:
    if roster.empty: return []
    df = roster[roster["Date"].between(start, end)]
    if center and center != "Overall": df = df[df["Center"] == center]
    if lob: df = df[df["LOB"] == lob]
    return sorted(df["Language"].dropna().astype(str).str.strip().unique().tolist())

def roster_distinct_shifts(roster: pd.DataFrame, center: Optional[str], start: date, end: date,
                           lob: Optional[str], language: Optional[str]) -> List[str]:
    if roster.empty: return []
    df = roster[roster["Date"].between(start, end)]
    if center and center != "Overall": df = df[df["Center"] == center]
    if lob: df = df[df["LOB"] == lob]
    if language: df = df[df["Language"] == language]
    pat = re.compile(r"^\d{4}-\d{4}(?:/\d{4}-\d{4})*$")
    vals = []
    for s in df["Shift"].dropna().astype(str):
        t = s.strip()
        if pat.fullmatch(t): vals.append(t)
    combined = [s for s in vals if "/" in s]
    parts_set = set()
    for s in combined:
        parts_set.update(p.strip() for p in s.split("/") if p.strip())
    vals_clean = [s for s in vals if ("/" in s) or (s not in parts_set)]
    def _key(s):
        first = s.split("/")[0]
        return (int(first[:2]) * 60 + int(first[2:4]), s)
    return sorted(dict.fromkeys(vals_clean), key=_key)

def roster_nonshift_codes(roster: pd.DataFrame, center: Optional[str], start: date, end: date,
                          lob: Optional[str], language: Optional[str]) -> List[str]:
    if roster.empty: return []
    df = roster[roster["Date"].between(start, end)]
    if center and center != "Overall": df = df[df["Center"] == center]
    if lob: df = df[df["LOB"] == lob]
    if language: df = df[df["Language"] == language]
    pat = re.compile(r"^\d{4}-\d{4}(?:/\d{4}-\d{4})*$")
    vals = []
    for s in df["Shift"].dropna().astype(str):
        t = s.strip()
        if t and not pat.fullmatch(t):
            vals.append(t)
    return sorted(dict.fromkeys(vals), key=lambda x: x.upper())

def pivot_roster_counts(roster: pd.DataFrame, center: str, languages_: List[str], start: date, end: date) -> pd.DataFrame:
    if roster.empty or not languages_: return pd.DataFrame()
    df = roster[(roster["Date"].between(start, end)) & (roster["Language"].isin(languages_))].copy()
    if center != "Overall": df = df[df["Center"] == center]
    if df.empty: return pd.DataFrame()
    time_rows, code_rows = [], []
    time_pat = re.compile(r"^\d{4}-\d{4}(?:/\d{4}-\d{4})*$")
    for _, row in df.iterrows():
        dt = row["Date"]; s = str(row["Shift"]).strip().upper()
        if s in {"WO","CL"}:
            code_rows.append((dt, s)); continue
        if time_pat.fullmatch(s):
            parts = [p.strip() for p in s.split("/") if p.strip()]
            for part in parts:
                if re.fullmatch(r"\d{4}-\d{4}", part):
                    time_rows.append((dt, part))
    p = pd.DataFrame()
    if time_rows:
        tdf = pd.DataFrame(time_rows, columns=["Date","Shift"])
        p = tdf.groupby(["Shift","Date"]).size().unstack("Date", fill_value=0).astype(float)
        p = p.reindex(sorted(p.columns), axis=1)
    if code_rows:
        cdf = pd.DataFrame(code_rows, columns=["Date","Code"])
        codes = cdf.groupby(["Date","Code"]).size().unstack("Code", fill_value=0)
        wocl = (codes.get("WO", 0) + codes.get("CL", 0)).astype(float)
        all_cols = sorted(set(p.columns) | set(wocl.index.tolist()))
        if p.empty: p = pd.DataFrame(columns=all_cols)
        else:       p = p.reindex(columns=all_cols, fill_value=0.0)
        p.loc["WO+CL"] = [float(wocl.get(c, 0.0)) for c in p.columns]
    return p

def add_total_row(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    out = df.copy()
    out.loc["Total"] = out.sum(axis=0)
    return out

def roster_agents_wide(roster: pd.DataFrame, center: Optional[str], lob: Optional[str], language: Optional[str],
                       agent_ids: Optional[List[str]], start: date, end: date) -> pd.DataFrame:
    if roster.empty: return pd.DataFrame()
    df = roster[roster["Date"].between(start, end)].copy()
    if center and center != "Overall": df = df[df["Center"] == center]
    if lob: df = df[df["LOB"] == lob]
    if language: df = df[df["Language"] == language]
    if agent_ids:
        ids = [str(x).strip() for x in agent_ids if str(x).strip()]
        df = df[df["AgentID"].isin(ids)]
    if df.empty: return pd.DataFrame()
    static_cols = [c for c in ["AgentID","AgentName","TLName","Status","WorkMode","Center","Location",
                               "Language","SecondaryLanguage","LOB","FTPT"] if c in df.columns]
    p = df.pivot_table(index=static_cols, columns="Date", values="Shift", aggfunc="first")
    if len(p.columns): p = p.reindex(sorted(p.columns), axis=1)
    p = p.reset_index()
    return p

# =========================
# Export utilities
# =========================
def export_excel(view_type: str, center: str, languages_sel: List[str], start: date, end: date,
                 records: pd.DataFrame, roster: pd.DataFrame) -> bytes:
    from openpyxl.utils import get_column_letter
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        req_s = dedup_requested_split_pairs(pivot_requested(records, center, languages_sel, start, end))
        ros_s = pivot_roster_counts(roster, center, languages_sel, start, end)
        all_dates = sorted(list(set(req_s.columns) | set(ros_s.columns)))
        all_shifts = sorted(list(set(req_s.index) | set(ros_s.index)))
        req_s = req_s.reindex(index=all_shifts, columns=all_dates, fill_value=0.0)
        ros_s = ros_s.reindex(index=all_shifts, columns=all_dates, fill_value=0.0)
        delt_s = ros_s - req_s
        req_s, ros_s, delt_s = add_total_row(req_s), add_total_row(ros_s), add_total_row(delt_s)

        if view_type in ["Shift_Wise Delta View", "Overall_Delta View"]:
            sheet_name = "Shift_Wise_View"
            req_s.to_excel(writer, sheet_name=sheet_name, index=True, startrow=1)
            ros_s.to_excel(writer, sheet_name=sheet_name, index=True, startrow=1, startcol=len(req_s.columns) + 2)
            delt_s.to_excel(writer, sheet_name=sheet_name, index=True, startrow=1,
                            startcol=len(req_s.columns) + len(ros_s.columns) + 4)
            ws = writer.sheets[sheet_name]
            for col_cells in ws.columns:
                max_len = max(len(str(c.value)) if c.value is not None else 0 for c in col_cells)
                ws.column_dimensions[get_column_letter(col_cells[0].column)].width = min(max_len + 2, 36)

        if view_type in ["Interval_Wise Delta View", "Overall_Delta View"]:
            req_i = transform_to_interval_view(req_s.drop('Total', errors='ignore'))
            ros_i = transform_to_interval_view(ros_s.drop('Total', errors='ignore'))
            all_cols_i = sorted(list(set(req_i.columns) | set(ros_i.columns)))
            req_i = req_i.reindex(columns=all_cols_i, fill_value=0.0)
            ros_i = ros_i.reindex(columns=all_cols_i, fill_value=0.0)
            delt_i = ros_i - req_i
            req_i, ros_i, delt_i = add_total_row(req_i), add_total_row(ros_i), add_total_row(delt_i)
            sheet_name = "Interval_Wise_View"
            req_i.to_excel(writer, sheet_name=sheet_name, index=True, startrow=1)
            ros_i.to_excel(writer, sheet_name=sheet_name, index=True, startrow=1, startcol=len(req_i.columns) + 2)
            delt_i.to_excel(writer, sheet_name=sheet_name, index=True, startrow=1,
                            startcol=len(req_i.columns) + len(ros_i.columns) + 4)
            ws = writer.sheets[sheet_name]
            for col_cells in ws.columns:
                max_len = max(len(str(c.value)) if c.value is not None else 0 for c in col_cells)
                ws.column_dimensions[get_column_letter(col_cells[0].column)].width = min(max_len + 2, 36)
    return output.getvalue()

# =========================
# UI
# =========================
st.title("📊 Center Shiftwise Web Dashboard")
st.caption("Google Drive-backed • Streamlit • DuckDB/Parquet (Shared Drive compatible)")

# Load current store
records, roster, diags = load_store()

with st.sidebar:
    st.subheader("🔐 Google Drive Setup")
    st.write("- Add service account JSON to **st.secrets['gcp_service_account']**."
             "\n- Set **st.secrets['DRIVE_FOLDER_ID']** to the Shared Drive folder that contains your DB."
             "\n- Optionally set **st.secrets['DUCKDB_FILE_ID']** to the file's ID to bypass name search.")
    if st.button("↻ Sync from Drive", use_container_width=True):
        load_parquet_from_drive.clear()
        load_from_duckdb.clear()
        st.rerun()

    st.divider()
    st.subheader("🧪 Data Check")
    st.write(f"**Source:** {diags.get('source')}")
    st.write(f"**Service account:** {diags.get('sa_email', SA_INFO.get('client_email','<unknown>'))}")
    st.write(f"**DRIVE_FOLDER_ID:** {diags.get('folder_id')}")
    st.write(f"**DuckDB file name:** {diags.get('db_file')}")
    st.write(f"**DUCKDB_FILE_ID:** {diags.get('file_id')}")
    st.write(f"**Found in Drive?** {'✅ Yes' if diags.get('found')=='yes' else '❌ No'}")
    st.write(f"**records rows:** {diags.get('records_rows')}")
    st.write(f"**roster_long rows:** {diags.get('roster_rows')}")
    if diags.get("note"): st.caption(f"_note_: {diags.get('note')}")

    if (records.empty) and (roster.empty):
        st.warning("Both `records` and `roster_long` are empty. Check filename, file/folder access, and table schemas.")

    with st.expander("🔍 List files in DRIVE_FOLDER_ID"):
        fid = DRIVE_FOLDER_ID
        if not fid:
            st.warning("DRIVE_FOLDER_ID is empty.")
        else:
            try:
                svc = _get_drive_service()
                items = _drive_list_folder(svc, fid)
                if not items:
                    st.info("No items visible to the service account in this folder.")
                else:
                    for f in items:
                        st.write(f"- **{f.get('name')}** (`{f.get('id')}`)")
            except Exception as e:
                st.error(f"List error: {e}")

    st.divider()
    st.subheader("⬆️ Import Files (Excel)")
    req_file = st.file_uploader("Requested workbook (.xlsx)", type=["xlsx"], key="req")
    if st.button("Import Requested", type="primary", use_container_width=True, disabled=not req_file):
        new_req = parse_workbook_to_df(req_file.read())
        if new_req.empty:
            st.warning("No Requested data parsed.")
        else:
            if DUCKDB_FILE_NAME:
                local_db, file_id, err = _download_duckdb_rw(DUCKDB_FILE_NAME)
                if err or not local_db or not file_id:
                    st.error(f"DuckDB issue: {err or 'File not found.'}")
                else:
                    try:
                        affected = duckdb_upsert_records(local_db, file_id, DUCKDB_FILE_NAME, new_req)
                        st.success(f"Upserted Requested rows: {affected}")
                    except Exception as e:
                        st.error(f"DuckDB upsert failed: {e}")
            else:
                cur = load_parquet_from_drive(RECORDS_FILE)
                if cur.empty:
                    out = new_req
                else:
                    keys = ["Center","Language","Shift","Date"]
                    left = cur.merge(new_req[keys].drop_duplicates(), on=keys, how="left", indicator=True)
                    keep = left["_merge"] == "left_only"
                    base = cur[keep].copy()
                    out = pd.concat([base, new_req], ignore_index=True)
                save_parquet_to_drive(RECORDS_FILE, out)
                st.success(f"Imported Requested rows: {len(new_req)}")

    rost_file = st.file_uploader("Roster workbook (.xlsx)", type=["xlsx"], key="rost")
    if st.button("Import Roster (replace)", use_container_width=True, disabled=not rost_file):
        raw = read_roster_sheet(rost_file.read())
        if raw.empty:
            st.warning("No Roster data found (need a 'Roster' sheet).")
        else:
            if DUCKDB_FILE_NAME:
                local_db, file_id, err = _download_duckdb_rw(DUCKDB_FILE_NAME)
                if err or not local_db or not file_id:
                    st.error(f"DuckDB issue: {err or 'File not found.'}")
                else:
                    try:
                        if "Date" not in raw or "Shift" not in raw:
                            st.error("Uploaded roster must have at least 'Date' and 'Shift' columns.")
                        else:
                            raw["Date"] = pd.to_datetime(raw["Date"]).dt.date
                            inserted = duckdb_replace_roster(local_db, file_id, DUCKDB_FILE_NAME, raw)
                            st.success(f"Roster replaced in DuckDB. Rows inserted: {inserted}")
                    except Exception as e:
                        st.error(f"DuckDB roster replace failed: {e}")
            else:
                save_parquet_to_drive(ROSTER_FILE, raw)
                st.success(f"Roster imported to Parquet. Rows: {len(raw)}")

    st.divider()
    st.subheader("📤 Export Current View")

# ========== Top filter row (union of centers/languages) ==========
c1, c2, c3, c4 = st.columns([1.5, 1.2, 1.3, 1.7])
with c1:
    center_vals = _centers_union(records, roster)
    center = st.selectbox("Center", center_vals, index=0 if center_vals else None)
with c2:
    ds = union_dates(records, roster, center)
    if ds: start_default, end_default = ds[0], ds[-1]
    else:  start_default = end_default = date.today()
    date_range = st.date_input("Date range", value=(start_default, end_default))
    start, end = (date_range if isinstance(date_range, tuple) else (start_default, end_default))
with c3:
    view_type = st.selectbox("View", ["Shift_Wise Delta View","Interval_Wise Delta View","Overall_Delta View"], index=0)
with c4:
    lang_choices = _languages_union(records, roster, center)
    langs_sel = st.multiselect("Languages", lang_choices, default=lang_choices)  # pick all by default

st.divider()

# Build data for views
req_shift = dedup_requested_split_pairs(pivot_requested(records, center, langs_sel, start, end))
ros_shift = pivot_roster_counts(roster, center, langs_sel, start, end)
all_dates = sorted(list(set(req_shift.columns) | set(ros_shift.columns)))
all_shifts = sorted(list(set(req_shift.index) | set(ros_shift.index)))
req_shift = req_shift.reindex(index=all_shifts, columns=all_dates, fill_value=0.0)
ros_shift = ros_shift.reindex(index=all_shifts, columns=all_dates, fill_value=0.0)
delt_shift = ros_shift - req_shift
req_shift_total = add_total_row(req_shift)
ros_shift_total = add_total_row(ros_shift)
delt_shift_total = add_total_row(delt_shift)

# KPI row
k1, k2, k3 = st.columns(3)
with k1: st.metric("Requested", _pretty_int(req_shift.values.sum()))
with k2: st.metric("Rostered", _pretty_int(ros_shift.values.sum()))
with k3: st.metric("Delta", _pretty_int(delt_shift.values.sum()))

# =========================
# Tabs: Dashboard | Plotter | Roster
# =========================
tab_dash, tab_plot, tab_roster = st.tabs(["Dashboard", "Plotter", "Roster"])

with tab_dash:
    sub1, sub2, sub3 = st.columns(3)
    with sub1:
        st.subheader("Requested – Shift-wise")
        st.dataframe(req_shift_total, use_container_width=True)
    with sub2:
        st.subheader("Rostered – Shift-wise")
        st.dataframe(ros_shift_total, use_container_width=True)
    with sub3:
        st.subheader("Delta – Shift-wise")
        st.dataframe(delt_shift_total, use_container_width=True)

    if view_type in ["Interval_Wise Delta View", "Overall_Delta View"]:
        st.markdown("---")
        st.subheader("Interval views")
        ci1, ci2, ci3 = st.columns(3)
        req_interval = transform_to_interval_view(req_shift)
        ros_interval = transform_to_interval_view(ros_shift)
        all_cols_i = sorted(list(set(req_interval.columns) | set(ros_interval.columns)))
        req_interval = req_interval.reindex(columns=all_cols_i, fill_value=0.0)
        ros_interval = ros_interval.reindex(columns=all_cols_i, fill_value=0.0)
        delt_interval = ros_interval - req_interval
        with ci1:
            st.caption("Requested – Interval-wise")
            st.dataframe(add_total_row(req_interval), use_container_width=True)
        with ci2:
            st.caption("Rostered – Interval-wise")
            st.dataframe(add_total_row(ros_interval), use_container_width=True)
        with ci3:
            st.caption("Delta – Interval-wise")
            st.dataframe(add_total_row(delt_interval), use_container_width=True)

with tab_plot:
    st.subheader("📈 Plotter")

    # ---- Time-series totals by day (safe concat guards) ----
    def _series_from_pivot(p: pd.DataFrame, label: str) -> pd.DataFrame:
        if p.empty: return pd.DataFrame(columns=["Date","Value","Metric"])
        s = p.sum(axis=0)
        out = pd.DataFrame({"Date": s.index, "Value": s.values})
        out["Metric"] = label
        return out

    ts_req  = _series_from_pivot(req_shift, "Requested")
    ts_ros  = _series_from_pivot(ros_shift, "Rostered")
    ts_del  = _series_from_pivot(delt_shift, "Delta")
    _frames = [x for x in [ts_req, ts_ros, ts_del] if x is not None and not x.empty]
    ts_all = pd.concat(_frames, ignore_index=True) if _frames else pd.DataFrame(columns=["Date","Value","Metric"])

    if ts_all.empty:
        st.info("No data to plot in the selected filters.")
    else:
        # safety: make sure types are correct
        ts_all["Date"] = pd.to_datetime(ts_all["Date"])
        ts_all["Value"] = pd.to_numeric(ts_all["Value"], errors="coerce").fillna(0)

        chart = alt.Chart(ts_all).mark_line(point=True).encode(
            x=alt.X("yearmonthdate(Date):T", title="Date"),
            y=alt.Y("Value:Q", title="Count"),
            color=alt.Color("Metric:N"),
            tooltip=["Metric", alt.Tooltip("Date:T"), alt.Tooltip("Value:Q")]
        ).properties(height=260, use_container_width=True)
    st.markdown("---")

    # ---- Shift-wise bar (sum over selected dates) ----
    def _melt_sum(p: pd.DataFrame, label: str) -> pd.DataFrame:
        if p.empty: return pd.DataFrame(columns=["Shift","Value","Metric"])
        m = p.reset_index().melt(id_vars="Shift", var_name="Date", value_name="Value")
        m = m.groupby("Shift", as_index=False)["Value"].sum(); m["Metric"] = label
        return m

    m_req = _melt_sum(req_shift, "Requested")
    m_ros = _melt_sum(ros_shift, "Rostered")
    m_del = _melt_sum(delt_shift, "Delta")
    _frames2 = [x for x in [m_req, m_ros, m_del] if x is not None and not x.empty]
    m_all = pd.concat(_frames2, ignore_index=True) if _frames2 else pd.DataFrame(columns=["Shift","Value","Metric"])

    if not m_all.empty:
        chart2 = alt.Chart(m_all).mark_bar().encode(
            x=alt.X("Shift:N", sort=None),
            y=alt.Y("Value:Q", stack=None),
            color="Metric:N",
            tooltip=["Metric","Shift","Value"]
        ).properties(height=300, use_container_width=True)
        st.altair_chart(chart2, use_container_width=True)
    else:
        st.info("No shift-wise data to plot for the current filters.")

    st.markdown("---")

    # ---- Interval heatmap for Delta ----
    if not delt_shift.empty:
        st.caption("Interval heatmap (Delta) – hours vs dates")
        delt_interval = transform_to_interval_view(delt_shift)
        if not delt_interval.empty:
            heat = delt_interval.copy()
            heat.index.name = "Hour"
            heat = heat.reset_index().melt(id_vars="Hour", var_name="Date", value_name="Hours")
            heat["Date"] = pd.to_datetime(heat["Date"])
            heat["Hours"] = pd.to_numeric(heat["Hours"], errors="coerce").fillna(0)

            hchart = alt.Chart(heat).mark_rect().encode(
                x=alt.X("yearmonthdate(Date):T", title="Date"),
                y=alt.Y("Hour:O", title="Hour of day"),
                color=alt.Color("Hours:Q", title="Delta hours"),
                tooltip=[alt.Tooltip("Date:T"), "Hour:O", alt.Tooltip("Hours:Q")]
            ).properties(height=360, use_container_width=True)

with tab_roster:
    # Roster Filters
    r1, r2, r3, r4, r5 = st.columns([1,1,1,1.3,1.2])
    with r1:
        lobs = roster_lobs(roster, center, start, end)
        lob_sel = st.selectbox("LOB", ["(All)"] + lobs)
        lob = None if lob_sel in ("", "(All)") else lob_sel
    with r2:
        langs_lob = roster_languages_for_lob(roster, center, lob, start, end)
        rlang_sel = st.selectbox("Language", ["(All)"] + langs_lob)
        rlang = None if rlang_sel in ("", "(All)") else rlang_sel
    with r3:
        codes = roster_nonshift_codes(roster, center, start, end, lob, rlang)
        nonshift_sel = st.selectbox("Non-Shift", ["(All)"] + codes)
    with r4:
        shifts_list = roster_distinct_shifts(roster, center, start, end, lob, rlang)
        shift_sel = st.selectbox("Shift", ["(All)"] + shifts_list)
    with r5:
        ids_text = st.text_input("Genesys IDs (comma/space)")
        ids = [x.strip() for x in re.findall(r"[A-Za-z0-9_]+", ids_text)] if ids_text else None

    # Build wide table
    wide = roster_agents_wide(roster, center, lob, rlang, ids, start, end)
    if not wide.empty:
        date_cols = [c for c in wide.columns if isinstance(c, (pd.Timestamp, date)) or re.match(r"\d{4}-\d{2}-\d{2}", str(c))]
        static_cols = [c for c in wide.columns if c not in date_cols]
        def is_time_like(s: str) -> bool:
            return bool(re.fullmatch(r"\d{4}-\d{4}(?:/\d{4}-\d{4})*", str(s).strip()))
        if nonshift_sel not in ("", "(All)"):
            for c in date_cols:
                wide[c] = wide[c].apply(lambda x: (str(x).strip() if (str(x).strip() and not is_time_like(str(x)) and str(x).strip().upper()==nonshift_sel.upper()) else ""))
        elif shift_sel not in ("", "(All)"):
            single_pat = re.compile(r"^\d{4}-\d{4}$")
            time_like = re.compile(r"^\d{4}-\d{4}(?:/\d{4}-\d{4})*$")
            chosen_is_single = bool(single_pat.fullmatch(shift_sel))
            def keep_if_matches(x: str) -> str:
                s = str(x).strip()
                if not s: return ""
                if not time_like.fullmatch(s): return ""
                if s == shift_sel: return s
                if chosen_is_single and "/" in s:
                    parts = [p.strip() for p in s.split("/") if p.strip()]
                    if shift_sel in parts: return s
                return ""
            for c in date_cols:
                wide[c] = wide[c].apply(keep_if_matches)
        if nonshift_sel not in ("", "(All)") or shift_sel not in ("", "(All)"):
            mask_rows = wide[date_cols].astype(str).apply(lambda r: any(v.strip() for v in r), axis=1)
            wide = wide[mask_rows].copy()
            nonempty_cols = [c for c in date_cols if wide[c].astype(str).str.strip().ne("").any()]
            wide = pd.concat([wide[static_cols], wide[nonempty_cols]], axis=1)
    st.dataframe(wide, use_container_width=True)
