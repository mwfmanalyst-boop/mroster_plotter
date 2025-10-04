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
# ✨ NEW: Auth / RBAC (lightweight shell, SSO-ready)
# =========================
# How it works now:
# - By default, runs in "demo auth" mode: a simple sign-in form that sets user & role in session.
# - SSO/JWT-ready: if you put a JWT in query param `token` (or configure to read a header via reverse proxy)
#   and set secrets["auth"]["jwt_public_key"], we will validate it and extract `email` and `role`.
# - Roles supported: "admin" (full access), "viewer" (read-only, no edits).
#
# To wire a real SSO:
# - Terminate auth at your proxy / IdP; forward a JWT as a query param or header.
# - Provide the RS256 public key in st.secrets["auth"]["jwt_public_key"] (PEM).
# - Ensure the token includes "email" and "role" claims (role in {"admin","viewer"}).
#
# IMPORTANT: All edit paths are gated on role == "admin".

def _auth_try_decode_jwt(token: str) -> Optional[dict]:
    try:
        import jwt  # PyJWT
    except Exception:
        return None
    pub = st.secrets.get("auth", {}).get("jwt_public_key")
    if not pub:
        return None
    try:
        data = jwt.decode(token, pub, algorithms=["RS256"], options={"verify_aud": False})
        return data
    except Exception:
        return None

def _auth_sign_in_ui():
    st.markdown("<div class='card'><h3>🔐 Sign in</h3>", unsafe_allow_html=True)
    demo = st.toggle("Use demo login (no SSO)", value=True, help="Disable to rely on JWT in ?token=…")
    if demo:
        email = st.text_input("Work email", value="user@example.com")
        role = st.selectbox("Role", ["viewer", "admin"], index=1)
        if st.button("Sign in", type="primary", use_container_width=True):
            st.session_state["auth_user"] = {"email": email, "role": role}
            st.rerun()
    else:
        st.info("Provide a JWT in the URL query like ?token=eyJ... (RS256).")
    st.markdown("</div>", unsafe_allow_html=True)

def require_auth():
    # 1) Try JWT in query param (SSO mode)
    params = st.experimental_get_query_params()
    token = None
    if "token" in params and params["token"]:
        token = params["token"][0]
    if token:
        decoded = _auth_try_decode_jwt(token)
        if decoded:
            user = {
                "email": decoded.get("email", "unknown@user"),
                "role": decoded.get("role", "viewer"),
                "claims": decoded
            }
            st.session_state["auth_user"] = user

    user = st.session_state.get("auth_user")
    if not user:
        st.set_page_config(page_title="Center Shiftwise Web Dashboard", layout="wide")
        _inject_css()  # minimal styling even on login page
        st.title("📊 Center Shiftwise Web Dashboard")
        _auth_sign_in_ui()
        st.stop()

    # Basic header bar
    _top_nav(user)
    return user

def _top_nav(user: dict):
    with st.container():
        cols = st.columns([6, 2, 1])
        with cols[0]:
            st.markdown("<h1>📊 Center Shiftwise Web Dashboard</h1>", unsafe_allow_html=True)
            st.caption("Google Drive-backed • Streamlit • DuckDB/Parquet • SSO-ready (Role-based)")
        with cols[1]:
            st.markdown(
                f"<div style='text-align:right;padding-top:10px;'>"
                f"<b>{user.get('email','')}</b><br/><span class='badge'>{user.get('role','viewer')}</span>"
                f"</div>",
                unsafe_allow_html=True
            )
        with cols[2]:
            if st.button("Sign out", use_container_width=True):
                st.session_state.pop("auth_user", None)
                st.experimental_set_query_params()  # clear ?token
                st.rerun()

# =========================
# ✨ NEW: Modern UI styling (CSS)
# =========================
def _inject_css():
    st.markdown(
        """
        <style>
        :root{
            --card-bg: #ffffff;
            --card-br: 18px;
            --card-shadow: 0 10px 25px rgba(0,0,0,0.06);
            --accent: #4c8bf5;
            --soft: #f6f8fe;
        }
        .block-container { padding-top: 1.5rem; max-width: 1400px; }
        .badge{ background:#EEF2FF; color:#3730A3; padding:2px 8px; border-radius:999px; font-size:12px; }
        .metric-card .stMetric { background:var(--card-bg); border-radius:var(--card-br); 
            box-shadow: var(--card-shadow); padding: 12px 16px; }
        .card{
            background:var(--card-bg); border-radius:var(--card-br); padding:16px 18px; 
            box-shadow: var(--card-shadow); margin-bottom: 16px;
        }
        .muted{ color:#6b7280; }
        .stDataFrame { border-radius: 14px; overflow: hidden; box-shadow: var(--card-shadow); }
        .stDownloadButton, .st-emotion-cache-1vt4y43 { width:100%; }
        </style>
        """,
        unsafe_allow_html=True
    )

# =========================
# Page + Global Config
# =========================
st.set_page_config(page_title="Center Shiftwise Web Dashboard", layout="wide")
_injected_once = st.session_state.get("_css_injected", False)
if not _injected_once:
    _inject_css()
    st.session_state["_css_injected"] = True

# --- UI state ---
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Dashboard"  # "Dashboard" | "Plotter" | "Roster"

if "busy_roster" not in st.session_state:
    st.session_state.busy_roster = False

# ✨ NEW: runtime override for Drive folder (so you can fetch by an ID you share)
if "override_drive_folder_id" not in st.session_state:
    st.session_state.override_drive_folder_id = None

# =========================
# Secrets (read once)
# =========================
SA_INFO = st.secrets.get("gcp_service_account", {})
# ✨ UPDATED: prefer runtime override if set
def _current_drive_folder_id():
    return (st.session_state.override_drive_folder_id or st.secrets.get("DRIVE_FOLDER_ID", "")).strip()

DRIVE_FOLDER_ID = _current_drive_folder_id()
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

# ===== Parquet helpers =====
@st.cache_data(ttl=300)
def load_parquet_from_drive(name: str) -> pd.DataFrame:
    service = _get_drive_service()
    folder_id = _current_drive_folder_id() or None
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
    folder_id = _current_drive_folder_id() or None
    file_id = _drive_find_file_id(service, name, folder_id)
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    _drive_upload_bytes(service, name, buf.getvalue(), "application/vnd.apache.parquet", folder_id, file_id)
    load_parquet_from_drive.clear()

# =========================
# DuckDB Load/Save (from/to Drive)
# =========================
def _download_duckdb_rw(name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    try:
        service = _get_drive_service()

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

        file_id = _drive_find_file_id(service, name, _current_drive_folder_id() or None)
        if not file_id:
            where = _current_drive_folder_id() or "all drives visible to the service account"
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
    _drive_upload_bytes(service, name, data, "application/octet-stream", _current_drive_folder_id() or None, file_id)

@st.cache_data(ttl=300)
def load_from_duckdb(db_name: str, _schema_version: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str,str]]:
    diags = {
        "db_file": db_name,
        "found": "no",
        "records_rows": "0",
        "roster_rows": "0",
        "note": "",
        "folder_id": _current_drive_folder_id() or "(empty)",
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
        df_records = con.execute("""SELECT Center, Language, Shift, Metric, Date, Value FROM records""").df()
    except Exception:
        try:
            df_records = con.execute("SELECT Center, Language, Shift, Date, Value FROM records").df()
            df_records["Metric"] = "Requested"
            df_records = df_records[["Center","Language","Shift","Metric","Date","Value"]]
        except Exception as e:
            df_records = pd.DataFrame()
            diags["note"] += f" | records read error: {e}"

    # roster_long (schema-tolerant)
    try:
        info = con.execute("PRAGMA table_info('roster_long')").df()
        if info.empty:
            raise Exception("table 'roster_long' not found or has no columns")
        name_map = {str(n).strip().lower(): str(n).strip() for n in info["name"].astype(str)}
        def has(col): return col.lower() in name_map
        def col(col): return name_map.get(col.lower(), col)
        select_bits = []
        def add_varchar(target_name, source_name=None):
            if source_name and has(source_name):
                select_bits.append(f"{col(source_name)} AS {target_name}")
            elif has(target_name):
                select_bits.append(f"{col(target_name)}")
            else:
                select_bits.append(f"CAST(NULL AS VARCHAR) AS {target_name}")
        add_varchar("AgentID"); add_varchar("EmpID"); add_varchar("AgentName"); add_varchar("TLName")
        add_varchar("Status"); add_varchar("WorkMode" if has("WorkMode") else "Reason")  # WorkMode/Reason
        add_varchar("Center"); add_varchar("Location")
        add_varchar("Language"); add_varchar("SecondaryLanguage"); add_varchar("LOB"); add_varchar("FTPT")
        add_varchar("BaseShift")
        select_bits.append(f"{col('Date')}" if has("Date") else "CAST(NULL AS DATE) AS Date")
        if has("Shift"): select_bits.append(f"{col('Shift')}")
        elif has("Reason"): select_bits.append(f"{col('Reason')} AS Shift")
        else: select_bits.append("CAST(NULL AS VARCHAR) AS Shift")
        sql = "SELECT " + ", ".join(select_bits) + " FROM roster_long"
        df_roster = con.execute(sql).df()
    except Exception as e:
        df_roster = pd.DataFrame()
        diags["note"] += f" | roster_long read error: {e}"

    try: con.close()
    except Exception: pass

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

# ✨ NEW: Upsert for roster single edits (merge on AgentID+Date)
def duckdb_upsert_roster(local_db_path: str, file_id: str, name_in_drive: str, roster_rows: pd.DataFrame) -> int:
    if roster_rows.empty: return 0
    df = roster_rows.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    con = duckdb.connect(local_db_path, read_only=False)
    _ensure_duckdb_schema(con)
    con.register("patch_df", df)
    # Update known editable columns; expand as needed.
    con.execute("""
        MERGE INTO roster_long AS t
        USING patch_df AS s
        ON COALESCE(t.AgentID,'') = COALESCE(s.AgentID,'')
           AND t.Date = s.Date
        WHEN MATCHED THEN UPDATE SET
            Shift = COALESCE(s.Shift, t.Shift),
            WorkMode = COALESCE(s.WorkMode, t.WorkMode),
            Status = COALESCE(s.Status, t.Status)
        WHEN NOT MATCHED THEN INSERT SELECT * FROM s;
    """)
    n = con.execute("SELECT COUNT(*) FROM patch_df;").fetchone()[0]
    con.close()
    _upload_duckdb_back(local_db_path, file_id, name_in_drive)
    load_from_duckdb.clear()
    return int(n)

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
        try:
            records, roster, d = load_from_duckdb(DUCKDB_FILE_NAME, _schema_version=2)
        except Exception:
            try:
                load_from_duckdb.clear()
            except Exception:
                pass
            records, roster, d = load_from_duckdb(DUCKDB_FILE_NAME, _schema_version=2)
        diags.update({"source": "duckdb", **d})

    else:
        records = load_parquet_from_drive(RECORDS_FILE)
        roster  = load_parquet_from_drive(ROSTER_FILE)
        diags.update({"source":"parquet",
                      "records_rows":str(0 if records.empty else len(records)),
                      "roster_rows": str(0 if roster.empty else len(roster)),
                      "found":"n/a","db_file":"n/a","note":"",
                      "folder_id": _current_drive_folder_id() or "(empty)",
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
        for c in ["AgentID", "Language", "Center", "LOB", "Shift"]:
            if c in roster.columns:
                roster[c] = roster[c].astype(str).str.strip()
        if "Shift" in roster.columns:
            roster["Shift"] = roster["Shift"].apply(
                lambda s: (_normalize_shift_label(s) or str(s).strip().upper())
            )

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

def pivot_requested(records, center, langs, start, end) -> pd.DataFrame:
    if records.empty or not langs: return pd.DataFrame()
    df = records[(records["Date"].between(start, end)) & (records["Language"].isin(langs))].copy()
    if center != "Overall": df = df[df["Center"] == center]
    if df.empty: return pd.DataFrame()
    p = df.groupby(["Shift","Date"], as_index=False)["Value"].sum()
    p = p.pivot(index="Shift", columns="Date", values="Value").fillna(0.0)
    p = p.reindex(sorted(p.columns), axis=1)
    p.index.name = "Shift"
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

def pivot_roster_counts(roster: pd.DataFrame,
                        center: Optional[str],
                        languages_: List[str],
                        start: date,
                        end: date) -> pd.DataFrame:
    if roster.empty or not languages_:
        return pd.DataFrame()

    df = roster[roster["Date"].between(start, end)].copy()
    if center and center != "Overall":
        df = df[df["Center"] == center]
    df = df[df["Language"].isin(languages_)]
    if df.empty:
        return pd.DataFrame()

    time_rows: list[tuple[date, str]] = []
    code_rows: list[tuple[date, str]] = []

    for _, row in df.iterrows():
        dt = row["Date"]
        raw = str(row.get("Shift", "")).strip()
        norm = _normalize_shift_label(raw)
        if not norm:
            continue
        if norm in {"WO", "CL", "WO+CL"}:
            code_rows.append((dt, "WO" if norm == "WO" else ("CL" if norm == "CL" else "WO+CL")))
            continue
        parts = [p.strip() for p in str(norm).split("/") if p.strip()]
        for p in parts:
            if re.fullmatch(r"\d{4}-\d{4}", p):
                time_rows.append((dt, p))

    p = pd.DataFrame()
    if time_rows:
        tdf = pd.DataFrame(time_rows, columns=["Date", "Shift"])
        p = tdf.groupby(["Shift", "Date"]).size().unstack("Date", fill_value=0).astype(float)
        p = p.reindex(sorted(p.columns), axis=1)

    if code_rows:
        cdf = pd.DataFrame(code_rows, columns=["Date", "Code"])
        codes = cdf.groupby(["Date", "Code"]).size().unstack("Code", fill_value=0)
        w = codes.get("WO", 0)
        c = codes.get("CL", 0)
        wc = codes.get("WO+CL", 0)
        wocl = (w + c + wc).astype(float)

        all_cols = sorted(set(p.columns) | set(wocl.index.tolist()))
        p = (pd.DataFrame(columns=all_cols) if p.empty else p.reindex(columns=all_cols, fill_value=0.0))
        p.loc["WO+CL"] = [float(wocl.get(col, 0.0)) for col in p.columns]

    p.index.name = "Shift"
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
def _safe_name(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', s)

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

# === Requested editor helpers (desktop-style) ===
def _validate_shift_label_for_edit(s: str) -> Optional[str]:
    if pd.isna(s) or str(s).strip() == "":
        return None
    return _normalize_shift_label(str(s).strip())

def _pivot_requested_one_lang(records: pd.DataFrame, center: str, language: str, start: date, end: date) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame()
    df = records[(records["Date"].between(start, end)) & (records["Language"] == language)].copy()
    if center and center != "Overall":
        df = df[df["Center"] == center]
    if df.empty:
        return pd.DataFrame()
    p = df.groupby(["Shift","Date"], as_index=False)["Value"].sum()
    p = p.pivot(index="Shift", columns="Date", values="Value").fillna(0.0)
    p = p.reindex(sorted(p.columns), axis=1)
    p.index.name = "Shift"
    return p

def _save_requested_edits(center: str, language: str, edited_df: pd.DataFrame, start: date, end: date) -> Tuple[int, str]:
    if edited_df is None or edited_df.empty:
        return 0, "Nothing to save."
    new_df = edited_df.copy()
    try:
        new_df.columns = [pd.to_datetime(c).date() for c in new_df.columns]
    except Exception:
        new_df = new_df[[c for c in new_df.columns if isinstance(c, (pd.Timestamp, date))]]
    if new_df.empty:
        return 0, "No valid date columns after validation."
    new_df = new_df.reset_index().rename(columns={new_df.index.name or "index": "Shift"})
    new_df["ShiftNorm"] = new_df["Shift"].apply(_validate_shift_label_for_edit)
    new_df = new_df[~new_df["ShiftNorm"].isna()].copy()
    if new_df.empty:
        return 0, "All rows had invalid/empty shift labels."
    new_df = new_df.drop(columns=["Shift"]).rename(columns={"ShiftNorm": "Shift"}).set_index("Shift")

    out_long = new_df.reset_index().melt(id_vars="Shift", var_name="Date", value_name="Value")
    out_long = out_long.dropna(subset=["Value"])
    if out_long.empty:
        return 0, "No numeric values to save."

    out_long["Value"] = pd.to_numeric(out_long["Value"], errors="coerce")
    out_long = out_long.dropna(subset=["Value"])
    if out_long.empty:
        return 0, "No numeric values to save."

    out_long["Center"] = center
    out_long["Language"] = language
    out_long["Metric"] = "Requested"
    out_long = out_long[["Center","Language","Shift","Metric","Date","Value"]].copy()

    if DUCKDB_FILE_NAME:
        local_db, file_id, err = _download_duckdb_rw(DUCKDB_FILE_NAME)
        if err or not local_db or not file_id:
            return 0, f"DuckDB issue: {err or 'File not found.'}"
        try:
            n = duckdb_upsert_records(local_db, file_id, DUCKDB_FILE_NAME, out_long)
            return n, f"Saved {n} Requested cells for {language}."
        except Exception as e:
            return 0, f"DuckDB upsert failed: {e}"
    else:
        cur = load_parquet_from_drive(RECORDS_FILE)
        if cur.empty:
            save_parquet_to_drive(RECORDS_FILE, out_long)
            load_parquet_from_drive.clear()
            return len(out_long), f"Saved {len(out_long)} Requested cells for {language}."
        keys = ["Center","Language","Shift","Date"]
        left = cur.merge(out_long[keys].drop_duplicates(), on=keys, how="left", indicator=True)
        keep = left["_merge"] == "left_only"
        base = cur[keep].copy()
        new_all = pd.concat([base, out_long], ignore_index=True)
        save_parquet_to_drive(RECORDS_FILE, new_all)
        load_parquet_from_drive.clear()
        return len(out_long), f"Saved {len(out_long)} Requested cells for {language}."

# =========================
# UI
# =========================

# ✨ NEW: Require sign-in & role
user = require_auth()
role = user.get("role", "viewer")
is_admin = (role == "admin")

# Load current store
records, roster, diags = load_store()

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔐 Google Drive Setup")
    st.write("- Add service account JSON to **st.secrets['gcp_service_account']**."
             "\n- Set **st.secrets['DRIVE_FOLDER_ID']** to the Shared Drive folder that contains your DB."
             "\n- Optionally set **st.secrets['DUCKDB_FILE_ID']** to the file's ID to bypass name search.")
    if st.button("↻ Sync from Drive", use_container_width=True):
        load_parquet_from_drive.clear()
        load_from_duckdb.clear()
        st.rerun()

    # ✨ NEW: Fetch by Folder ID (runtime)
    st.divider()
    st.subheader("📁 Fetch by Folder ID")
    folder_id_in = st.text_input("Shared Drive Folder ID", value=_current_drive_folder_id())
    if st.button("Use this Folder", use_container_width=True):
        st.session_state.override_drive_folder_id = folder_id_in.strip() or None
        load_parquet_from_drive.clear(); load_from_duckdb.clear()
        st.success("Folder set. Reloading …")
        st.rerun()

    st.divider()
    st.subheader("🧪 Data In Database ")
    st.write(f"**Records_Data:** {diags.get('records_rows')}")
    st.write(f"**Roster_Data:** {diags.get('roster_rows')}")
    if diags.get("note"): st.caption(f"_note_: {diags.get('note')}")

    if (records.empty) and (roster.empty):
        st.warning("Both `records` and `roster_long` are empty. Check filename, file/folder access, and table schemas.")

    st.divider()
    st.subheader("⬆️ Import Files (Excel)")
    # Requested
    req_file = st.file_uploader("Requested workbook (.xlsx)", type=["xlsx"], key="req")
    if st.button("Import Requested", type="primary", use_container_width=True, disabled=(not req_file or not is_admin)):
        if not is_admin:
            st.error("You need admin role to import.")
        else:
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

    # Roster
    rost_file = st.file_uploader("Roster workbook (.xlsx)", type=["xlsx"], key="rost")
    if st.button("Import Roster (replace)", use_container_width=True, disabled=(not rost_file or not is_admin)):
        if not is_admin:
            st.error("You need admin role to import.")
        else:
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

    st.markdown("</div>", unsafe_allow_html=True)

# ========== Top filter row (union of centers/languages) ==========
st.markdown("<div class='card'>", unsafe_allow_html=True)
top1, top2, top3, top4 = st.columns([1.5, 1.2, 1.3, 1.7])
with top1:
    center_vals = _centers_union(records, roster)
    center = st.selectbox("Center", center_vals, index=0 if center_vals else None)
with top2:
    ds = union_dates(records, roster, center)
    if ds: start_default, end_default = ds[0], ds[-1]
    else:  start_default = end_default = date.today()
    date_range = st.date_input("Date range", value=(start_default, end_default))
    start, end = (date_range if isinstance(date_range, tuple) else (start_default, end_default))
with top3:
    view_type = st.selectbox("View", ["Shift_Wise Delta View","Interval_Wise Delta View","Overall_Delta View"], index=0)
with top4:
    lang_choices = _languages_union(records, roster, center)
    langs_sel = st.multiselect("Languages", lang_choices, default=(lang_choices if lang_choices else []))
st.markdown("</div>", unsafe_allow_html=True)

# >>> EXPORT SIDEBAR (depends on center, view_type, start, end, langs_sel) <<<
with st.sidebar:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📤 Export Current View")
    exp_name = _safe_name(f"{center}_{view_type.replace(' ', '_')}_{start:%Y%m%d}_{end:%Y%m%d}.xlsx")
    if st.button("Build Excel", type="primary", use_container_width=True):
        try:
            xlsx_bytes = export_excel(
                view_type=view_type,
                center=center,
                languages_sel=langs_sel,
                start=start,
                end=end,
                records=records,
                roster=roster,
            )
            st.session_state["last_export_bytes"] = xlsx_bytes
            st.success("Excel generated.")
        except Exception as e:
            st.error(f"Export failed: {e}")

    if "last_export_bytes" in st.session_state:
        st.download_button(
            "⬇️ Download Excel",
            data=st.session_state["last_export_bytes"],
            file_name=exp_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

        if st.checkbox("Also upload to Drive (same folder as DB)"):
            if st.button("📤 Upload to Drive", use_container_width=True):
                try:
                    service = _get_drive_service()
                    file_id = _drive_find_file_id(service, exp_name, _current_drive_folder_id() or None)
                    up_id = _drive_upload_bytes(
                        service=service,
                        name=exp_name,
                        data=st.session_state["last_export_bytes"],
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        folder_id=_current_drive_folder_id() or None,
                        file_id=file_id,
                    )
                    st.success(f"Uploaded to Drive (file id: {up_id}).")
                except Exception as e:
                    st.error(f"Drive upload failed: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='card'>", unsafe_allow_html=True)

# Build data for views
req_shift = dedup_requested_split_pairs(pivot_requested(records, center, langs_sel, start, end))
ros_shift = pivot_roster_counts(roster, center, langs_sel, start, end)
all_dates = sorted(list(set(req_shift.columns) | set(ros_shift.columns)))
all_shifts = sorted(list(set(req_shift.index) | set(ros_shift.index)))

req_shift = req_shift.reindex(index=all_shifts, columns=all_dates, fill_value=0.0)
ros_shift = ros_shift.reindex(index=all_shifts, columns=all_dates, fill_value=0.0)
delt_shift = ros_shift - req_shift

for _df in (req_shift, ros_shift, delt_shift):
    _df.index.name = "Shift"

req_shift_total = add_total_row(req_shift)
ros_shift_total = add_total_row(ros_shift)
delt_shift_total = add_total_row(delt_shift)

# KPI row (in soft cards)
k1, k2, k3 = st.columns(3)
with k1:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("Requested", _pretty_int(req_shift.values.sum()))
    st.markdown("</div>", unsafe_allow_html=True)
with k2:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("Rostered", _pretty_int(ros_shift.values.sum()))
    st.markdown("</div>", unsafe_allow_html=True)
with k3:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("Delta", _pretty_int(delt_shift.values.sum()))
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# =========================
# PERSISTENT TABS + FULL RENDERERS
# =========================

def render_dashboard(view_type: str,
                     req_shift: pd.DataFrame, ros_shift: pd.DataFrame, delt_shift: pd.DataFrame,
                     req_shift_total: pd.DataFrame, ros_shift_total: pd.DataFrame, delt_shift_total: pd.DataFrame):
    st.subheader("📈 Dashboard")
    sub1, sub2, sub3 = st.columns(3)
    with sub1:
        st.markdown("<div class='card'><h4>Requested – Shift-wise</h4>", unsafe_allow_html=True)
        st.dataframe(req_shift_total, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with sub2:
        st.markdown("<div class='card'><h4>Rostered – Shift-wise</h4>", unsafe_allow_html=True)
        st.dataframe(ros_shift_total, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with sub3:
        st.markdown("<div class='card'><h4>Delta – Shift-wise</h4>", unsafe_allow_html=True)
        st.dataframe(delt_shift_total, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if view_type in ["Interval_Wise Delta View", "Overall_Delta View"]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
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
        st.markdown("</div>", unsafe_allow_html=True)

# ==== Adapter so the new Plotter can read/write from your current data ====
class DBAdapter:
    def __init__(self, records_df, roster_df):
        self.records = records_df
        self.roster = roster_df

    def centers(self, include_overall=False):
        cs = _centers_union(self.records, self.roster)
        return [c for c in cs if c != "Overall"] if not include_overall else cs

    def languages(self, center):
        return _languages_union(self.records, self.roster, center)

    def pivot_requested(self, center, langs, start, end):
        return pivot_requested(self.records, center, langs, start, end)

    def upsert_requested_cells(self, center, language, edits: dict):
        if not is_admin:
            st.error("You need admin role to save edits.")
            return 0
        if not edits:
            return 0
        rows = []
        for (shift, d), v in edits.items():
            rows.append({
                "Center": center,
                "Language": language,
                "Shift": shift,
                "Metric": "Requested",
                "Date": d,
                "Value": float(v),
            })
        df = pd.DataFrame(rows)

        if DUCKDB_FILE_NAME:
            local_db, file_id, err = _download_duckdb_rw(DUCKDB_FILE_NAME)
            if err or not local_db or not file_id:
                st.error(f"DuckDB issue: {err or 'File not found.'}")
                return 0
            return duckdb_upsert_records(local_db, file_id, DUCKDB_FILE_NAME, df)
        else:
            cur = load_parquet_from_drive(RECORDS_FILE)
            if cur.empty:
                save_parquet_to_drive(RECORDS_FILE, df)
                return len(df)
            keys = ["Center","Language","Shift","Date"]
            left = cur.merge(df[keys].drop_duplicates(), on=keys, how="left", indicator=True)
            keep = left["_merge"] == "left_only"
            base = cur[keep].copy()
            new_all = pd.concat([base, df], ignore_index=True)
            save_parquet_to_drive(RECORDS_FILE, new_all)
            return len(df)

# =========================
# PLOTTER (Desktop-like) — FULL REPLACEMENT
# =========================
import re
from datetime import date as _date

if "plot_grid" not in st.session_state:     st.session_state.plot_grid = None
if "plot_center" not in st.session_state:   st.session_state.plot_center = None
if "plot_lang" not in st.session_state:     st.session_state.plot_lang = None
if "plot_range" not in st.session_state:    st.session_state.plot_range = (None, None)
if "plot_dirty" not in st.session_state:    st.session_state.plot_dirty = False

_TIME_LIKE = re.compile(r"^\d{4}-\d{4}(?:/\d{4}-\d{4})*$")

def _parse_ddmm_header_to_date(lbl: str, fallback_year: int) -> _date | None:
    s = str(lbl).strip()
    for fmt in ("%d-%b-%Y", "%d-%b-%y", "%Y-%m-%d"):
        try:
            return pd.to_datetime(s, format=fmt).date()
        except Exception:
            pass
    try:
        return pd.to_datetime(f"{s}-{fallback_year}", format="%d-%b-%Y").date()
    except Exception:
        return None

def _ensure_total_row(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    has_total = df["Shift"].astype(str).str.lower().eq("total").any()
    if not has_total:
        new = df.copy()
        new.loc[len(new)] = ["Total"] + [0]*(df.shape[1]-1)
        return new
    return df

def _recalc_total(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    body = df[~df["Shift"].astype(str).str.lower().eq("total")]
    sums = body.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=0)
    out = df.copy()
    mask_total = out["Shift"].astype(str).str.lower().eq("total")
    out.loc[mask_total, out.columns[1:]] = sums.values
    return out

def _grid_to_requested_df(grid_df: pd.DataFrame, from_year: int) -> pd.DataFrame:
    if grid_df is None or grid_df.empty: return pd.DataFrame()
    cols = list(grid_df.columns)
    if not cols or cols[0] != "Shift": return pd.DataFrame()
    parsed = []
    for h in cols[1:]:
        d = _parse_ddmm_header_to_date(h, from_year)
        parsed.append(d)
    keep = [i+1 for i, d in enumerate(parsed) if d is not None]
    out_dates = [d for d in parsed if d is not None]
    if not keep: return pd.DataFrame()
    body = grid_df[~grid_df["Shift"].astype(str).str.lower().eq("total")].copy()
    body_rows = []
    idx_labels = []
    for _, r in body.iterrows():
        lbl = str(r["Shift"]).strip()
        if not lbl or not _TIME_LIKE.fullmatch(lbl):
            continue
        vals = []
        for j in keep:
            try:
                v = float(r.iloc[j])
            except Exception:
                v = 0.0
            vals.append(int(v) if float(v).is_integer() else float(v))
        body_rows.append(vals); idx_labels.append(lbl)
    if not body_rows: return pd.DataFrame()
    df = pd.DataFrame(body_rows, index=idx_labels, columns=out_dates)
    return df

def _plotter_load_grid(db, centre: str, lang: str, d_from: _date, d_to: _date) -> pd.DataFrame:
    req_df = db.pivot_requested(centre, [lang], d_from, d_to)
    req_df = dedup_requested_split_pairs(req_df)
    keep_cols = []
    for c in req_df.columns:
        cd = c.date() if hasattr(c, "date") else c
        if d_from <= cd <= d_to:
            keep_cols.append(c)
    keep_cols = sorted(keep_cols)
    req_df = req_df.reindex(columns=keep_cols, fill_value=0.0)
    headers = ["Shift"] + [pd.to_datetime(c).strftime("%d-%b") for c in keep_cols]
    grid = []
    for shift_label in req_df.index:
        row = [str(shift_label)]
        for c in keep_cols:
            v = req_df.at[shift_label, c]
            try:
                vf = float(v); row.append(int(vf) if vf.is_integer() else round(vf, 2))
            except Exception:
                row.append(0)
        grid.append(row)
    df = pd.DataFrame(grid, columns=headers)
    df = _ensure_total_row(df)
    df = _recalc_total(df)
    return df

def _plotter_interval_from_grid(grid_df: pd.DataFrame, from_year: int) -> pd.DataFrame:
    req = _grid_to_requested_df(grid_df, from_year)
    if req.empty:
        return pd.DataFrame(columns=["Interval"])
    req.index.name = "Shift"
    inter = transform_to_interval_view(req)
    inter = inter.reindex(index=range(24), fill_value=0.0)
    inter = add_total_row(inter)
    inter.index.name = "Interval"
    return inter

def render_plotter(db):
    st.subheader("🧮 Plotter (desktop workflow)")
    if not is_admin:
        st.info("You are in viewer mode. Editing is disabled.")
    c1, c2, c3, c4, c5 = st.columns([1.2, 1.2, 1, 1, 1.6])
    with c1:
        centers = db.centers(include_overall=False) or []
        centre = st.selectbox("Center", centers, index=0 if centers else None, key="plot_center_sel")
    with c2:
        langs = db.languages(centre) if centre else []
        lang = st.selectbox("Language", langs, index=0 if langs else None, key="plot_lang_sel")
    with c3:
        d_from = st.date_input("From", value=st.session_state.get("plot_from") or date.today(), key="plot_from")
    with c4:
        d_to = st.date_input("To", value=st.session_state.get("plot_to") or date.today(), key="plot_to")

    def _reload_grid():
        if not centre or not lang: return
        grid = _plotter_load_grid(db, centre, lang, d_from, d_to)
        st.session_state.plot_grid = grid
        st.session_state.plot_center = centre
        st.session_state.plot_lang = lang
        st.session_state.plot_range = (d_from, d_to)
        st.session_state.plot_dirty = False

    if (st.session_state.plot_grid is None or
        st.session_state.plot_center != centre or
        st.session_state.plot_lang != lang or
        st.session_state.plot_range != (d_from, d_to)):
        _reload_grid()

    with c5:
        b1, b2, b3, b4 = st.columns([1,1,1,1.2])
        if b1.button("Add Date"):
            if st.session_state.plot_grid is not None and not st.session_state.plot_grid.empty:
                headers = list(st.session_state.plot_grid.columns)
                base_year = d_from.year
                existing = set()
                for h in headers[1:]:
                    dd = _parse_ddmm_header_to_date(h, base_year)
                    if dd: existing.add(dd)
                wanted = pd.date_range(d_from, d_to, freq="D").date
                to_add = [d for d in wanted if d not in existing]
                if to_add:
                    g = st.session_state.plot_grid.copy()
                    for nh in [d.strftime("%d-%b") for d in to_add]:
                        if nh not in g.columns:
                            g[nh] = 0
                    head = ["Shift"]
                    pairs = []
                    for h in g.columns[1:]:
                        dd = _parse_ddmm_header_to_date(h, base_year)
                        if dd: pairs.append((h, dd))
                    pairs.sort(key=lambda x: x[1])
                    head += [h for h,_ in pairs]
                    g = g[head]
                    g = _ensure_total_row(g)
                    g = _recalc_total(g)
                    st.session_state.plot_grid = g
                    st.session_state.plot_dirty = True
                else:
                    st.info("No new dates to add in the selected range.")
            else:
                _reload_grid()
        b2.button("Manage Shifts", help="(Open your shifts screen in web app)")
        b3.button("Manage Languages", help="(Open your languages screen in web app)")
        if b4.button("Save", type="primary", help="Write non-zero Requested cells to DB", disabled=not is_admin):
            grid = st.session_state.plot_grid
            if grid is None or grid.empty or grid.shape[1] < 2:
                st.warning("Nothing to save.")
            else:
                req_df = _grid_to_requested_df(grid, d_from.year)
                edits = {}
                for sh, row in req_df.iterrows():
                    for d, v in row.items():
                        if float(v) != 0.0:
                            edits[(sh, d)] = float(v)
                if edits:
                    db.upsert_requested_cells(centre, lang, edits)
                    st.success(f"Saved {len(edits)} cells.")
                    st.session_state.plot_dirty = False
                else:
                    st.info("No non-zero changes to save.")
    st.caption("Shift-wise Requested (editable)")
    if st.session_state.plot_grid is None:
        st.info("Select Center, Language and date range to load.")
        return
    cfg = {"Shift": st.column_config.TextColumn("Shift", disabled=True)}
    edited = st.data_editor(
        st.session_state.plot_grid,
        num_rows="dynamic",
        use_container_width=True,
        column_config=cfg,
        key="plot_editor",
        disabled=not is_admin
    )
    if not edited.empty:
        if "Shift" in edited.columns:
            edited.loc[edited["Shift"].astype(str).str.lower() == "total", "Shift"] = "Total"
        mask_total = edited["Shift"].astype(str).str.lower() == "total"
        if mask_total.any():
            cols = [c for c in edited.columns if c != "Shift"]
            edited.loc[mask_total, cols] = 0
        edited = _ensure_total_row(edited)
        edited = _recalc_total(edited)
    st.session_state.plot_grid = edited
    st.session_state.plot_dirty = True

    inter = _plotter_interval_from_grid(edited, d_from.year)
    if inter.empty:
        st.info("No interval view for current grid.")
    else:
        st.caption("Interval-wise Requested (auto)")
        st.dataframe(inter, use_container_width=True)

    st.markdown(
        f"<div style='text-align:right;color:#888'>"
        f"{'Unsaved changes' if st.session_state.plot_dirty else 'Up to date'}"
        f"</div>",
        unsafe_allow_html=True
    )

# =========================
# ✨ NEW: Roster Edit Dialogs (Single/Bulk)
# =========================
@st.dialog("Edit / Update Roster")
def roster_edit_dialog(roster_df: pd.DataFrame):
    if not is_admin:
        st.error("You need admin role to edit roster.")
        return
    choice = st.radio("Mode", ["Single Edit", "Bulk Edit"], horizontal=True)
    st.markdown("---")

    if choice == "Single Edit":
        agent_id = st.text_input("Genesys Agent ID")
        # Date range limiter: max 7 days
        today = date.today()
        d_from = st.date_input("From date", value=today)
        d_to = st.date_input("To date", value=today)
        if (d_to - d_from).days > 6:
            st.warning("Maximum editable range is 7 days. Please reduce the range.")
            return
        if st.button("Load rows", use_container_width=True):
            dfr = roster_df.copy()
            if agent_id:
                dfr = dfr[dfr["AgentID"].astype(str) == agent_id.strip()]
            dfr = dfr[(dfr["Date"] >= d_from) & (dfr["Date"] <= d_to)]
            if dfr.empty:
                st.info("No rows found for the given Agent ID and date range.")
                return
            # Editable columns
            editable_cols = [c for c in ["Shift","WorkMode","Status"] if c in dfr.columns]
            show_cols = ["Date","AgentID"] + editable_cols + [c for c in dfr.columns if c not in ["Date","AgentID"] + editable_cols]
            dfr = dfr[show_cols]
            st.caption("Edit desired fields below, then click Save.")
            edited = st.data_editor(dfr, num_rows="fixed", use_container_width=True)
            if st.button("Save changes", type="primary", use_container_width=True):
                # Build patch rows: only keep changed rows & editable fields
                diffs = []
                for idx in range(len(dfr)):
                    before = dfr.iloc[idx]
                    after = edited.iloc[idx]
                    changed = {}
                    for col in editable_cols:
                        if str(before[col]) != str(after[col]):
                            changed[col] = after[col]
                    if changed:
                        changed["AgentID"] = after["AgentID"]
                        changed["Date"] = after["Date"]
                        diffs.append(changed)
                if not diffs:
                    st.info("No changes to save.")
                    return
                patch_df = pd.DataFrame(diffs)
                if DUCKDB_FILE_NAME:
                    local_db, file_id, err = _download_duckdb_rw(DUCKDB_FILE_NAME)
                    if err or not local_db or not file_id:
                        st.error(f"DuckDB issue: {err or 'File not found.'}")
                        return
                    n = duckdb_upsert_roster(local_db, file_id, DUCKDB_FILE_NAME, patch_df)
                    st.success(f"Saved {n} roster updates.")
                else:
                    # Parquet fallback: load, update rows in-memory, write back
                    base = load_parquet_from_drive(ROSTER_FILE)
                    if base.empty:
                        st.error("Roster store is empty; cannot patch.")
                        return
                    base = base.copy()
                    base["Date"] = pd.to_datetime(base["Date"]).dt.date
                    for _, r in patch_df.iterrows():
                        m = (base["AgentID"].astype(str) == str(r["AgentID"])) & (base["Date"] == r["Date"])
                        for col in editable_cols:
                            if col in r and pd.notna(r[col]):
                                base.loc[m, col] = r[col]
                    save_parquet_to_drive(ROSTER_FILE, base)
                    st.success(f"Saved {len(patch_df)} roster updates.")
                # refresh
                load_parquet_from_drive.clear(); load_from_duckdb.clear()
                st.rerun()

    else:  # Bulk Edit
        st.info("Upload a roster file for bulk update. (Validation rules to be added later.)")
        up = st.file_uploader("Upload roster (.xlsx)", type=["xlsx"], key="bulk_roster_up")
        if st.button("Validate & Stage", use_container_width=True, disabled=(not up)):
            # NOTE: Placeholder: accept & preview. We'll add validations when you share criteria.
            raw = read_roster_sheet(up.read())
            if raw.empty:
                st.error("No 'Roster' sheet found or no data.")
                return
            st.success(f"Loaded {len(raw)} rows. Validation pending (to be implemented).")
            st.dataframe(raw.head(50), use_container_width=True, height=300)
            st.caption("✅ This is only staged. Final write will be enabled after we add your validation rules.")
        st.caption("After validations are defined, we'll enable a 'Commit' button here.")

# ------- Persistent tab switcher -------
db = DBAdapter(records, roster)

tab_choice = st.radio(
    "View",
    ["Dashboard", "Plotter", "Roster"],
    horizontal=True,
    key="active_tab"
)

if tab_choice == "Dashboard":
    render_dashboard(
        view_type,
        req_shift, ros_shift, delt_shift,
        req_shift_total, ros_shift_total, delt_shift_total
    )

elif tab_choice == "Plotter":
    render_plotter(db)

else:
    # ✨ NEW: Roster with Date as first column (DD-MMM-YY)
    def _fmt_date(d):
        try:
            return pd.to_datetime(d).strftime("%d-%b-%y")
        except Exception:
            return d

    st.subheader("👥 Roster")
    # When Language changes, mark UI as busy
    def _roster_set_busy():
        st.session_state.busy_roster = True

    disabled = st.session_state.busy_roster
    r1, r2, r3, r4, r5 = st.columns([1,1,1,1.3,1.2])
    with r1:
        lobs = roster_lobs(roster, center, start, end)
        lob_sel = st.selectbox("LOB", ["(All)"] + lobs, disabled=disabled)
        lob = None if lob_sel in ("", "(All)") else lob_sel
    with r2:
        langs_lob = roster_languages_for_lob(roster, center, lob, start, end)
        rlang_sel = st.selectbox("Language", ["(All)"] + langs_lob, key="rlang_sel",
                                 on_change=_roster_set_busy, disabled=disabled)
        rlang = None if rlang_sel in ("", "(All)") else rlang_sel
    with r3:
        codes = roster_nonshift_codes(roster, center, start, end, lob, rlang)
        nonshift_sel = st.selectbox("Non-Shift", ["(All)"] + codes, disabled=disabled)
    with r4:
        shifts_list = roster_distinct_shifts(roster, center, start, end, lob, rlang)
        shift_disabled = disabled or (nonshift_sel not in ("", "(All)"))
        shift_sel = st.selectbox("Shift", ["(All)"] + shifts_list, disabled=shift_disabled)
    with r5:
        ids_text = st.text_input("Genesys IDs (comma/space)", disabled=disabled)
        ids = [x.strip() for x in re.findall(r"[A-Za-z0-9_]+", ids_text)] if ids_text else None

    with st.spinner("Loading roster…"):
        dfr = roster[roster["Date"].between(start, end)].copy()
        if center and center != "Overall":
            dfr = dfr[dfr["Center"] == center]
        if lob:
            dfr = dfr[dfr["LOB"] == lob]
        if rlang:
            dfr = dfr[dfr["Language"] == rlang]
        if ids:
            dfr = dfr[dfr["AgentID"].isin(ids)]

        if not dfr.empty and "Shift" in dfr.columns:
            dfr["Shift"] = dfr["Shift"].apply(lambda s: (_normalize_shift_label(s) or str(s).strip().upper()))

        if not dfr.empty:
            time_like = re.compile(r"^\d{4}-\d{4}(?:/\d{4}-\d{4})*$")
            single_pat = re.compile(r"^\d{4}-\d{4}$")
            if nonshift_sel not in ("", "(All)"):
                dfr = dfr[
                    (~dfr["Shift"].astype(str).str.match(time_like)) &
                    (dfr["Shift"].astype(str).str.upper() == nonshift_sel.upper())
                ]
            elif shift_sel not in ("", "(All)"):
                chosen_is_single = bool(single_pat.fullmatch(shift_sel))
                def keep_shift(s: str) -> bool:
                    s = str(s).strip()
                    if not time_like.fullmatch(s):
                        return False
                    if s == shift_sel:
                        return True
                    if chosen_is_single and "/" in s:
                        parts = [p.strip() for p in s.split("/") if p.strip()]
                        return shift_sel in parts
                    return False
                dfr = dfr[dfr["Shift"].apply(keep_shift)]

        # Reorder with Date first, formatted, then other columns
        if dfr.empty:
            long_df = pd.DataFrame(columns=["Date","AgentID","AgentName","TLName","Status","WorkMode","Center","Location","Language","SecondaryLanguage","LOB","FTPT","Shift"])
        else:
            dfr = dfr.copy()
            dfr["Date_str"] = dfr["Date"].apply(_fmt_date)
            static_cols = [c for c in [
                "AgentID","AgentName","TLName","Status","WorkMode","Center","Location",
                "Language","SecondaryLanguage","LOB","FTPT"
            ] if c in dfr.columns]
            long_cols = ["Date_str"] + static_cols + ["Shift"]
            long_df = dfr[long_cols].rename(columns={"Date_str":"Date"})

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.caption("Roster (Date-first, DD-MMM-YY)")
        st.dataframe(long_df, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.session_state.busy_roster = False

    # ✨ NEW: Edit/Update button (opens modal with Single/Bulk)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Edit / Update")
    st.caption("Single edit supports Agent ID + date range (max 7 days). Bulk edit supports file upload (validation TBD).")
    st.button("Edit / Update Roster", on_click=lambda: roster_edit_dialog(roster), type="primary", disabled=not is_admin)
    if not is_admin:
        st.info("You are in viewer mode; contact an admin for edit access.")
    st.markdown("</div>", unsafe_allow_html=True)
