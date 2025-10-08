# =========================
# Imports
# =========================
import io
import re
import json
import os
import tempfile
from datetime import date, timedelta, datetime
from typing import Optional, Tuple, List, Dict

import pandas as pd
import streamlit as st
import base64

# Google Drive API (service account)
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload

# DuckDB
import duckdb

# Auth deps (SSO)
from streamlit_oauth import OAuth2Component

# Password (no-bcrypt) helpers
import hashlib
import hmac

# =========================
# Page + Global Config
# =========================
st.set_page_config(page_title="Center Shiftwise Web Dashboard", page_icon="📊", layout="wide")

# --- UI state ---
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Dashboard"

# =========================
# Theme / Global Styles
# =========================
st.markdown("""
<style>
/* page bg */
.stApp {
    background: linear-gradient(170deg, #0f172a, #1e293b, #334155);
    background-size: 400% 400%;
    animation: gradientMove 20s ease infinite;
}
@keyframes gradientMove {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
/* cards & containers */
.card {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 18px;
    backdrop-filter: blur(10px) saturate(150%);
    -webkit-backdrop-filter: blur(10px) saturate(150%);
}
/* headers & text */
h1, h2, h3, h4, .stMarkdown p, .stMarkdown li { color: #f1f5f9 !important; }
h3 {
    border-bottom: 1px solid rgba(255,255,255,0.1);
    padding-bottom: 8px;
    margin-bottom: 12px;
}
/* Metric styling */
[data-testid="stMetric"] {
    background-color: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 10px;
    backdrop-filter: blur(10px);
}
[data-testid="stMetricDelta"], [data-testid="stMetricValue"] { color: #f1f5f9 !important; }
/* table tweaks */
[data-testid="stDataFrame"] div[role="grid"] {
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.2);
    background-color: rgba(15, 23, 42, 0.7);
}
[data-testid="stDataFrame"] .row-header, [data-testid="stDataFrame"] .col_heading {
    color: #94a3b8;
    background-color: transparent;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Secrets & Global Config
# =========================
SA_INFO = st.secrets.get("gcp_service_account", {})
DRIVE_FOLDER_ID = st.secrets.get("DRIVE_FOLDER_ID", "").strip()
DUCKDB_FILE_NAME = st.secrets.get("DUCKDB_FILE_NAME", "").strip() or "cmb_delta.duckdb"
DUCKDB_FILE_ID = st.secrets.get("DUCKDB_FILE_ID", "").strip()
ACL_FILE_NAME = st.secrets.get("ACL_FILE_NAME", "center_acl.json")
SHIFT_CODES_FILE_NAME = st.secrets.get("SHIFT_CODES_FILE_NAME", "shift_codes.json")


# =========================
# Password & Hashing Utilities
# =========================
def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_local_password(entered_password: str, salt_hex: str, stored_sha256_hex: str) -> bool:
    if not salt_hex or not stored_sha256_hex: return False
    calc = _sha256_hex(salt_hex + entered_password).lower()
    return hmac.compare_digest(calc, stored_sha256_hex.lower())


# =========================
# Authentication & Access Control (ACL)
# =========================
def _get_auth_config():
    cfg = st.secrets.get("auth", {})
    sso_roles = cfg.get("sso_roles", {})
    google = st.secrets.get("oauth", {}).get("google", {})
    return {
        "password_enabled": bool(cfg.get("enable_password_login", True)),
        "sso_enabled": bool(cfg.get("enable_sso_login", True)),
        "password_users": cfg.get("password_users", []),
        "admin_domains": set(sso_roles.get("admin_domains", [])),
        "viewer_domains": set(sso_roles.get("viewer_domains", [])),
        "google": google,
    }


def _sso_login_google(google_cfg: dict, admin_domains: set[str], viewer_domains: set[str]) -> dict | None:
    import base64, json, requests
    st.markdown("### 🔓 Login with Google (SSO)")
    client_id = (google_cfg.get("client_id") or "").strip()
    client_secret = (google_cfg.get("client_secret") or "").strip()
    authorize_url = (google_cfg.get("authorize_url") or "").strip()
    token_url = (google_cfg.get("token_url") or "").strip()
    user_info_url = (google_cfg.get("user_info_url") or "").strip()
    redirect_uri = (google_cfg.get("redirect_uri") or google_cfg.get("redirect_url") or "").strip()

    if not all([client_id, client_secret, authorize_url, token_url, user_info_url, redirect_uri]):
        st.error("SSO is not fully configured. Check st.secrets['oauth']['google']")
        return None
    st.session_state.setdefault("_oauth_redirect_uri", redirect_uri)
    try:
        oauth2 = OAuth2Component(client_id, client_secret, authorize_url, token_url, token_url,
                                 st.session_state["_oauth_redirect_uri"])
    except Exception as e:
        st.error(f"OAuth2Component init failed: {e}");
        return None

    def _decode_jwt_payload(id_token: str) -> dict:
        try:
            parts = id_token.split(".");
            pad = lambda s: s + "=" * (-len(s) % 4)
            return json.loads(base64.urlsafe_b64decode(pad(parts[1])).decode("utf-8"))
        except Exception:
            return {}

    def _role(email: str) -> str:
        dom = (email.split("@", 1)[-1] or "").lower().strip()
        if any(d == "*" or d == dom for d in admin_domains): return "admin"
        if any(d == "*" or d == dom for d in viewer_domains): return "viewer"
        return "viewer"

    try:
        result = oauth2.authorize_button("Sign in with Google", st.session_state["_oauth_redirect_uri"],
                                         "openid email profile", key="google_oauth_btn", use_container_width=True,
                                         extras_params={"prompt": "consent", "access_type": "offline",
                                                        "response_type": "code"})
    except Exception as e:
        st.error(f"authorize_button failed: {e}");
        return None
    if not result: return None
    token = result.get("token", result if isinstance(result, dict) else {}) or {}
    email = name = None
    access_token, id_token = token.get("access_token"), token.get("id_token")
    if access_token:
        try:
            r = requests.get(user_info_url, headers={"Authorization": f"Bearer {access_token}"}, timeout=10)
            if r.ok:
                info = r.json() if r.content else {}
                email = (info.get("email") or "").strip().lower()
                name = (info.get(
                    "name") or f"{info.get('given_name', '')} {info.get('family_name', '')}").strip() or email or "User"
        except Exception as e:
            st.info(f"Note: userinfo fetch failed, using id_token if present ({e})")
    if not email and id_token:
        payload = _decode_jwt_payload(id_token)
        email = (payload.get("email") or "").strip().lower()
        name = (payload.get(
            "name") or f"{payload.get('given_name', '')} {payload.get('family_name', '')}").strip() or email or "User"
    if not email:
        st.error("SSO error: Could not obtain user email from Google.");
        return None
    try:
        st.query_params.clear()
    except Exception:
        pass
    return {"name": name, "email": email, "role": _role(email), "auth": "sso"}


def _password_login_form(pusers: list[dict]) -> dict | None:
    st.markdown("### 🔑 Login with Username & Password")
    with st.form("pw_login"):
        u = st.text_input("Username", key="auth_username")
        p = st.text_input("Password", type="password", key="auth_password")
        submitted = st.form_submit_button("Login")
    if not submitted: return None
    if not u or not p: st.warning("Enter both username and password."); return None
    for rec in pusers:
        if rec.get("username", "").strip().lower() == u.strip().lower():
            if verify_local_password(p, rec.get("password_salt", ""), rec.get("password_sha256", "")):
                return {"name": rec.get("name") or u, "email": rec.get("email") or f"{u}@local",
                        "role": rec.get("role", "viewer"), "auth": "password"}
            break
    st.error("Invalid username or password.");
    return None


def _render_auth_gate():
    if "user" in st.session_state and st.session_state["user"]:
        return st.session_state["user"]
    cfg = _get_auth_config()
    tabs_to_show = []
    if cfg["password_enabled"]: tabs_to_show.append("User/Password")
    if cfg["sso_enabled"]: tabs_to_show.append("SSO (Google)")
    st.markdown('<div class="card" style="max-width: 500px; margin: 4rem auto;">', unsafe_allow_html=True)
    st.subheader("Welcome — Please Sign In")
    tabs = st.tabs(tabs_to_show or ["Sign-in disabled"])
    user = None
    if cfg["password_enabled"]:
        with tabs[0]: user = _password_login_form(cfg["password_users"])
    if cfg["sso_enabled"]:
        with tabs[-1]: user = _sso_login_google(cfg["google"], cfg["admin_domains"], cfg["viewer_domains"]) or user
    st.markdown('</div>', unsafe_allow_html=True)
    if user:
        st.session_state["user"] = user
        st.rerun()
    st.stop()


# =========================
# Google Drive Utilities
# =========================
@st.cache_resource
def _get_drive_service():
    if not SA_INFO: st.error("Missing `gcp_service_account` in Streamlit secrets."); st.stop()
    creds = service_account.Credentials.from_service_account_info(SA_INFO,
                                                                  scopes=['https://www.googleapis.com/auth/drive'])
    return build('drive', 'v3', credentials=creds, cache_discovery=False)


def _drive_find_file_id(service, name: str, folder_id: Optional[str]) -> Optional[str]:
    q = f"name = '{name}' and trashed = false"
    if folder_id: q += f" and '{folder_id}' in parents"
    resp = service.files().list(q=q, fields="files(id,name,mimeType,shortcutDetails)", pageSize=1,
                                supportsAllDrives=True, includeItemsFromAllDrives=True).execute()
    files = resp.get("files", [])
    if not files: return None
    file_obj = files[0]
    if file_obj.get("mimeType") == "application/vnd.google-apps.shortcut":
        return file_obj.get("shortcutDetails", {}).get("targetId")
    return file_obj["id"]


def _drive_download_bytes(service, file_id: str) -> bytes:
    req = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, req)
    done = False
    while not done: _, done = downloader.next_chunk()
    return buf.getvalue()


def _drive_upload_bytes(service, name: str, data: bytes, mime: str, folder_id: Optional[str],
                        file_id: Optional[str] = None) -> str:
    media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mime, resumable=True)
    if file_id:
        file = service.files().update(fileId=file_id, media_body=media, supportsAllDrives=True).execute()
    else:
        meta = {"name": name, "parents": [folder_id]} if folder_id else {"name": name}
        file = service.files().create(body=meta, media_body=media, fields="id", supportsAllDrives=True).execute()
    return file["id"]


def _drive_list_folder(service, folder_id: str) -> list[dict]:
    items, page_token = [], None
    while True:
        resp = service.files().list(q=f"'{folder_id}' in parents and trashed = false",
                                    fields="nextPageToken, files(id,name,mimeType,shortcutDetails)", pageSize=200,
                                    pageToken=page_token, supportsAllDrives=True,
                                    includeItemsFromAllDrives=True).execute()
        for f in resp.get("files", []):
            if f.get("mimeType") == "application/vnd.google-apps.shortcut":
                target_id = f.get("shortcutDetails", {}).get("targetId")
                if target_id:
                    try:
                        target_file = service.files().get(fileId=target_id, fields="id,name,mimeType",
                                                          supportsAllDrives=True).execute()
                        items.append(target_file)
                    except:
                        pass
            else:
                items.append(f)
        page_token = resp.get("nextPageToken")
        if not page_token: break
    return items


# =========================
# Database Management
# =========================
def _ensure_duckdb_schema(con: duckdb.DuckDBPyConnection):
    con.execute(
        "CREATE TABLE IF NOT EXISTS records (Center VARCHAR, Language VARCHAR, Shift VARCHAR, Metric VARCHAR, Date DATE, Value DOUBLE);")
    con.execute(
        "CREATE TABLE IF NOT EXISTS roster_long (AgentID VARCHAR, EmpID VARCHAR, AgentName VARCHAR, TLName VARCHAR, Status VARCHAR, WorkMode VARCHAR, Center VARCHAR, Location VARCHAR, Language VARCHAR, SecondaryLanguage VARCHAR, LOB VARCHAR, FTPT VARCHAR, BaseShift VARCHAR, Date DATE, Shift VARCHAR);")


@st.cache_resource(ttl=3600)
def get_duckdb_connection(db_name: str) -> Tuple[Optional[duckdb.DuckDBPyConnection], str, str, Optional[str]]:
    try:
        service = _get_drive_service()
        file_id = DUCKDB_FILE_ID or _drive_find_file_id(service, db_name, DRIVE_FOLDER_ID)
        local_dir = os.path.join(tempfile.gettempdir(), "streamlit_db_cache")
        os.makedirs(local_dir, exist_ok=True)
        if not file_id:
            st.warning(f"Database '{db_name}' not found on Drive. Creating a new one.")
            local_path = os.path.join(local_dir, f"new-{db_name}")
            con = duckdb.connect(local_path, read_only=False)
            _ensure_duckdb_schema(con);
            con.close()
            with open(local_path, "rb") as f:
                new_file_id = _drive_upload_bytes(service, db_name, f.read(), "application/octet-stream",
                                                  DRIVE_FOLDER_ID, None)
            st.info(f"New DB created. Set DUCKDB_FILE_ID='{new_file_id}' in secrets for faster startups.")
            file_id = new_file_id
        local_path = os.path.join(local_dir, file_id)
        if not os.path.exists(local_path):
            with st.spinner("Downloading database from Google Drive..."):
                raw = _drive_download_bytes(service, file_id)
                with open(local_path, "wb") as f: f.write(raw)
        con = duckdb.connect(local_path, read_only=False)
        return con, local_path, file_id, None
    except Exception as e:
        return None, "", "", str(e)


def _upload_duckdb_back(local_path: str, file_id: str, name: str):
    with st.spinner("Saving changes to Google Drive..."):
        try:
            service = _get_drive_service()
            with open(local_path, "rb") as f:
                data = f.read()
            _drive_upload_bytes(service, name, data, "application/octet-stream", DRIVE_FOLDER_ID, file_id)
            get_duckdb_connection.clear()
            st.toast("✅ Changes saved to cloud!")
        except Exception as e:
            st.error(f"Failed to save to Google Drive: {e}")


def duckdb_upsert_records(con: duckdb.DuckDBPyConnection, local_path: str, file_id: str, name_in_drive: str,
                          new_rows: pd.DataFrame):
    if new_rows.empty: return
    new_rows["Date"] = pd.to_datetime(new_rows["Date"]).dt.date
    con.register("new_records_df", new_rows)
    con.execute("""
        MERGE INTO records AS t USING new_records_df AS s
        ON t.Center = s.Center AND t.Language = s.Language AND t.Shift = s.Shift AND t.Date = s.Date
        WHEN MATCHED THEN UPDATE SET Value = s.Value
        WHEN NOT MATCHED THEN INSERT (Center, Language, Shift, Metric, Date, Value)
        VALUES (s.Center, s.Language, s.Shift, 'Requested', s.Date, s.Value);
    """)
    con.commit()
    _upload_duckdb_back(local_path, file_id, name_in_drive)


def duckdb_replace_roster_for_center(con: duckdb.DuckDBPyConnection, local_path: str, file_id: str, name_in_drive: str,
                                     roster_df: pd.DataFrame, center: str):
    if roster_df.empty: return
    roster_df["Center"] = center
    roster_df["Date"] = pd.to_datetime(roster_df["Date"]).dt.date
    con.execute("DELETE FROM roster_long WHERE Center = ?", [center])
    con.register("new_center_roster_df", roster_df)
    con.execute("INSERT INTO roster_long SELECT * FROM new_center_roster_df")
    con.commit()
    _upload_duckdb_back(local_path, file_id, name_in_drive)


# =========================
# ACL & Shift Code Management
# =========================
@st.cache_data(ttl=300)
def _load_json_from_drive_cached(file_name: str, default_val):
    try:
        service = _get_drive_service()
        file_id = _drive_find_file_id(service, file_name, DRIVE_FOLDER_ID)
        if not file_id: return default_val
        raw = _drive_download_bytes(service, file_id)
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return default_val


def _save_json_to_drive(file_name: str, data):
    try:
        service = _get_drive_service()
        json_data = json.dumps(data, indent=2).encode("utf-8")
        file_id = _drive_find_file_id(service, file_name, DRIVE_FOLDER_ID)
        _drive_upload_bytes(service, file_name, json_data, "application/json", DRIVE_FOLDER_ID, file_id)
        _load_json_from_drive_cached.clear()
    except Exception as e:
        st.error(f"Failed saving {file_name}: {e}")


def get_acl(): return _load_json_from_drive_cached(ACL_FILE_NAME, {})


def save_acl(acl: dict): _save_json_to_drive(ACL_FILE_NAME, acl)


def get_custom_shift_codes() -> set[str]:
    codes = _load_json_from_drive_cached(SHIFT_CODES_FILE_NAME, [])
    return {str(c).strip().upper() for c in codes if c}


def save_shift_codes(codes: list[str]):
    _save_json_to_drive(SHIFT_CODES_FILE_NAME, sorted({str(c).strip().upper() for c in codes if c}))


def resolve_allowed_centers_from_email(email: str, all_centers: List[str], acl: dict) -> tuple[list[str], bool]:
    if not email: return [], False
    e_lower = email.lower()
    admin_domains = st.secrets.get("auth", {}).get("sso_roles", {}).get("admin_domains", [])
    if any(domain in e_lower for domain in admin_domains):
        return all_centers, True
    allowed = {c for c in all_centers if e_lower in {str(ed).lower() for ed in acl.get(c, {}).get("editors", [])}}
    return sorted(list(allowed)), False


def user_can_edit_center(role: str, center: str, allowed_centers: list[str], has_full_access: bool) -> bool:
    if role != "admin": return False
    acl = get_acl()
    if not acl.get(center, {}).get("allow_edit", True): return False
    if has_full_access: return True
    return center in set(allowed_centers)


# =========================
# Parsing & Data Transformation
# =========================
SHIFT_SPLIT = re.compile(r"\s*/\s*")
WOCL_VARIANTS = re.compile(r"^\s*W\s*O\s*[\+&/ ]\s*C\s*L\s*$", re.I)
TIME_RANGE_RE = re.compile(
    r"""^\s*(?:(?:\d{1,2}[:.]?\d{2})\s*-\s*(?:\d{1,2}[:.]?\d{2}))(?:\s*/\s*(?:\d{1,2}[:.]?\d{2})\s*-\s*(?:\d{1,2}[:.]?\d{2}))*\s*$""",
    re.X)


def _is_off_code(s: str) -> Optional[str]:
    t = re.sub(r"\s+", "", str(s)).upper()
    if t in {"WO", "OFF"}: return "WO"
    if t in {"CL", "CO", "COMPOFF", "COMP-OFF"}: return "CL"
    if WOCL_VARIANTS.fullmatch(str(s)): return "WO+CL"
    if t in get_custom_shift_codes(): return t
    return None


def _normalize_shift_label(s: str) -> Optional[str]:
    off_code = _is_off_code(s)
    if off_code: return off_code
    txt = str(s).strip()
    if txt.upper() in get_custom_shift_codes(): return txt.upper()
    if not TIME_RANGE_RE.fullmatch(txt): return None
    norm_parts = []
    for part in SHIFT_SPLIT.split(txt):
        m = re.fullmatch(r"(\d{1,2})[:.]?(\d{2})\s*-\s*(\d{1,2})[:.]?(\d{2})", part.strip())
        if not m: return None
        sh, sm, eh, em = map(int, m.groups())
        norm_parts.append(f"{sh:02d}{sm:02d}-{eh:02d}{em:02d}")
    return "/".join(norm_parts)


def read_roster_sheet(file_bytes: bytes) -> pd.DataFrame:
    try:
        df = pd.read_excel(io.BytesIO(file_bytes), sheet_name="Roster")
        for col in df.columns:
            key = str(col).strip().lower().replace("_", " ")
            if key in ("agent name", "agentname", "tl name", "tlname"):
                df[col] = df[col].astype(str).apply(lambda x: " ".join(p.capitalize() for p in str(x).strip().split()))
        return df
    except Exception as e:
        st.warning(f"Could not read 'Roster' sheet: {e}")
        return pd.DataFrame()


def unpivot_roster_wide(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    date_cols, static_cols = [], []
    for col in df.columns:
        try:
            # A column is a date if it can be parsed as a date, but isn't just a number
            if isinstance(col, (datetime, date)):
                date_cols.append(col)
                continue
            if isinstance(col, (int, float)):
                static_cols.append(col)
                continue
            pd.to_datetime(str(col), errors='raise')
            date_cols.append(col)
        except (ValueError, TypeError):
            static_cols.append(col)

    if not date_cols:
        st.error("Roster format error: No date-like columns (e.g., '2025-10-08', '08-Oct') found in the header.")
        return pd.DataFrame()

    melted_df = pd.melt(df, id_vars=static_cols, value_vars=date_cols, var_name="Date", value_name="Shift").dropna(
        subset=["Shift"])
    melted_df['Date'] = pd.to_datetime(melted_df['Date'])
    return melted_df


def parse_workbook_to_df(file_bytes: bytes) -> pd.DataFrame:
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    EXCLUDED = {"settings", "roster", "overall"}
    all_dfs = []
    for sheet_name in xls.sheet_names:
        if sheet_name.lower() in EXCLUDED: continue
        try:
            df = pd.read_excel(xls, sheet_name, header=None)
            if df.empty: continue

            shift_loc = None
            for r, row in df.iterrows():
                for c, cell in enumerate(row):
                    if isinstance(cell, str) and "shift" in cell.lower():
                        shift_loc = (r, c);
                        break
                if shift_loc: break
            if not shift_loc: continue

            shift_r, shift_c = shift_loc
            date_cols = {}
            for c in range(shift_c + 1, df.shape[1]):
                try:
                    date_val = pd.to_datetime(df.iloc[shift_r, c]).date()
                    date_cols[c] = date_val
                except (ValueError, TypeError):
                    continue
            if not date_cols: continue

            language = sheet_name
            for r in range(shift_r - 1, -1, -1):
                cell = df.iloc[r, shift_c]
                if isinstance(cell, str) and cell.strip():
                    language = cell.strip();
                    break

            sheet_data = []
            for r in range(shift_r + 1, df.shape[0]):
                shift_label = str(df.iloc[r, shift_c]).strip()
                norm_shift = _normalize_shift_label(shift_label)
                if not norm_shift: continue
                for c, date_val in date_cols.items():
                    value = pd.to_numeric(df.iloc[r, c], errors='coerce')
                    if pd.notna(value) and value > 0:
                        sheet_data.append([sheet_name, language, norm_shift, 'Requested', date_val, value])
            all_dfs.append(pd.DataFrame(sheet_data, columns=['Center', 'Language', 'Shift', 'Metric', 'Date', 'Value']))
        except Exception as e:
            st.warning(f"Could not parse sheet '{sheet_name}': {e}")

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


def _get_shift_duration(shift_label):
    if not isinstance(shift_label, str) or "/" in shift_label: return 0
    m = re.fullmatch(r"(\d{2})(\d{2})-(\d{2})(\d{2})", shift_label)
    if not m: return 0
    sh, sm, eh, em = map(int, m.groups())
    start = sh * 60 + sm;
    end = eh * 60 + em
    return (end - start) if end > start else (1440 - start + end)


def dedup_requested_split_pairs(pivot_df: pd.DataFrame) -> pd.DataFrame:
    if pivot_df.empty: return pivot_df
    df = pivot_df.copy()
    for col in df.columns:
        five_hour_shifts = {idx for idx in df.index if _get_shift_duration(idx) == 300}
        four_hour_shifts = {idx for idx in df.index if _get_shift_duration(idx) == 240}
        vals_in_5h = df.loc[list(five_hour_shifts), col].value_counts()
        for val, count in vals_in_5h.items():
            if val > 0 and count > 0:
                matching_4h = df.loc[list(four_hour_shifts), col] == val
                if matching_4h.sum() >= count:
                    indices_to_zero = matching_4h[matching_4h].index[:count]
                    df.loc[indices_to_zero, col] = 0
    return df


def transform_to_interval_view(shift_df: pd.DataFrame) -> pd.DataFrame:
    if shift_df.empty: return pd.DataFrame(index=range(24))
    interval_df = pd.DataFrame(0.0, index=range(24), columns=pd.to_datetime(shift_df.columns).date)
    for shift_label, row in shift_df.iterrows():
        if not isinstance(shift_label, str): continue
        for part in shift_label.split("/"):
            m = re.fullmatch(r"(\d{2})(\d{2})-(\d{2})(\d{2})", part.strip())
            if not m: continue
            sh, sm, eh, em = map(int, m.groups())
            start_min, end_min = sh * 60 + sm, eh * 60 + em
            for day, count in row.items():
                if count == 0: continue
                day = pd.to_datetime(day).date()
                if end_min > start_min:
                    for h in range(sh, eh + 1):
                        overlap = max(0, min((h + 1) * 60, end_min) - max(h * 60, start_min))
                        if overlap > 0 and h < 24: interval_df.loc[h, day] += count * (overlap / 60.0)
                else:  # Overnight
                    for h in range(sh, 24):
                        overlap = max(0, min((h + 1) * 60, 1440) - max(h * 60, start_min))
                        if overlap > 0: interval_df.loc[h, day] += count * (overlap / 60.0)
                    next_day = day + timedelta(days=1)
                    if next_day not in interval_df.columns: interval_df[next_day] = 0.0
                    for h in range(0, eh + 1):
                        overlap = max(0, min((h + 1) * 60, end_min) - max(h * 60, 0))
                        if overlap > 0: interval_df.loc[h, next_day] += count * (overlap / 60.0)
    return interval_df.sort_index().sort_index(axis=1)


# =========================
# SQL-Based Query Utilities
# =========================
def _execute_query(con, query, params=None) -> pd.DataFrame:
    try:
        return con.execute(query, params or []).df()
    except Exception:
        return pd.DataFrame()


def get_all_centers(con) -> List[str]:
    df = _execute_query(con, "SELECT DISTINCT Center FROM records UNION SELECT DISTINCT Center FROM roster_long")
    return ["Overall"] + sorted([c for c in df['Center'].dropna().tolist() if c])


def get_languages_for_center(con, center: Optional[str]) -> List[str]:
    params = []
    where_clause = ""
    if center and center != "Overall":
        where_clause = "WHERE Center = ?"
        params.append(center)
    df = _execute_query(con,
                        f"SELECT DISTINCT Language FROM records {where_clause} UNION SELECT DISTINCT Language FROM roster_long {where_clause}",
                        params + params)
    return sorted([lang for lang in df['Language'].dropna().tolist() if lang])


def get_date_range(con, center: Optional[str]) -> Tuple[date, date]:
    params = []
    where_clause = ""
    if center and center != "Overall":
        where_clause = "WHERE Center = ?"
        params.append(center)
    q = f"SELECT min(Date) as min_date, max(Date) as max_date FROM (SELECT Date FROM records {where_clause} UNION ALL SELECT Date FROM roster_long {where_clause})"
    df = _execute_query(con, q, params + params)
    if df.empty or df['min_date'].iloc[0] is None: return date.today(), date.today()
    return df['min_date'].iloc[0], df['max_date'].iloc[0]


def get_kpi_metrics(con, center, langs, start, end):
    if not langs: return 0, 0, 0
    params = [start, end] + langs
    where_clause = f"WHERE Date BETWEEN ? AND ? AND Language IN ({','.join('?' for _ in langs)})"
    if center and center != "Overall":
        where_clause += " AND Center = ?"
        params.append(center)
    req_val = _execute_query(con, f"SELECT SUM(Value) FROM records {where_clause}", params).iloc[0, 0] or 0
    ros_val = \
    _execute_query(con, f"SELECT COUNT(*) FROM roster_long {where_clause} AND Shift NOT IN ('WO', 'CL', 'WO+CL')",
                   params).iloc[0, 0] or 0
    return int(req_val), int(ros_val), int(ros_val - req_val)


def pivot_requested(con, center, langs, start, end) -> pd.DataFrame:
    if not langs: return pd.DataFrame()
    params = [start, end] + langs
    q = f"SELECT Shift, Date, SUM(Value) as Value FROM records WHERE Date BETWEEN ? AND ? AND Language IN ({','.join('?' for _ in langs)})"
    if center != 'Overall': q += " AND Center = ?"; params.append(center)
    q += " GROUP BY Shift, Date"
    df = _execute_query(con, q, params)
    if df.empty: return pd.DataFrame()
    pivot_df = df.pivot(index="Shift", columns="Date", values="Value").fillna(0.0)
    return pivot_df.reindex(sorted(pivot_df.columns), axis=1)


def pivot_roster_counts(con, center, langs, start, end) -> pd.DataFrame:
    if not langs: return pd.DataFrame()
    params = [start, end] + langs
    q = f"SELECT Shift, Date, COUNT(*) as Count FROM roster_long WHERE Date BETWEEN ? AND ? AND Language IN ({','.join('?' for _ in langs)})"
    if center != 'Overall': q += " AND Center = ?"; params.append(center)
    q += " GROUP BY Shift, Date"
    df = _execute_query(con, q, params)
    if df.empty: return pd.DataFrame()
    df['Shift'] = df['Shift'].apply(lambda s: _normalize_shift_label(s) or str(s).strip().upper())
    pivot_df = df.pivot_table(index="Shift", columns="Date", values="Count", aggfunc='sum').fillna(0.0)
    return pivot_df.reindex(sorted(pivot_df.columns), axis=1)


def get_roster_data_long(con, center, start, end, filters: dict) -> pd.DataFrame:
    params = [start, end]
    query = "SELECT Date, AgentID, AgentName, TLName, Status, WorkMode, Center, Language, LOB, FTPT, Shift FROM roster_long WHERE Date BETWEEN ? AND ?"
    if center != "Overall":
        query += " AND Center = ?"
        params.append(center)
    for key, value in filters.items():
        if value:
            query += f" AND {key} = ?"
            params.append(value)
    return _execute_query(con, query, params)


# =========================
# UI Renderers
# =========================
def render_dashboard(con, view_type, center, langs, start, end):
    with st.spinner("Loading dashboard data..."):
        req_s = pivot_requested(con, center, langs, start, end)
        ros_s = pivot_roster_counts(con, center, langs, start, end)
    all_dates = sorted(list(set(req_s.columns) | set(ros_s.columns)))
    all_shifts = sorted(list(set(req_s.index) | set(ros_s.index)))
    req_s = req_s.reindex(index=all_shifts, columns=all_dates, fill_value=0)
    ros_s = ros_s.reindex(index=all_shifts, columns=all_dates, fill_value=0)
    delta_s = ros_s - req_s

    def _render_shiftwise():
        st.subheader("📈 Shift-wise View")
        c1, c2, c3 = st.columns(3)
        with c1: st.markdown("<div class='card'><h4>Requested</h4>", unsafe_allow_html=True); st.dataframe(
            req_s.style.format('{:.0f}')); st.markdown("</div>", unsafe_allow_html=True)
        with c2: st.markdown("<div class='card'><h4>Rostered</h4>", unsafe_allow_html=True); st.dataframe(
            ros_s.style.format('{:.0f}')); st.markdown("</div>", unsafe_allow_html=True)
        with c3: st.markdown("<div class='card'><h4>Delta</h4>", unsafe_allow_html=True); st.dataframe(
            delta_s.style.format('{:.0f}')); st.markdown("</div>", unsafe_allow_html=True)

    def _render_intervalwise():
        st.subheader("⏱️ Interval-wise View")
        with st.spinner("Calculating interval data..."):
            req_i = transform_to_interval_view(req_s)
            ros_i = transform_to_interval_view(ros_s)
            all_i_dates = sorted(list(set(req_i.columns) | set(ros_i.columns)))
            req_i = req_i.reindex(columns=all_i_dates, fill_value=0)
            ros_i = ros_i.reindex(columns=all_i_dates, fill_value=0)
            delta_i = ros_i - req_i
        c1, c2, c3 = st.columns(3)
        with c1: st.markdown("<div class='card'><h4>Requested</h4>", unsafe_allow_html=True); st.dataframe(
            req_i.style.format('{:.1f}')); st.markdown("</div>", unsafe_allow_html=True)
        with c2: st.markdown("<div class='card'><h4>Rostered</h4>", unsafe_allow_html=True); st.dataframe(
            ros_i.style.format('{:.1f}')); st.markdown("</div>", unsafe_allow_html=True)
        with c3: st.markdown("<div class='card'><h4>Delta</h4>", unsafe_allow_html=True); st.dataframe(
            delta_i.style.format('{:.1f}')); st.markdown("</div>", unsafe_allow_html=True)

    if "Shift" in view_type: _render_shiftwise()
    if "Interval" in view_type: _render_intervalwise()
    if "Overall" in view_type: _render_shiftwise(); st.divider(); _render_intervalwise()


def render_plotter(con, local_path, file_id, db_name, can_edit):
    st.subheader("🧮 Plotter (Edit 'Requested' Numbers)")
    c1, c2, c3, c4 = st.columns([1.5, 1.5, 1, 1])
    with c1:
        center = st.selectbox("Center", [c for c in get_all_centers(con) if c != "Overall"], key="plot_center")
    with c2:
        lang = st.selectbox("Language", get_languages_for_center(con, center), key="plot_lang")
    with c3:
        d_from = st.date_input("From", date.today(), key="plot_from")
    with c4:
        d_to = st.date_input("To", date.today() + timedelta(days=6), key="plot_to")

    if center and lang:
        with st.spinner("Loading data for plotter..."):
            req_df = pivot_requested(con, center, [lang], d_from, d_to)
        req_df = dedup_requested_split_pairs(req_df)

        # Convert to editable format
        req_df = req_df.reset_index()
        req_df = req_df.rename(columns={"Shift": "Shift_Label"})

        edited_df = st.data_editor(req_df, num_rows="dynamic", use_container_width=True, disabled=not can_edit,
                                   key=f"editor_{center}_{lang}")

        if st.button("💾 Save Changes", type="primary", disabled=not can_edit):
            if edited_df.equals(req_df):
                st.toast("No changes to save.")
            else:
                out_long = edited_df.melt(id_vars="Shift_Label", var_name="Date", value_name="Value")
                out_long = out_long[pd.to_numeric(out_long['Value'], errors='coerce').notna()]
                out_long = out_long.rename(columns={"Shift_Label": "Shift"})
                out_long["Center"] = center
                out_long["Language"] = lang
                out_long["Metric"] = "Requested"
                out_long = out_long[["Center", "Language", "Shift", "Metric", "Date", "Value"]]

                # Filter only for rows that actually changed
                # This requires a slightly more complex merge logic, but for now we save the visible range
                duckdb_upsert_records(con, local_path, file_id, db_name, out_long)
                st.success("Changes saved!")


def render_roster(con, local_path, file_id, db_name, center, start, end, can_edit):
    st.subheader("👥 Roster View & Edit")
    filters = {}
    c1, c2, c3 = st.columns(3)
    with c1:
        filters['LOB'] = st.selectbox("Filter by LOB", [""] + _execute_query(con,
                                                                             f"SELECT DISTINCT LOB FROM roster_long WHERE Center = '{center}'").LOB.tolist())
    with c2:
        filters['Language'] = st.selectbox("Filter by Language", [""] + get_languages_for_center(con, center))
    with c3:
        filters['AgentID'] = st.text_input("Filter by Agent ID")

    with st.spinner("Fetching roster data..."):
        df = get_roster_data_long(con, center, start, end, filters)
    st.dataframe(df)

    if can_edit:
        st.divider()
        st.write("#### Quick Edit")
        c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 2, 1])
        agent_id = c1.text_input("Agent ID to edit")
        edit_date = c2.date_input("Date")
        new_shift_raw = c3.text_input("New Shift")

        if c4.button("Update Shift", disabled=not all([agent_id, edit_date, new_shift_raw])):
            new_shift = _normalize_shift_label(new_shift_raw)
            if new_shift:
                con.execute("UPDATE roster_long SET Shift = ? WHERE AgentID = ? AND Date = ? AND Center = ?",
                            [new_shift, agent_id, edit_date, center])
                con.commit()
                _upload_duckdb_back(local_path, file_id, db_name)
                st.success(f"Updated shift for {agent_id} on {edit_date}.")
            else:
                st.error("Invalid shift format.")


def render_admin_panel(con, local_path, file_id, db_name):
    st.subheader("🛡️ Admin Panel")
    all_centers = [c for c in get_all_centers(con) if c != "Overall"]
    center_sel = st.selectbox("Select Center to Manage", all_centers)

    if center_sel:
        acl = get_acl()

        # --- Edit Access ---
        st.markdown("##### Edit Access Control")
        allow_edit = acl.get(center_sel, {}).get("allow_edit", True)
        new_allow_edit = st.checkbox("Allow editing for this center", value=allow_edit)
        if new_allow_edit != allow_edit:
            acl.setdefault(center_sel, {})['allow_edit'] = new_allow_edit
            save_acl(acl)
            st.rerun()

        # --- Shift Codes ---
        st.markdown("##### Custom Shift Codes")
        codes = get_custom_shift_codes()
        new_code = st.text_input("Add new non-working code (e.g., TRAINING)")
        if st.button("Add Code"):
            if new_code and new_code.upper() not in codes:
                save_shift_codes(list(codes) + [new_code])
                st.rerun()
        st.multiselect("Current custom codes", list(codes), default=list(codes), disabled=True)


# =========================
# Main Application Flow
# =========================
user = _render_auth_gate()

con, local_db_path, db_file_id, db_err = get_duckdb_connection(DUCKDB_FILE_NAME)
if db_err:
    st.error(f"Fatal Error: Could not connect to the database. Please contact support. Details: {db_err}");
    st.stop()

# --- Sidebar ---
with st.sidebar:
    st.subheader(f"👤 {user.get('name', '')}")
    st.caption(f"{user.get('email', '')} ({user.get('role', 'viewer')})")
    if st.button("Logout", use_container_width=True):
        st.session_state.clear();
        st.rerun()
    st.divider()
    if st.button("↻ Force Refresh from Drive", use_container_width=True):
        if os.path.exists(local_db_path): os.remove(local_db_path)
        get_duckdb_connection.clear();
        st.rerun()
    st.divider()
    st.subheader("🧪 Database Stats")
    try:
        st.metric("Records Data", con.execute("SELECT COUNT(*) FROM records").fetchone()[0])
        st.metric("Roster Data", con.execute("SELECT COUNT(*) FROM roster_long").fetchone()[0])
    except Exception as e:
        st.warning(f"Could not fetch DB stats: {e}")
    st.divider()
    with st.expander("⬆️ Import / Fetch Files (Excel)"):
        st.markdown("**Requested workbook (.xlsx)**")
        req_file = st.file_uploader("Upload 'Requested' data", type=["xlsx"], key="req_up")
        if st.button("Import Requested", type="primary", use_container_width=True, disabled=(not req_file)):
            new_req = parse_workbook_to_df(req_file.getvalue())
            if new_req.empty:
                st.warning("No valid data parsed from 'Requested' file.")
            else:
                duckdb_upsert_records(con, local_db_path, db_file_id, DUCKDB_FILE_NAME, new_req); st.rerun()
        st.markdown("---")
        st.markdown("**Roster workbook (.xlsx)**")
        rost_file = st.file_uploader("Upload Roster data", type=["xlsx"], key="rost_up")
        roster_import_center = st.selectbox("Import Roster for Center",
                                            [c for c in get_all_centers(con) if c != "Overall"])
        if st.button("Import Roster (Replace for selected Center)", use_container_width=True,
                     disabled=(not rost_file or not roster_import_center)):
            raw_df = read_roster_sheet(rost_file.getvalue())
            if not raw_df.empty:
                long_df = unpivot_roster_wide(raw_df)
                if long_df.empty:
                    st.warning("No data rows after processing Roster file.")
                else:
                    duckdb_replace_roster_for_center(con, local_db_path, db_file_id, DUCKDB_FILE_NAME, long_df,
                                                     roster_import_center); st.rerun()
            else:
                st.warning("No 'Roster' sheet found or it's empty.")

# --- Main App Body ---
st.title("Center Shiftwise Dashboard")

all_centers = get_all_centers(con)
acl = get_acl()
allowed_centers, has_full_access = resolve_allowed_centers_from_email(user.get("email", ""),
                                                                      [c for c in all_centers if c != "Overall"], acl)
is_admin = user.get("role") == "admin"
center_options = all_centers if has_full_access or is_admin else (["Overall"] + allowed_centers)

c1, c2, c3, c4 = st.columns([1.5, 1.2, 1.3, 1.7])
with c1: center = st.selectbox("Center", center_options, index=0)
with c2:
    start_default, end_default = get_date_range(con, center)
    date_range = st.date_input("Date range", value=(start_default, end_default))
    start, end = (date_range if len(date_range) == 2 else (start_default, end_default))
with c3: view_type = st.selectbox("View", ["Shift_Wise Delta View", "Interval_Wise Delta View", "Overall_Delta View"])
with c4: langs_sel = st.multiselect("Languages", get_languages_for_center(con, center),
                                    default=get_languages_for_center(con, center))

with st.container():
    k1, k2, k3 = st.columns(3)
    req_total, ros_total, delta_total = get_kpi_metrics(con, center, langs_sel, start, end)
    k1.metric("Requested", req_total)
    k2.metric("Rostered", ros_total)
    k3.metric("Delta", delta_total, delta_color="inverse")

tabs_base = ["Dashboard", "Plotter", "Roster"]
if is_admin: tabs_base.append("Admin")
tab_choice = st.radio("Navigation", tabs_base, horizontal=True, key="active_tab", label_visibility="collapsed")

can_edit = user_can_edit_center(user.get("role"), center, allowed_centers, has_full_access)

if tab_choice == "Dashboard":
    render_dashboard(con, view_type, center, langs_sel, start, end)
elif tab_choice == "Plotter":
    render_plotter(con, local_db_path, db_file_id, DUCKDB_FILE_NAME, can_edit)
elif tab_choice == "Roster":
    render_roster(con, local_db_path, db_file_id, DUCKDB_FILE_NAME, center, start, end, can_edit)
elif tab_choice == "Admin":
    render_admin_panel(con, local_db_path, db_file_id, DUCKDB_FILE_NAME)
