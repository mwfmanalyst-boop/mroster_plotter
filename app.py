import io
import re
import json
@@ -48,132 +49,42 @@ def verify_local_password(entered_password: str, salt_hex: str, stored_sha256_he
    st.session_state.busy_roster = False

# Animated gradient background + card style
# ===== Theme / Global Styles =====
st.set_page_config(page_title="Center Shiftwise Web Dashboard", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class^="css"] {
    font-family: 'Inter', system-ui, Segoe UI, Roboto, Ubuntu, 'Helvetica Neue', Arial, 'Noto Sans' !important;
}

/* Animated main gradient */
/* page bg */
.stApp {
    background: linear-gradient(115deg, #667eea 0%, #764ba2 52%, #ff7eb3 100%);
    background-size: 320% 320%;
    animation: gradientMove 13s ease infinite;
    min-height: 100vh;
    padding: 28px 16px 48px 16px;
  background: linear-gradient(120deg, #0f172a, #1e293b, #0f172a);
  background-size: 400% 400%;
  animation: gradientMove 15s ease infinite;
}
@keyframes gradientMove {
    0%,100% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
}

/* Modern tabs */
.stTabs [data-baseweb="tab-list"] {
    background: linear-gradient(90deg, #ff7eb3 0%, #48cae4 100%);
    border-radius: 20px !important;
    box-shadow: 0 4px 14px rgba(80,80,200,0.14);
    padding: 8px 6px;
    margin-bottom: 26px;
}
.stTabs [data-baseweb="tab"] {
    color: #fff !important;
    font-weight: 600;
    border-radius: 14px !important;
    margin: 0 2px !important;
    background: rgba(255,255,255,0.11);
    transition: background .16s, color .16s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg,#764ba2 0,#48cae4 100%) !important;
    color: #fff !important;
    box-shadow: 0 2px 12px rgba(80,80,200,0.12);
  0% {background-position:0% 50%}
  50%{background-position:100% 50%}
  100%{background-position:0% 50%}
}

/* Floating cards and containers */
.card, .block-container, .stTabs [data-baseweb="tab-list"], [data-testid="stMetric"], [data-testid="stDataFrame"] div[role="grid"], section[data-testid="stSidebar"], .stForm {
    background: rgba(255,255,255,0.10) !important;
    border: 1px solid rgba(255,255,255,0.21) !important;
    border-radius: 22px !important;
    box-shadow: 0 10px 36px 0 rgba(31, 38, 135, 0.14) !important;
    backdrop-filter: blur(17px) !important;
    margin-bottom: 24px;
    position: relative;
    z-index: 2;
}

/* Emphasize headers */
h1, h2, h3, h4, h5, h6, .stMarkdown p, .stMarkdown li {
    color: #FFFFFF !important;
    text-shadow: 1px 2px 16px rgba(0,0,0,0.22), 0 1px 4px #764ba24d;
}

/* Buttons, modern gradients */
.stButton>button {
    border-radius: 20px !important;
    border: none !important;
    background: linear-gradient(95deg, #ff7eb3 0%, #48cae4 100%);
    color: #fff !important;
    font-weight: 600 !important;
    padding: 9px 36px !important;
    box-shadow: 0 4px 22px rgba(188,143,255,0.19);
    transition: background 0.3s, transform 0.18s;
}
.stButton>button:hover {
    background: linear-gradient(95deg, #48cae4 0%, #764ba2 100%);
    transform: translateY(-2px) scale(1.048);
/* cards */
.card {
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.15);
  border-radius: 14px;
  padding: 14px;
  backdrop-filter: blur(6px);
}

/* Metrics & DataFrame tweaks */
[data-testid="stMetricValue"], [data-testid="stMetricDelta"] {
    color: #e0e8ff !important;
/* headers & text */
h1, h2, h3, h4, h5, h6, .stMarkdown p, .stMarkdown li, .st-emotion-cache-ue6h4q {
  color: #e5e7eb !important;
}
[data-testid="stDataFrame"] table thead th {
    backdrop-filter: blur(10px);
    background: rgba(118,75,162,0.18);
    color: #fff;
[data-testid="stMetricDelta"], [data-testid="stMetricValue"] {
  color: #e5e7eb !important;
}
[data-testid="stDataFrame"] tbody tr:nth-child(odd) td {
    background: rgba(118,75,162,0.10);
/* table tweaks */
[data-testid="stDataFrame"] div[role="grid"] {
  border-radius: 10px;
  border: 1px solid rgba(255,255,255,0.15);
}

/* Sidebar gradient glassmorphism */
section[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #764ba2 0%, #48cae4 100%);
    opacity: 0.95;
    border-right: 1px solid rgba(255,255,255,0.16);
}

/* Input fields */
input, textarea {
    background: rgba(255,255,255,0.19) !important;
    border-radius: 12px !important;
    color: #232b5c !important;
    border: none !important;
    font-weight: 500 !important;
    box-shadow: 0 1px 10px #764ba226;
}

/* Card title bold */
.card h4, .card h3, .card h2, .card h1 {
    font-weight: 700;
    letter-spacing: 1px;
    color: #fff;
    background: linear-gradient(90deg,#ff7eb3 0,#48cae4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.block-container {
    max-width: 1400px;
    margin-top: 18px !important;
}

</style>
""", unsafe_allow_html=True)

# =========================
# 🔐 Auth + RBAC + ACL
# =========================
@@ -385,6 +296,13 @@ def _render_auth_gate():

ACL_FILE_NAME = st.secrets.get("ACL_FILE_NAME", "center_acl.json")

# === Custom shift codes configuration ===
# Administrators can add their own off/day codes (e.g. FS2, TRAINING_DAY) via the
# admin panel.  These codes are stored in a JSON file on Drive so that all
# instances of the app remain in sync.  A missing file simply yields an empty
# list.  See `_load_shift_codes_from_drive_cached` and `_save_shift_codes_to_drive`.
SHIFT_CODES_FILE_NAME = st.secrets.get("SHIFT_CODES_FILE_NAME", "shift_codes.json")

def _get_drive_service():
    SA_INFO = st.secrets.get("gcp_service_account", {})
    if not SA_INFO:
@@ -494,6 +412,63 @@ def _save_acl_to_drive(acl: dict, folder_id: Optional[str]):
    except Exception as e:
        st.error(f"Failed saving ACL: {e}")

# -- Shift-code helpers --
#
# The application supports a set of "shift codes" that are not time ranges
# (for example, WO, CL and other administrator-defined codes).  These codes
# live in a JSON file (`SHIFT_CODES_FILE_NAME`) stored in the same Drive
# folder as the ACL.  The helpers below load and save the codes in a
# cached manner so that editing them is inexpensive.

@st.cache_data(ttl=300)
def _load_shift_codes_from_drive_cached(folder_id: Optional[str]) -> list[str]:
    """
    Retrieve the list of custom shift codes from Google Drive.  If the file
    does not exist or contains invalid JSON, an empty list is returned.
    The returned list may contain strings of varying case; callers should
    normalise to uppercase when performing comparisons.
    """
    try:
        service = _get_drive_service()
        file_id = _drive_find_file_id(service, SHIFT_CODES_FILE_NAME, folder_id or None)
        if not file_id:
            return []
        raw = _drive_download_bytes(service, file_id)
        data = json.loads(raw.decode("utf-8"))
        if isinstance(data, list):
            return [str(x).strip() for x in data if x]
        return []
    except Exception:
        return []

def _save_shift_codes_to_drive(codes: list[str], folder_id: Optional[str]):
    """
    Persist the provided list of custom shift codes back to Google Drive.  The
    list is normalised to unique uppercase values before saving.  Any
    exceptions are surfaced to the user via `st.error`.  After saving the
    cached loader `_load_shift_codes_from_drive_cached` is cleared so that
    subsequent reads pick up the updated file.
    """
    try:
        service = _get_drive_service()
        codes_clean = sorted({str(c).strip().upper() for c in codes if c})
        data = json.dumps(codes_clean, indent=2).encode("utf-8")
        file_id = _drive_find_file_id(service, SHIFT_CODES_FILE_NAME, folder_id or None)
        _drive_upload_bytes(service, SHIFT_CODES_FILE_NAME, data, "application/json", folder_id or None, file_id)
        _load_shift_codes_from_drive_cached.clear()
    except Exception as e:
        st.error(f"Failed saving shift codes: {e}")

def get_custom_shift_codes() -> set[str]:
    """
    Return the set of administrator-defined shift codes as uppercase strings.  If
    no codes have been defined yet, this will return an empty set.  Codes
    defined in the JSON file are cached for a short duration to avoid
    excessive Drive API calls.
    """
    codes = _load_shift_codes_from_drive_cached(DRIVE_FOLDER_ID or None)
    return set(str(c).strip().upper() for c in codes if c)

def _norm_token(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', (s or '').lower())

@@ -515,10 +490,30 @@ def resolve_allowed_centers_from_email(email: str, all_centers: List[str], acl:
    return sorted(allowed), False

def user_can_edit_center(email: str, role: str, center: str, allowed_centers: list[str], has_full_access: bool) -> bool:
    """
    Determine whether the current user is allowed to edit data for the given
    center.  Editing requires admin role, that the center is within the
    allowed_centers list (unless the user has full access), and that the
    center itself has not been disabled for editing in the ACL.  The
    `allow_edit` flag is stored per center in the ACL.  If absent, edits
    default to being allowed.
    """
    # Only administrators can edit
    if role != "admin":
        return False
    # Check whether this center has been globally disabled via the ACL
    try:
        acl = _load_acl_from_drive_cached(DRIVE_FOLDER_ID or None)
        allow_flag = acl.get(center, {}).get("allow_edit", True)
        if allow_flag is False:
            return False
    except Exception:
        # In case ACL loading fails, fall through and rely on other checks
        pass
    # Admins with full access (e.g. domain match) can edit all centers
    if has_full_access:
        return True
    # Otherwise check membership of allowed_centers
    return center in set(allowed_centers)

# =========================
@@ -765,6 +760,18 @@ def _is_language_token(x) -> bool:
            return False
    return True

def _format_name(name: str) -> str:
    """
    Normalise a person name so that each word is capitalised (e.g. "john DOE"
    becomes "John Doe").  Non-alphabetic tokens are left untouched.  Empty or
    non-string inputs are returned unchanged.
    """
    try:
        parts = [p for p in str(name).strip().split() if p]
        return " ".join(p.capitalize() for p in parts)
    except Exception:
        return str(name)

def _find_language_header(df: pd.DataFrame, shift_row: int, shift_col: int, win: int = 16, rows_up: int = 3):
    ncols = df.shape[1]
    for up in range(1, rows_up + 1):
@@ -793,20 +800,73 @@ def _collect_contiguous_dates_in_row(df: pd.DataFrame, row: int, start_col: int)

def _is_off_code(s: str) -> Optional[str]:
    t = re.sub(r"\s+", "", str(s)).upper()
    if t in {"WO","OFF"}: return "WO"
    if t in {"CL","CO","COMPOFF","COMP-OFF"}: return "CL"
    if WOCL_VARIANTS.fullmatch(str(s)): return "WO+CL"
    # recognise built-in off codes
    if t in {"WO", "OFF"}:
        return "WO"
    if t in {"CL", "CO", "COMPOFF", "COMP-OFF"}:
        return "CL"
    if WOCL_VARIANTS.fullmatch(str(s)):
        return "WO+CL"
    # support custom codes defined by administrators
    try:
        if t in get_custom_shift_codes():
            return t
    except Exception:
        pass
    return None

def _is_valid_shift_label(s: str) -> bool:
    if _is_off_code(s): return True
    return bool(TIME_RANGE_RE.fullmatch(str(s)))
    # Off codes (built in or custom) are inherently valid
    if _is_off_code(s):
        return True
    # Custom codes may be provided in mixed case; allow them explicitly
    try:
        if str(s).strip().upper() in get_custom_shift_codes():
            return True
    except Exception:
        pass
    text = str(s).strip()
    # Must match a time range pattern at a minimum
    if not TIME_RANGE_RE.fullmatch(text):
        return False
    # Split on '/' to handle split shifts
    parts = [p.strip() for p in SHIFT_SPLIT.split(text) if p.strip()]
    durations = []
    for part in parts:
        m = re.fullmatch(r"(\d{1,2})[:]?(\d{2})\s*-\s*(\d{1,2})[:]?(\d{2})", part)
        if not m:
            return False
        sh, sm, eh, em = map(int, m.groups())
        sh = max(0, min(23, sh)); sm = max(0, min(59, sm))
        eh = max(0, min(23, eh)); em = max(0, min(59, em))
        start = sh * 60 + sm
        end = eh * 60 + em
        dur = end - start if end > start else (24 * 60 - start + end)
        durations.append(dur)
    # Single shift must be at least 9 hours (540 minutes)
    if len(durations) == 1:
        return durations[0] >= 540
    # Split shift must consist of exactly one 5h block and one 4h block (300 & 240 minutes)
    if len(durations) == 2:
        return sorted(durations) == [240, 300]
    # Other combinations are invalid
    return False

def _normalize_shift_label(s: str) -> Optional[str]:
    off = _is_off_code(s)
    if off: return off
    if off:
        # either a built-in off code or a custom code; return canonical uppercase form
        return off
    txt = str(s).strip()
    if not _is_valid_shift_label(txt): return None
    if not _is_valid_shift_label(txt):
        return None
    # If this matches a custom shift code, normalise and return
    try:
        t = txt.upper()
        if t in get_custom_shift_codes():
            return t
    except Exception:
        pass
    parts = SHIFT_SPLIT.split(txt)
    norm_parts = []
    for part in parts:
@@ -870,11 +930,96 @@ def parse_workbook_to_df(file_bytes: bytes) -> pd.DataFrame:
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=['Center','Language','Shift','Metric','Date','Value'])

def read_roster_sheet(file_bytes: bytes) -> pd.DataFrame:
    """
    Read the 'Roster' worksheet from an uploaded Excel file.  In addition to
    simply loading the sheet, this helper normalises names (Agent and TL)
    using `_format_name`.  If the workbook is missing a 'Roster' sheet or
    cannot be parsed, an empty DataFrame is returned and a warning is
    displayed.
    """
    try:
        return pd.read_excel(io.BytesIO(file_bytes), sheet_name="Roster")
        df = pd.read_excel(io.BytesIO(file_bytes), sheet_name="Roster")
    except Exception as e:
        st.warning(f"Could not read 'Roster' sheet: {e}")
        return pd.DataFrame()
    if df is None or df.empty:
        return df
    # Normalise person names.  Accept a variety of column labels for agent and TL names.
    for col in df.columns:
        key = str(col).strip().lower().replace("_", " ")
        if key in ("agent name", "agentname"):
            df[col] = df[col].astype(str).apply(_format_name)
        elif key in ("tl name", "tlname"):
            df[col] = df[col].astype(str).apply(_format_name)
    return df


def unpivot_roster_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a wide-format roster DataFrame containing date-wise shift columns
    into the long-format required by the database.  All columns that are not
    recognised as dates are treated as static agent attributes and are
    preserved.  Each date column produces a new row with a 'Date' and
    corresponding 'Shift'.  Shift values are passed through `_normalize_shift_label`;
    if normalisation fails the original text (uppercased) is retained so that
    invalid shifts remain visible in the resulting dataset.  If the input
    DataFrame is empty or no date columns are found, an empty DataFrame is
    returned.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    date_cols: list = []
    static_cols: list = []
    for col in df.columns:
        # Use the internal date-like heuristic to decide if a header is a date
        try:
            if _is_date_like(col):
                date_cols.append(col)
            else:
                static_cols.append(col)
        except Exception:
            static_cols.append(col)
    if not date_cols:
        return pd.DataFrame()
    records: list[dict] = []
    for _, row in df.iterrows():
        for dcol in date_cols:
            cell_val = row.get(dcol)
            # Skip blank cells completely
            if pd.isna(cell_val) or (isinstance(cell_val, str) and not cell_val.strip()):
                continue
            # Parse the header into a date.  Attempt ISO-like parsing first; fall back
            # to '%d-%b' with current year if necessary.
            dt_obj = None
            try:
                dt_obj = pd.to_datetime(dcol, errors='raise').date()
            except Exception:
                try:
                    dt_tmp = pd.to_datetime(f"{dcol}-{datetime.now().year}", format="%d-%b-%Y", errors='raise')
                    dt_obj = dt_tmp.date()
                except Exception:
                    dt_obj = None
            if dt_obj is None:
                continue
            rec: dict = {}
            # Copy through static columns
            for sc in static_cols:
                rec[sc] = row.get(sc)
            rec["Date"] = dt_obj
            # Normalise shift
            sval = str(cell_val)
            norm = _normalize_shift_label(sval)
            rec["Shift"] = norm if norm else sval.strip().upper()
            records.append(rec)
    out_df = pd.DataFrame(records)
    # Apply name normalisation to relevant columns
    for col in out_df.columns:
        key = str(col).strip().lower().replace("_", " ")
        if key in ("agent name", "agentname"):
            out_df[col] = out_df[col].astype(str).apply(_format_name)
        elif key in ("tl name", "tlname"):
            out_df[col] = out_df[col].astype(str).apply(_format_name)
    return out_df

# =========================
# Data model selection (prefer DuckDB)
@@ -1406,20 +1551,25 @@ def _save_requested_edits(center: str, language: str, edited_df: pd.DataFrame, s
            if raw_df.empty:
                st.warning("No Roster data found (need a 'Roster' sheet).")
            else:
                if DUCKDB_FILE_NAME:
                    local_db, file_id, err = _download_duckdb_rw(DUCKDB_FILE_NAME)
                    if err or not local_db or not file_id:
                        st.error(f"DuckDB issue: {err or 'File not found.'}")
                    else:
                        if "Date" not in raw_df or "Shift" not in raw_df:
                            st.error("Uploaded roster must have at least 'Date' and 'Shift' columns.")
                # Convert wide-format roster into long-format prior to saving.  This
                # will unpivot any date columns into rows with a Date and Shift.
                parsed_df = unpivot_roster_wide(raw_df)
                if parsed_df.empty:
                    st.error("Parsed roster contained no rows after processing. Please ensure the sheet contains date-wise shift data.")
                else:
                    if DUCKDB_FILE_NAME:
                        local_db, file_id, err = _download_duckdb_rw(DUCKDB_FILE_NAME)
                        if err or not local_db or not file_id:
                            st.error(f"DuckDB issue: {err or 'File not found.'}")
                        else:
                            raw_df["Date"] = pd.to_datetime(raw_df["Date"]).dt.date
                            inserted = duckdb_replace_roster(local_db, file_id, DUCKDB_FILE_NAME, raw_df)
                            # normalise date column
                            parsed_df["Date"] = pd.to_datetime(parsed_df["Date"]).dt.date
                            inserted = duckdb_replace_roster(local_db, file_id, DUCKDB_FILE_NAME, parsed_df)
                            st.success(f"Roster replaced in DuckDB. Rows inserted: {inserted}")
                else:
                    save_parquet_to_drive(ROSTER_FILE, raw_df)
                    st.success(f"Roster imported to Parquet. Rows: {len(raw_df)}")
                    else:
                        # fall back to Parquet; overwrite existing file with unpivoted roster
                        save_parquet_to_drive(ROSTER_FILE, parsed_df)
                        st.success(f"Roster imported to Parquet. Rows: {len(parsed_df)}")
        except Exception as e:
            st.error(f"Import Roster failed: {e}")

@@ -1966,29 +2116,31 @@ def roster_edit_dialog(roster_df: pd.DataFrame):
                    st.error("No 'Roster' sheet found or it's empty.")
                else:
                    try:
                        if DUCKDB_FILE_NAME:
                            local_db, file_id, err = _download_duckdb_rw(DUCKDB_FILE_NAME)
                            if err or not local_db or not file_id:
                                st.error(f"DuckDB issue: {err or 'File not found.'}")
                            else:
                                if "Date" not in raw or "Shift" not in raw:
                                    st.error("File must contain at least 'Date' and 'Shift' columns.")
                        # Convert the uploaded wide roster to long format
                        parsed_df = unpivot_roster_wide(raw)
                        if parsed_df.empty:
                            st.error("Parsed roster contained no rows after processing. Please ensure the sheet contains date-wise shift data.")
                        else:
                            # Assign the selected centre to all rows
                            parsed_df["Center"] = center
                            if DUCKDB_FILE_NAME:
                                local_db, file_id, err = _download_duckdb_rw(DUCKDB_FILE_NAME)
                                if err or not local_db or not file_id:
                                    st.error(f"DuckDB issue: {err or 'File not found.'}")
                                else:
                                    raw["Center"] = center
                                    raw["Date"] = pd.to_datetime(raw["Date"]).dt.date
                                    inserted = duckdb_replace_roster_for_center(local_db, file_id, DUCKDB_FILE_NAME, raw, center)
                                    parsed_df["Date"] = pd.to_datetime(parsed_df["Date"]).dt.date
                                    inserted = duckdb_replace_roster_for_center(local_db, file_id, DUCKDB_FILE_NAME, parsed_df, center)
                                    st.success(f"Roster replaced for {center}. Rows after replace: {inserted}")
                        else:
                            cur = load_parquet_from_drive(ROSTER_FILE)
                            raw["Center"] = center
                            raw["Date"] = pd.to_datetime(raw["Date"]).dt.date
                            if cur.empty:
                                save_parquet_to_drive(ROSTER_FILE, raw)
                            else:
                                keep = cur[cur.get("Center","").astype(str) != center]
                                new_all = pd.concat([keep, raw], ignore_index=True)
                                save_parquet_to_drive(ROSTER_FILE, new_all)
                            st.success(f"Roster replaced in Parquet for {center}. Rows added: {len(raw)}")
                                cur = load_parquet_from_drive(ROSTER_FILE)
                                parsed_df["Date"] = pd.to_datetime(parsed_df["Date"]).dt.date
                                if cur.empty:
                                    save_parquet_to_drive(ROSTER_FILE, parsed_df)
                                else:
                                    keep = cur[cur.get("Center","").astype(str) != center]
                                    new_all = pd.concat([keep, parsed_df], ignore_index=True)
                                    save_parquet_to_drive(ROSTER_FILE, new_all)
                                st.success(f"Roster replaced in Parquet for {center}. Rows added: {len(parsed_df)}")
                    except Exception as e:
                        st.error(f"Bulk replace failed: {e}")

@@ -2034,6 +2186,55 @@ def render_admin_panel(all_centers_no_overall: list[str], ACL: dict, user_email:
                    st.experimental_rerun()
                else:
                    st.info("Already present.")
    # ---- Center edit toggle ----
    st.markdown("### Edit Access Toggle")
    # The ACL may contain an 'allow_edit' flag per center; default is True if missing
    allow_edit_flag = ACL.get(center_sel, {}).get("allow_edit", True)
    toggle_val = st.checkbox("Allow editing for this center", value=bool(allow_edit_flag), key=f"edit_toggle_{center_sel}")
    if toggle_val != allow_edit_flag:
        ACL.setdefault(center_sel, {})['allow_edit'] = bool(toggle_val)
        _save_acl_to_drive(ACL, DRIVE_FOLDER_ID or None)
        st.success("Center edit access updated.")
        st.experimental_rerun()

    # ---- Shift code management ----
    st.markdown("### Manage Shift Codes")
    st.caption("Custom codes beyond WO/CL (e.g. FS2) can be created here. These codes are accepted in shift cells and bypass time-range validation.")
    current_codes = sorted(get_custom_shift_codes())
    if current_codes:
        for i, sc in enumerate(current_codes):
            c1, c2 = st.columns([6,1])
            with c1:
                st.write(sc)
            with c2:
                if st.button("❌", key=f"del_shift_code_{sc}"):
                    # Remove and persist
                    new_codes = [c for c in current_codes if c != sc]
                    _save_shift_codes_to_drive(new_codes, DRIVE_FOLDER_ID or None)
                    st.success(f"Removed shift code {sc}")
                    st.experimental_rerun()
    else:
        st.caption("_No custom shift codes defined._")
    new_code = st.text_input("Add new shift code").strip()
    if st.button("➕ Add Shift Code"):
        if not new_code:
            st.warning("Enter a code to add.")
        else:
            # Normalise to uppercase and ensure no whitespace
            candidate = re.sub(r"\s+", "", new_code).upper()
            if not candidate:
                st.warning("Invalid code.")
            elif candidate in {"WO", "CL", "WO+CL"}:
                st.warning("Built-in codes do not need to be added.")
            else:
                codes_set = set(current_codes)
                if candidate in codes_set:
                    st.info("Code already exists.")
                else:
                    codes_set.add(candidate)
                    _save_shift_codes_to_drive(sorted(codes_set), DRIVE_FOLDER_ID or None)
                    st.success(f"Added shift code {candidate}")
                    st.experimental_rerun()
    st.markdown("---")
    st.markdown("### Replace Roster for This Center Only")
    st.caption("Upload a Roster workbook (.xlsx) with a 'Roster' sheet. Only rows for this center will be replaced.")
@@ -2046,32 +2247,34 @@ def render_admin_panel(all_centers_no_overall: list[str], ACL: dict, user_email:
            if raw.empty:
                st.error("No 'Roster' sheet found or it's empty.")
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
                                raw["Center"] = center_sel
                                raw["Date"] = pd.to_datetime(raw["Date"]).dt.date
                                inserted = duckdb_replace_roster_for_center(local_db, file_id, DUCKDB_FILE_NAME, raw, center_sel)
                                st.success(f"Roster replaced for {center_sel}. Rows now present for this center: {inserted}")
                        except Exception as e:
                            st.error(f"Center roster replace failed: {e}")
                # Unpivot the uploaded roster into long format
                parsed_df = unpivot_roster_wide(raw)
                if parsed_df.empty:
                    st.error("Parsed roster contained no rows after processing. Please ensure the sheet contains date-wise shift data.")
                else:
                    cur = load_parquet_from_drive(ROSTER_FILE)
                    raw["Center"] = center_sel
                    raw["Date"] = pd.to_datetime(raw["Date"]).dt.date
                    if cur.empty:
                        save_parquet_to_drive(ROSTER_FILE, raw)
                    # assign the selected centre to all rows
                    parsed_df["Center"] = center_sel
                    if DUCKDB_FILE_NAME:
                        local_db, file_id, err = _download_duckdb_rw(DUCKDB_FILE_NAME)
                        if err or not local_db or not file_id:
                            st.error(f"DuckDB issue: {err or 'File not found.'}")
                        else:
                            try:
                                parsed_df["Date"] = pd.to_datetime(parsed_df["Date"]).dt.date
                                inserted = duckdb_replace_roster_for_center(local_db, file_id, DUCKDB_FILE_NAME, parsed_df, center_sel)
                                st.success(f"Roster replaced for {center_sel}. Rows now present for this center: {inserted}")
                            except Exception as e:
                                st.error(f"Center roster replace failed: {e}")
                    else:
                        keep = cur[cur.get("Center","").astype(str) != center_sel]
                        new_all = pd.concat([keep, raw], ignore_index=True)
                        save_parquet_to_drive(ROSTER_FILE, new_all)
                    st.success(f"Roster replaced in Parquet for {center_sel}. Rows: {len(raw)}")
                        cur = load_parquet_from_drive(ROSTER_FILE)
                        parsed_df["Date"] = pd.to_datetime(parsed_df["Date"]).dt.date
                        if cur.empty:
                            save_parquet_to_drive(ROSTER_FILE, parsed_df)
                        else:
                            keep = cur[cur.get("Center","*").astype(str) != center_sel]
                            new_all = pd.concat([keep, parsed_df], ignore_index=True)
                            save_parquet_to_drive(ROSTER_FILE, new_all)
                        st.success(f"Roster replaced in Parquet for {center_sel}. Rows: {len(parsed_df)}")

# ------- Persistent tab switcher -------
db = DBAdapter(records, roster, is_admin=is_admin, allowed_centers=allowed_centers, has_full_access=has_full_access)
