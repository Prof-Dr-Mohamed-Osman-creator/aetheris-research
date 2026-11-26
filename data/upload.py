# ==========================================
# Aetheris Research - Data Upload Module
# ==========================================

import pandas as pd
import streamlit as st

from core.memory import set_memory, has_data
from core.utils import tr


def _read_file(uploaded_file, sheet=None) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".tsv"):
        return pd.read_csv(uploaded_file, sep="\t")
    if name.endswith(".json"):
        return pd.read_json(uploaded_file)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file, sheet_name=sheet or 0)
    raise ValueError("Unsupported file format")


def render_upload_section():
    """Render the My Data upload interface."""
    st.subheader("📁 " + tr("My Data - Upload Files", "بياناتي - رفع الملفات"))
    st.markdown(tr(
        "Upload your dataset. It is stored in Smart Memory and becomes available to all analysis wings. "
        "You can upload several files — each one is kept in the Data Basket.",
        "ارفع مجموعة بياناتك. ستُحفظ في الذاكرة الذكية وتصبح متاحة لجميع أجنحة التحليل. "
        "يمكنك رفع عدة ملفات، وتُحفظ كلها في سلة البيانات."))

    uploaded_file = st.file_uploader(
        label=tr("Choose a file", "اختر ملفاً"),
        type=["csv", "xlsx", "xls", "tsv", "json"],
        accept_multiple_files=False,
        help="CSV, Excel (.xlsx, .xls), TSV, JSON",
    )
    if uploaded_file is None:
        return

    sheet = None
    if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
        try:
            sheets = pd.ExcelFile(uploaded_file).sheet_names
        except Exception:  # noqa: BLE001
            sheets = []
        if len(sheets) > 1:
            sheet = st.selectbox(tr("Sheet", "ورقة العمل"), sheets, key="up_sheet")
        uploaded_file.seek(0)

    # Only (re)store when a new file or sheet is chosen, so the active dataset
    # chosen in the Data Basket is not overwritten on every rerun.
    signature = (uploaded_file.name, uploaded_file.size, sheet)
    try:
        df = _read_file(uploaded_file, sheet)
    except Exception as e:  # noqa: BLE001
        st.error(tr(f"Error reading the file: {e}", f"حدث خطأ أثناء قراءة الملف: {e}"))
        return
    if df.empty:
        st.warning(tr("The uploaded file is empty.", "الملف المرفوع فارغ."))
        return

    name = uploaded_file.name if not sheet else f"{uploaded_file.name} [{sheet}]"
    if st.session_state.get("up_signature") != signature:
        set_memory(df, source=f"upload:{uploaded_file.name}", name=name)
        st.session_state["up_signature"] = signature

    st.success(tr(f"File loaded: {df.shape[0]} rows × {df.shape[1]} columns",
                  f"تم تحميل الملف: {df.shape[0]} صف × {df.shape[1]} عمود"))
    st.markdown("### " + tr("Data Preview", "معاينة البيانات"))
    st.dataframe(df.head(20), width="stretch")

    with st.expander(tr("Dataset Info", "معلومات عن البيانات")):
        info = pd.DataFrame({
            tr("Column", "العمود"): df.columns,
            tr("Type", "النوع"): df.dtypes.astype(str).values,
            tr("Non-missing", "غير مفقود"): df.notna().sum().values,
            tr("Missing", "مفقود"): df.isna().sum().values,
            tr("Unique values", "قيم فريدة"): df.nunique().values,
        })
        st.dataframe(info, width="stretch", hide_index=True)

    if has_data():
        st.info(tr("Data is loaded in Smart Memory and ready for analysis.",
                   "البيانات محمّلة في الذاكرة الذكية وجاهزة للتحليل."))
