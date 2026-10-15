# ==========================================
# Aetheris Research - Data Upload Module
# ==========================================

import streamlit as st
import pandas as pd
from io import BytesIO
from core.memory import set_memory, has_data, get_memory
from core.utils import t, get_current_language


def render_upload_section():
    """Render the My Data upload interface."""
    lang = get_current_language()

    st.subheader("📁 " + ("My Data - Upload Files" if lang == "en" else "بياناتي - رفع الملفات"))

    st.markdown(
        "Upload your dataset (CSV or Excel). The data will be stored in Smart Memory and become available for all analysis wings."
        if lang == "en" else
        "ارفع مجموعة بياناتك (CSV أو Excel). سيتم حفظ البيانات في الذاكرة الذكية وتصبح متاحة لجميع أجنحة التحليل."
    )

    uploaded_file = st.file_uploader(
        label=("Choose a file" if lang == "en" else "اختر ملفاً"),
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=False,
        help="Supported formats: CSV, Excel (.xlsx, .xls)"
    )

    if uploaded_file is not None:
        try:
            # Read the file based on extension
            file_name = uploaded_file.name.lower()

            if file_name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif file_name.endswith((".xlsx", ".xls")):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format." if lang == "en" else "صيغة الملف غير مدعومة.")
                return

            # Basic validation
            if df.empty:
                st.warning("The uploaded file is empty." if lang == "en" else "الملف المرفوع فارغ.")
                return

            # Store in Smart Memory
            set_memory(df, source=f"upload:{uploaded_file.name}")

            st.success(
                f"File uploaded successfully! Shape: {df.shape[0]} rows × {df.shape[1]} columns"
                if lang == "en" else
                f"تم رفع الملف بنجاح! الأبعاد: {df.shape[0]} صف × {df.shape[1]} عمود"
            )

            # Preview
            st.markdown("### " + ("Data Preview" if lang == "en" else "معاينة البيانات"))
            st.dataframe(df.head(20), use_container_width=True)

            # Quick info
            with st.expander("Dataset Info" if lang == "en" else "معلومات عن البيانات"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Columns:**" if lang == "en" else "**الأعمدة:**")
                    st.write(list(df.columns))
                with col2:
                    st.write("**Data Types:**" if lang == "en" else "**أنواع البيانات:**")
                    st.write(df.dtypes.astype(str).to_dict())

        except Exception as e:
            st.error(
                f"Error reading the file: {str(e)}"
                if lang == "en" else
                f"حدث خطأ أثناء قراءة الملف: {str(e)}"
            )

    # Show current memory status if data exists
    if has_data():
        st.markdown("---")
        st.info(
            "Data is currently loaded in Smart Memory and ready for analysis."
            if lang == "en" else
            "البيانات محمّلة حالياً في الذاكرة الذكية وجاهزة للتحليل."
        )
