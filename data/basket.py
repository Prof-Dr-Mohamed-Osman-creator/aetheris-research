# ==========================================
# Aetheris Research - Data Basket
# ------------------------------------------
# Manage every dataset in Smart Memory and merge them
# (e.g. your survey/field data + World Bank indicators).
# ==========================================

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.memory import (list_datasets, get_dataset, active_name, activate_dataset,
                         remove_dataset, set_memory)
from core.utils import tr, csv_download
from data.merge import merge_datasets


def render_basket():
    names = list_datasets()
    if not names:
        st.info(tr("Your basket is empty. Upload a file or fetch global data first.",
                   "السلة فارغة. ارفع ملفاً أو اجلب بيانات دولية أولاً."))
        return

    # --- Library ---
    st.subheader(tr("📚 Datasets in Smart Memory", "📚 البيانات المحفوظة في الذاكرة الذكية"))
    rows = []
    for n in names:
        df = get_dataset(n)
        rows.append({tr("Name", "الاسم"): n,
                     tr("Rows", "الصفوف"): df.shape[0],
                     tr("Columns", "الأعمدة"): df.shape[1],
                     tr("Active", "نشطة"): "✅" if n == active_name() else ""})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        pick = st.selectbox(tr("Make active", "اجعلها البيانات النشطة"), names,
                            index=names.index(active_name()) if active_name() in names else 0, key="bk_active")
        if st.button(tr("✔ Activate", "✔ تفعيل"), key="bk_act_btn"):
            activate_dataset(pick)
            st.rerun()
    with c2:
        drop = st.selectbox(tr("Remove dataset", "حذف مجموعة بيانات"), names, key="bk_drop")
        if st.button(tr("🗑 Remove", "🗑 حذف"), key="bk_drop_btn"):
            remove_dataset(drop)
            st.rerun()

    # --- Merge ---
    if len(names) < 2:
        st.caption(tr("Add a second dataset to enable merging.", "أضف مجموعة بيانات ثانية لتفعيل الدمج."))
        return

    st.markdown("---")
    st.subheader(tr("🔗 Merge two datasets", "🔗 دمج مجموعتي بيانات"))
    st.caption(tr("Typical use: join your country-year or governorate-year data with World Bank indicators.",
                  "استخدام شائع: ربط بياناتك (دولة-سنة أو محافظة-سنة) بمؤشرات البنك الدولي."))
    c1, c2 = st.columns(2)
    with c1:
        left_name = st.selectbox(tr("Left (main) dataset", "المجموعة الأساسية (اليسار)"), names, key="bk_left")
    with c2:
        right_name = st.selectbox(tr("Right dataset", "المجموعة المضافة (اليمين)"),
                                  [n for n in names if n != left_name], key="bk_right")
    left, right = get_dataset(left_name), get_dataset(right_name)

    common = [c for c in left.columns if c in right.columns]
    c1, c2 = st.columns(2)
    with c1:
        lk = st.multiselect(tr("Key columns (left)", "أعمدة الربط (اليسار)"), list(left.columns),
                            default=common[:2], key="bk_lk")
    with c2:
        rk = st.multiselect(tr("Key columns (right, same order)", "أعمدة الربط (اليمين، بنفس الترتيب)"),
                            list(right.columns), default=common[:2], key="bk_rk")
    how = st.radio(tr("Join type", "نوع الدمج"), ["left", "inner", "outer"], horizontal=True,
                   format_func=lambda h: {"left": tr("Keep all left rows", "الاحتفاظ بكل صفوف الأساسية"),
                                          "inner": tr("Only matching rows", "الصفوف المتطابقة فقط"),
                                          "outer": tr("All rows from both", "كل الصفوف من الاثنتين")}[h],
                   key="bk_how")

    if st.button(tr("🔗 Merge", "🔗 دمج"), type="primary", key="bk_merge",
                 disabled=not lk or len(lk) != len(rk)):
        try:
            merged = merge_datasets(left, right, lk, rk, how)
        except Exception as e:  # noqa: BLE001 - show any pandas error to the user
            st.error(tr(f"Merge failed: {e}", f"فشل الدمج: {e}"))
            return
        name = f"{left_name} + {right_name}"
        set_memory(merged, source="merge", name=name)
        st.success(tr(f"Merged → '{name}': {merged.shape[0]} rows × {merged.shape[1]} columns (now active).",
                      f"تم الدمج ← '{name}': {merged.shape[0]} صف × {merged.shape[1]} عمود (أصبحت نشطة)."))
        st.dataframe(merged.head(30), width="stretch", hide_index=True)
        csv_download(merged, "merged.csv", key="bk_dl")
