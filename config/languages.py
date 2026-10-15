# ==========================================
# Aetheris Research - Language Dictionary
# English is Primary | Arabic is Secondary
# ==========================================

LANG = {
    "en": {
        # System
        "sys_title": "Aetheris Research",
        "sys_subtitle": "Global Research Platform for Economists & Decision-Makers",
        "sidebar_header": "Research Wings",
        "menu_title": "Select Wing",
        "language": "Language",
        "designed_by": "Designed for global researchers © 2026",

        # Wings
        "home": "Home",
        "data_hub": "Data Hub",
        "descriptive": "Descriptive Statistics",
        "inferential": "Inferential Statistics",
        "econometrics": "Econometrics Lab",
        "operations": "Operations Research",
        "reports": "Report Studio",
        "academy": "Academy",
        "community": "Community",

        # Common
        "welcome_msg": "Select a research wing from the sidebar to begin.",
        "upload_data": "Upload Your Data",
        "fetch_data": "Fetch International Data",
        "smart_memory": "Smart Memory",
        "no_data": "No data loaded in memory. Please upload or fetch data first.",
        "success": "Success",
        "error": "Error",
        "warning": "Warning",
        "info": "Info",
        "clear_memory": "Clear Memory",
        "download": "Download",
        "analyze": "Analyze",
        "generate_report": "Generate Report",
    },
    "ar": {
        # System
        "sys_title": "Aetheris Research",
        "sys_subtitle": "منصة بحثية عالمية للاقتصاديين وصناع القرار",
        "sidebar_header": "أجنحة البحث",
        "menu_title": "اختر الجناح",
        "language": "اللغة",
        "designed_by": "مصممة للباحثين حول العالم © 2026",

        # Wings
        "home": "الصفحة الرئيسية",
        "data_hub": "مركز البيانات",
        "descriptive": "الإحصاء الوصفي",
        "inferential": "الإحصاء الاستدلالي",
        "econometrics": "مختبر الاقتصاد القياسي",
        "operations": "بحوث العمليات",
        "reports": "استوديو التقارير",
        "academy": "الأكاديمية",
        "community": "المجتمع",

        # Common
        "welcome_msg": "اختر جناح البحث من القائمة الجانبية للبدء.",
        "upload_data": "رفع بياناتك",
        "fetch_data": "جلب بيانات دولية",
        "smart_memory": "الذاكرة الذكية",
        "no_data": "لا توجد بيانات محمّلة في الذاكرة. يرجى رفع أو جلب بيانات أولاً.",
        "success": "نجاح",
        "error": "خطأ",
        "warning": "تحذير",
        "info": "معلومة",
        "clear_memory": "مسح الذاكرة",
        "download": "تحميل",
        "analyze": "تحليل",
        "generate_report": "توليد تقرير",
    }
}


def get_text(lang_code: str, key: str) -> str:
    """Return translated text for the given language and key."""
    return LANG.get(lang_code, LANG["en"]).get(key, key)
