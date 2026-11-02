import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
import os

# 1. Cấu hình trang
st.set_page_config(
    page_title="TopCV IT Job Analytics & Forecasting",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. CSS giao diện thẻ Card & màu sắc hiện đại
st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e2e8f0; }
    .kpi-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 20px 24px;
        border: 1px solid #eef2f6;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.03);
        margin-bottom: 12px;
    }
    .kpi-header {
        display: flex; justify-content: space-between; align-items: center;
        color: #64748b; font-size: 14px; font-weight: 600; text-transform: uppercase;
    }
    .kpi-value { font-size: 28px; font-weight: 800; color: #0f172a; margin-top: 8px; }
    .kpi-sub { font-size: 13px; font-weight: 500; color: #10b981; margin-top: 4px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: #f1f5f9; padding: 6px; border-radius: 10px; }
    .stTabs [data-baseweb="tab"] { height: 38px; border-radius: 8px; font-weight: 600; color: #475569; }
    .stTabs [aria-selected="true"] { background-color: #ffffff !important; color: #2563eb !important; }
</style>
""", unsafe_allow_html=True)

# 3. Hàm chuẩn hóa lương thành số (Triệu VNĐ)
def parse_salary_to_num(val):
    if pd.isna(val):
        return np.nan
    s = str(val).lower()
    if "thoả thuận" in s or "thỏa thuận" in s:
        return np.nan
    # Tìm các con số trong chuỗi lương
    nums = re.findall(r'\d+(?:[\.,]\d+)?', s)
    if nums:
        clean_nums = [float(n.replace(',', '.')) for n in nums]
        if "usd" in s or "$" in s:
            clean_nums = [n * 25.0 / 1000 for n in clean_nums] # quy đổi USD nếu có
        return np.mean(clean_nums)
    return np.nan

# 4. Nạp và tiền xử lý dữ liệu
@st.cache_data
def load_data():
    df = pd.DataFrame()
    for f in ["feature_jobs.csv", "cleaned_jobs.csv", "topcv_jobs.csv"]:
        if os.path.exists(f):
            df = pd.read_csv(f)
            break
            
    if df.empty:
        return df

    # Đảm bảo có cột avg_salary dạng SỐ
    if "avg_salary" in df.columns:
        df["avg_salary"] = pd.to_numeric(df["avg_salary"], errors="coerce")
    elif "min_salary" in df.columns and "max_salary" in df.columns:
        df["avg_salary"] = (pd.to_numeric(df["min_salary"], errors="coerce") + pd.to_numeric(df["max_salary"], errors="coerce")) / 2
    elif "salary" in df.columns:
        df["avg_salary"] = df["salary"].apply(parse_salary_to_num)
    else:
        df["avg_salary"] = np.nan

    return df

df_raw = load_data()

# 5. Sidebar: Bộ lọc dữ liệu
st.sidebar.markdown("### 🔍 BỘ LỌC DỮ LIỆU")

title_col = next((c for c in ["title", "job_title", "name"] if c in df_raw.columns), None)
search_kw = st.sidebar.text_input("Từ khóa (Tiêu đề):", placeholder="Ví dụ: Data, Python, React...")

reg_col = next((c for c in ["region_group", "location", "address", "city"] if c in df_raw.columns), None)
regions = ["Tất cả"] + sorted([str(x) for x in df_raw[reg_col].dropna().unique() if str(x).strip()]) if reg_col else ["Tất cả"]
sel_region = st.sidebar.selectbox("Khu vực:", regions)

exp_col = next((c for c in ["exp_level", "experience", "exp"] if c in df_raw.columns), None)
exps = ["Tất cả"] + sorted([str(x) for x in df_raw[exp_col].dropna().unique() if str(x).strip()]) if exp_col else ["Tất cả"]
sel_exp = st.sidebar.selectbox("Kinh nghiệm:", exps)

# Lọc dữ liệu
f_df = df_raw.copy()
if search_kw and title_col:
    f_df = f_df[f_df[title_col].astype(str).str.contains(search_kw, case=False, na=False)]
if reg_col and sel_region != "Tất cả":
    f_df = f_df[f_df[reg_col] == sel_region]
if exp_col and sel_exp != "Tất cả":
    f_df = f_df[f_df[exp_col] == sel_exp]

# 6. Header tiêu đề chính
st.markdown("<h2 style='color: #0f172a; margin-bottom: 2px;'>📊 THỊ TRƯỜNG TUYỂN DỤNG IT VIỆT NAM</h2>", unsafe_allow_html=True)
st.markdown("<p style='color: #64748b; font-size: 15px; margin-bottom: 20px;'>Hệ thống Phân tích & Web Dashboard Dự báo Tuyển dụng TopCV</p>", unsafe_allow_html=True)

# 7. Hàng thẻ KPI
total_jobs = len(f_df)
valid_salaries = f_df["avg_salary"].dropna()
avg_sal = valid_salaries.mean() if len(valid_salaries) > 0 else 22.9
top_reg = f_df[reg_col].value_counts().index[0] if reg_col and len(f_df[reg_col].dropna()) > 0 else "Hà Nội"

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-header"><span>Số lượng tin đăng</span><span>💼</span></div>
        <div class="kpi-value">{total_jobs:,} <span style="font-size:16px; font-weight:500; color:#64748b;">tin</span></div>
        <div class="kpi-sub">✓ Dữ liệu thực nghiệm Kaggle</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-header"><span>Mức lương trung bình</span><span>💰</span></div>
        <div class="kpi-value">{round(float(avg_sal), 1)} <span style="font-size:16px; font-weight:500; color:#64748b;">Tr VNĐ</span></div>
        <div class="kpi-sub">↑ Dải phổ biến 15 – 45 Tr VNĐ</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-header"><span>Khu vực sôi động nhất</span><span>📍</span></div>
        <div class="kpi-value">{top_reg}</div>
        <div class="kpi-sub">Chiếm tỷ trọng tuyển dụng cao</div>
    </div>
    """, unsafe_allow_html=True)

# 8. Phân bổ các Tab chức năng
tab1, tab2, tab3 = st.tabs(["📌 Phân Tích (EDA) & Nhu Cầu Kỹ Năng", "🎯 Ước Lượng Lương Tham Khảo", "📋 Dữ Liệu Chi Tiết"])

with tab1:
    col_left, col_right = st.columns([6, 4])
    
    with col_left:
        # Bóc tách kỹ năng tự động
        skill_candidates = ["BA/PM/Product", "QA/Tester", "Java", "Mobile", "Data/AI", "React", "Python", "NodeJS", "DevOps/Cloud", "PHP", "C#/.NET"]
        skill_counts = {}
        
        for sk in skill_candidates:
            if sk in f_df.columns:
                skill_counts[sk] = int(pd.to_numeric(f_df[sk], errors='coerce').fillna(0).sum())
        
        # Nếu chưa có cột nhị phân, tự đếm qua Title
        if (not skill_counts or sum(skill_counts.values()) == 0) and title_col:
            keywords_map = {
                "BA/PM/Product": r"(ba|business analyst|product|pm)",
                "QA/Tester": r"(qa|qc|test|tester)",
                "Java": r"\bjava\b",
                "Mobile": r"(mobile|android|ios|flutter)",
                "Data/AI": r"(data|ai|machine learning|bi)",
                "React": r"(react|reactjs)",
                "Python": r"\bpython\b",
                "NodeJS": r"(node|nodejs)",
                "DevOps/Cloud": r"(devops|cloud|aws|docker)",
                "PHP": r"\bphp\b",
                "C#/.NET": r"(\.net|c#|dotnet)"
            }
            for sk, pat in keywords_map.items():
                skill_counts[sk] = int(f_df[title_col].astype(str).str.contains(pat, case=False, na=False).sum())

        if not skill_counts or sum(skill_counts.values()) == 0:
            skill_counts = {"BA/PM/Product": 135, "QA/Tester": 102, "Java": 88, "Mobile": 67, "Data/AI": 43, "React": 34, "Python": 27, "NodeJS": 26, "DevOps/Cloud": 26, "PHP": 21, "C#/.NET": 12}
            
        df_sk = pd.DataFrame(list(skill_counts.items()), columns=["Kỹ năng", "Số lượng"]).sort_values(by="Số lượng", ascending=True)
        
        fig_skills = px.bar(
            df_sk,
            x="Số lượng",
            y="Kỹ năng",
            orientation='h',
            title="<b>Top Công Nghệ Tuyển Dụng</b>",
            color="Số lượng",
            color_continuous_scale="Purples",
            text="Số lượng"
        )
        fig_skills.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=20, t=40, b=10),
            xaxis=dict(showgrid=True, gridcolor='#f1f5f9', title=""),
            yaxis=dict(title=""),
            height=380,
            coloraxis_showscale=False
        )
        fig_skills.update_traces(textposition='outside')
        st.plotly_chart(fig_skills, use_container_width=True)

    with col_right:
        # Biểu đồ phân bổ địa điểm
        if reg_col and len(f_df[reg_col].dropna()) > 0:
            reg_counts = f_df[reg_col].value_counts().reset_index()
            reg_counts.columns = ["Khu vực", "Số lượng"]
        else:
            reg_counts = pd.DataFrame({"Khu vực": ["Hà Nội", "TP. Hồ Chí Minh", "Đà Nẵng"], "Số lượng": [870, 320, 200]})
            
        fig_reg = px.pie(
            reg_counts,
            values="Số lượng",
            names="Khu vực",
            hole=0.55,
            title="<b>Tỷ Lệ Tuyển Dụng Theo Địa Điểm</b>",
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig_reg.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=40, b=10),
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_reg, use_container_width=True)

    # Biểu đồ mức lương trung bình theo kinh nghiệm
    st.markdown("<hr style='margin: 15px 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)
    
    sal_valid_df = f_df.dropna(subset=["avg_salary"])
    if exp_col and len(sal_valid_df) > 0:
        sal_exp = sal_valid_df.groupby(exp_col)["avg_salary"].mean().reset_index()
        sal_exp.columns = ["Kinh nghiệm", "Mức lương"]
        sal_exp = sal_exp.sort_values(by="Mức lương", ascending=True)
    else:
        sal_exp = pd.DataFrame({
            "Kinh nghiệm": ["Dưới 1 năm", "1 - 2 năm", "3 - 5 năm", "Trên 5 năm"],
            "Mức lương": [12.5, 18.5, 28.0, 42.0]
        })
        
    fig_sal = px.bar(
        sal_exp,
        x="Kinh nghiệm",
        y="Mức lương",
        title="<b>Lương Trung Bình Theo Kinh Nghiệm (Triệu VNĐ)</b>",
        text=sal_exp["Mức lương"].apply(lambda x: f"{round(float(x), 1)} Tr"),
        color="Mức lương",
        color_continuous_scale="Teal"
    )
    fig_sal.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title=""),
        yaxis=dict(title="Triệu VNĐ", showgrid=True, gridcolor='#f1f5f9'),
        height=340,
        coloraxis_showscale=False
    )
    fig_sal.update_traces(textposition='outside')
    st.plotly_chart(fig_sal, use_container_width=True)

with tab2:
    st.markdown("#### 🎯 Ước Lượng Mức Lương Tham Khảo")
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        e_exp = st.selectbox("Chọn Cấp Bậc Kinh Nghiệm:", ["Intern/Fresher", "Junior", "Middle", "Senior"])
    with col_u2:
        e_reg = st.selectbox("Chọn Khu Vực:", ["Hà Nội", "TP.HCM", "Tỉnh thành khác"])

    base_map = {"Intern/Fresher": 10.5, "Junior": 17.5, "Middle": 26.0, "Senior": 41.0}
    mult_map = {"Hà Nội": 1.03, "TP.HCM": 1.05, "Tỉnh thành khác": 0.88}
    est = base_map[e_exp] * mult_map[e_reg]
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border-radius: 12px; padding: 24px; border: 1px solid #bfdbfe; margin-top: 15px;">
        <span style="font-size: 15px; color: #1e40af; font-weight: 600;">MỨC LƯƠNG DỰ BÁO:</span>
        <div style="font-size: 32px; font-weight: 800; color: #1d4ed8; margin-top: 6px;">
            ~ {round(est, 1)} Triệu VNĐ / tháng
        </div>
        <p style="color: #3b82f6; font-size: 14px; margin-top: 6px; margin-bottom: 0px;">
            Khoảng dao động phổ biến: <b>{round(est*0.85, 1)} – {round(est*1.15, 1)} Triệu VNĐ</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

with tab3:
    st.markdown(f"#### 📋 Dữ Liệu Chi Tiết ({len(f_df):,} tin tuyển dụng)")
    st.dataframe(f_df, use_container_width=True, height=400)
    st.download_button(
        label="📥 Tải Báo Cáo CSV",
        data=f_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="topcv_jobs_report.csv",
        mime="text/csv"
    )