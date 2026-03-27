import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. CẤU HÌNH & LOAD ASSETS ---
st.set_page_config(page_title="HCMC Smart Property AI", layout="wide")

@st.cache_resource
def load_all():
    try:
        model = joblib.load('model_real_estate_xgb.pkl')
        feature_names = joblib.load('feature_names.pkl')
        cats = joblib.load('categories.pkl')
        return model, feature_names, cats
    except: return None, None, None

model, feature_names, cats = load_all()

# --- 2. BỘ RÀO CHẮN LOGIC TỔNG LỰC (GIỮ NGUYÊN TÍNH CHẤT) ---
def validate_house(dt, ngang, stang, pngu, loai):
    errors, warnings = [], []
    dai = dt / ngang
    dt_su_dung = dt * stang 
    dt_san_phong = dt_su_dung / pngu  
    phong_tang = pngu / stang        
    
    if stang > 8: errors.append(f"❌ Vượt quy mô: {stang} tầng là Cao ốc/Khách sạn.")
    if loai == "nha_ngo_hem":
        if stang > 6: errors.append("❌ Sai quy hoạch: Nhà hẻm tối đa 6 tầng.")
        if dt > 300: errors.append(f"❌ Vô lý: Nhà hẻm không thể rộng {dt}m2.")
        if dt < 40 and stang > 3: errors.append("❌ Sai quy hoạch: Nhà hẻm < 40m2 tối đa 3 tầng.")
    if loai == "nha_biet_thu":
        if stang > 4: errors.append("❌ Sai quy chuẩn: Biệt thự thường tối đa 4 tầng.")
        if dt < 150 or ngang < 8: errors.append("❌ Sai chuẩn Biệt thự: Cần >= 150m2 và ngang >= 8m.")
    if stang >= 3 and pngu < (stang - 1): errors.append(f"❌ Logic công năng: Nhà {stang} tầng mà chỉ có {pngu} phòng là quá ít.")
    if pngu > 15 and loai != "nha_biet_thu": errors.append(f"❌ Sai mục đích: {pngu} phòng vượt quy mô dân dụng.")
    
    max_p_tang = 3 if (loai != "nha_biet_thu" and dt <= 150) else 5
    if phong_tang > max_p_tang: errors.append(f"❌ Vô lý: {phong_tang:.1f} phòng/tầng (Tối đa {max_p_tang}).")
    if dt_san_phong < 15: errors.append(f"❌ Diện tích hẹp: {dt_san_phong:.1f}m2/phòng là quá nhỏ.")
    if ngang > dai * 3: errors.append("❌ Nhà 'Băng rôn': Ngang quá lớn so với sâu.")

    return errors, warnings

# --- 3. HÀM DỰ ĐOÁN (CORE PREDICT) ---
def predict_price_single(dt, ngang, stang, pngu, quan, loai):
    final_df = pd.DataFrame(0, index=[0], columns=feature_names)
    final_df['dien_tich_dat_log_scaled'] = np.log1p(dt)
    final_df['dien_tich_su_dung_log_scaled'] = np.log1p(dt * 1.2)
    final_df['tong_so_tang_log_scaled'] = np.log1p(stang)
    final_df['so_phong_ngu_log_scaled'] = np.log1p(pngu)
    final_df['chieu_ngang_log_scaled'] = np.log1p(ngang)
    
    if f"quan_huyen_{quan}_encoded" in feature_names: final_df[f"quan_huyen_{quan}_encoded"] = 1
    if f"loai_hinh_{loai}_encoded" in feature_names: final_df[f"loai_hinh_{loai}_encoded"] = 1
    
    return np.expm1(model.predict(final_df))[0]

# --- 4. MENU SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/609/609803.png", width=100)
    st.title("HCMC Property AI")
    menu = st.radio("Menu chính:", ["🏠 Trang chủ", "💰 Dự đoán giá nhà", "🔍 Phát hiện bất thường"])

# --- 5. XỬ LÝ NỘI DUNG TỪNG TAB ---

if menu == "🏠 Trang chủ":
    st.header("Chào mừng đến với Hệ thống Định giá Bất động sản HCMC")
    
    try:
        st.image("banner_nhatot.png", use_container_width=True)
    except:
        st.error("⚠️ Vẫn không tìm thấy file banner_nhatot.png bro ơi!")

    st.markdown("""
    Hệ thống sử dụng mô hình **XGBoost** được huấn luyện trên dữ liệu thực tế tại 3 quận TP.HCM (Bình Thạnh, Gò Vấp, Phú Nhuận).
    - **Dự đoán giá**: Tính toán giá trị thị trường dựa trên thông số nhà.
    - **Phát hiện bất thường**: So sánh giá niêm yết với giá model dự đoán để tìm ra các trường hợp 'bất thường'.
    """)

    st.divider()

    # --- THÔNG TIN NHÓM & PHÂN CÔNG ---
    st.subheader("👥 Nhóm 3: Nguyễn Huỳnh Duy - Ngô Thị Phương Yến")
    work_df = pd.DataFrame({
        "Hạng mục công việc": ["GUI - Project 1", "GUI - Project 2"],
        "Người phụ trách": ["Ngô Thị Phương Yến", "Nguyễn Huỳnh Duy"]
    })
    st.table(work_df)

    # --- HƯỚNG DẪN SỬ DỤNG CHI TIẾT ---
    st.subheader("📖 Hướng dẫn sử dụng nhanh")
    
    guide_steps = pd.DataFrame({
        "Cách dùng": ["Nhập tay từng căn", "Upload hàng loạt (CSV)"],
        "Quy trình thực hiện (Sơ đồ)": [
            "Chọn Menu ➡️ Nhập thông số ➡️ Nhấn '🚀 Chạy'",
            "Tải file mẫu bên dưới ➡️ Điền dữ liệu ➡️ Upload ➡️ Nhấn '🚀 Quét'"
        ],
        "Lưu ý": ["Điền đủ diện tích, tầng, quận", "Phải dùng đúng file mẫu của hệ thống"]
    })
    st.table(guide_steps)

    # --- NÚT TẢI FILE MẪU (CHO NGƯỜI DÙNG KHỎI BỠ NGỠ) ---
    st.write("📂 **Dành cho mục Upload hàng loạt:**")
    # Tạo dữ liệu mẫu chuẩn
    template_data = pd.DataFrame({
        "dien_tich_dat": [50.5, 80.0],
        "chieu_ngang": [4.0, 5.0],
        "tong_so_tang": [2, 3],
        "so_phong_ngu": [2, 4],
        "quan_huyen": ["Binh Thanh", "Go Vap"],
        "loai_hinh": ["nha_ngo_hem", "nha_mat_tien"],
        "gia_rao_ban": [8.5, 12.0] # Cột này dùng cho tab Phát hiện bất thường
    })
    csv_template = template_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Tải File CSV Mẫu tại đây",
        data=csv_template,
        file_name="mau_du_lieu_nhatot.csv",
        mime="text/csv",
        help="Tải về, điền thông tin nhà của m vào rồi upload lên lại nhé!"
    )

elif menu == "💰 Dự đoán giá nhà":
    tab1, tab2 = st.tabs(["Nhập tay 1 căn", "Upload File (CSV)"])
    
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            dt = st.number_input("Diện tích đất (m2)", 5.0, 1500.0, 50.0)
            stang = st.number_input("Số tầng", 1, 15, 2)
            quan = st.selectbox("Quận/Huyện", cats['quan_huyen'] if cats else ["Quận 1"])
        with c2:
            ngang = st.number_input("Chiều ngang (m)", 0.5, 100.0, 4.0)
            pngu = st.number_input("Số phòng ngủ", 1, 100, 2)
            loai = st.selectbox("Loại hình", cats['loai_hinh'] if cats else ["nha_ngo_hem"])
        
        if st.button("🚀 Chạy Dự Đoán"):
            errs, warns = validate_house(dt, ngang, stang, pngu, loai)
            for w in warns: st.warning(w)
            if errs:
                for e in errs: st.error(e)
            else:
                gia = predict_price_single(dt, ngang, stang, pngu, quan, loai)
                st.success(f"💰 Giá AI dự báo: **{gia:.2f} tỷ VNĐ** ({ (gia*1000/dt):.1f} tr/m2)")

    with tab2:
        st.info("📤 Upload file CSV để AI tính toán giá trị thị trường hàng loạt.")
        # Dùng key riêng để không bị trùng với các tab khác
        file_pred = st.file_uploader("Chọn tệp CSV nhà cần định giá", type="csv", key="file_uploader_tab2")
        
        if file_pred:
            df_input = pd.read_csv(file_pred)
            # Nút bấm để kích hoạt dự đoán
            if st.button("🚀 Quét dự đoán hàng loạt", key="btn_predict_tab2"):
                prices = []
                # Chạy vòng lặp để tính giá từng căn trong file
                for idx, row in df_input.iterrows():
                    gia_du_bao = predict_price_single(
                        row['dien_tich_dat'], 
                        row['chieu_ngang'], 
                        row['tong_so_tang'], 
                        row['so_phong_ngu'], 
                        row['quan_huyen'], 
                        row['loai_hinh']
                    )
                    prices.append(round(gia_du_bao, 2))
                
                # Thêm cột kết quả vào bảng hiện tại
                df_input['Gia_AI_Du_Bao (Ty)'] = prices
                
                st.success("✅ Đã tính toán xong toàn bộ danh sách!")
                # Hiển thị bảng có cột giá mới
                st.dataframe(df_input, use_container_width=True)
                
                # Nút tải file kết quả về máy
                csv_data = df_input.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Tải file kết quả (.csv)", csv_data, "ket_qua_dinh_gia.csv", "text/csv", key="dl_tab2")

elif menu == "🔍 Phát hiện bất thường":
    tab3, tab4 = st.tabs(["Kiểm tra 1 căn", "Quét bất thường hàng loạt"])
    
    with tab3:
        st.info("Nhập thông số và GIÁ ĐANG RAO để kiểm tra độ bất thường.")
        c1, c2 = st.columns(2)
        with c1:
            dt_a = st.number_input("Diện tích (m2)", 5.0, 1500.0, 50.0, key="dt_a")
            stang_a = st.number_input("Số tầng", 1, 15, 2, key="stang_a")
            gia_nhap = st.number_input("Giá đang rao (Tỷ VNĐ)", 0.1, 1000.0, 10.0)
        with c2:
            ngang_a = st.number_input("Chiều ngang (m)", 0.5, 100.0, 4.0, key="ngang_a")
            pngu_a = st.number_input("Phòng ngủ", 1, 100, 2, key="pngu_a")
            quan_a = st.selectbox("Quận", cats['quan_huyen'], key="quan_a")
            loai_a = st.selectbox("Loại", cats['loai_hinh'], key="loai_a")

        if st.button("⚖️ Kiểm tra bất thường"):
            errs, warns = validate_house(dt_a, ngang_a, stang_a, pngu_a, loai_a)
            if errs:
                st.error(f"❌ Bất thường về Cấu trúc: {errs[0]}")
            else:
                gia_ai = predict_price_single(dt_a, ngang_a, stang_a, pngu_a, quan_a, loai_a)
                lech = (gia_nhap - gia_ai) / gia_ai * 100
                
                st.write(f"Giá AI dự báo: {gia_ai:.2f} tỷ | Giá bạn nhập: {gia_nhap:.2f} tỷ")
                
                if abs(lech) > 30:
                    st.error(f"🚨 Bất thường về Giá: Lệch {lech:.1f}% so với thị trường!")
                    if lech > 0: st.write("👉 Đánh giá: Nhà này đang bị **'Ngáo giá'** (Quá đắt).")
                    else: st.write("👉 Đánh giá: **'Kèo cực hời'** hoặc có vấn đề pháp lý ngầm nên rẻ bất thường.")
                else:
                    st.success(f"✅ Bình thường: Giá lệch {lech:.1f}% (Nằm trong vùng giao dịch an toàn).")

    with tab4:
        st.info("📤 Upload file CSV có cột `gia_rao_ban` để kiểm tra.")
        file_anom = st.file_uploader("Chọn file CSV quét hàng loạt", type="csv", key="tab4_upload")
        
        if file_anom:
            # Đọc file, đảm bảo bỏ qua các cột index thừa nếu có
            df_an = pd.read_csv(file_anom)
            
            if st.button("🔍 Bắt đầu truy quét"):
                results = []
                # Vòng lặp quét từng dòng
                for idx, row in df_an.iterrows():
                    # 1. Kiểm tra rào chắn
                    errs, _ = validate_house(
                        row['dien_tich_dat'], row['chieu_ngang'], 
                        row['tong_so_tang'], row['so_phong_ngu'], row['loai_hinh']
                    )
                    
                    # 2. Dự báo giá - SỬA TÊN HÀM THÀNH predict_price_single cho khớp phía trên
                    gia_ai = predict_price_single(
                        row['dien_tich_dat'], row['chieu_ngang'], 
                        row['tong_so_tang'], row['so_phong_ngu'], 
                        row['quan_huyen'], row['loai_hinh']
                    )
                    
                    # 3. Tính độ lệch
                    gia_rao = row['gia_rao_ban']
                    diff = ((gia_rao - gia_ai) / gia_ai) * 100
                    
                    # 4. Phân loại
                    status = "✅ Bình thường"
                    if errs:
                        status = f"🚨 Lỗi: {errs[0]}"
                    elif diff < -25:
                        status = "💎 KÈO HỜI"
                    elif diff > 30:
                        status = "🚩 Ngáo giá"
                    
                    results.append({
                        "STT": idx + 1,
                        "Quận": row['quan_huyen'],
                        "Diện tích": row['dien_tich_dat'],
                        "Giá Rao (Tỷ)": round(float(gia_rao), 2),
                        "Giá AI (Tỷ)": round(float(gia_ai), 2),
                        "Độ lệch (%)": f"{diff:.1f}%",
                        "Trạng thái": status
                    })
                
                # Hiển thị bảng
                st.subheader("📊 Kết quả phân tích danh sách")
                res_df = pd.DataFrame(results)
                st.dataframe(res_df, use_container_width=True)
                
                # Thống kê
                kèo_hời = res_df[res_df['Trạng thái'] == "💎 KÈO HỜI"]
                if not kèo_hời.empty:
                    st.success(f"🔥 Tìm thấy **{len(kèo_hời)}** kèo hời trong danh sách!")
                else:
                    st.write("Chưa tìm thấy kèo nào rẻ hơn 25% so với giá AI.")