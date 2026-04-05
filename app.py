import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. CẤU HÌNH & LOAD ASSETS ---
st.set_page_config(page_title="HCMC Smart Property", layout="wide")

@st.cache_resource
def load_all():
    try:
        model = joblib.load('final_model.pkl') 
        cats = joblib.load('categories.pkl')
        anomalies_list = joblib.load('anomalies_list.pkl') 
        try:
            f_names = model.feature_names_in_
        except:
            f_names = None 
            
        return model, cats, anomalies_list, f_names
    except Exception as e:
        st.error(f"Lỗi load file: {e}. Hãy chắc chắn đã để 3 file .pkl cùng thư mục với app.py")
        return None, None, None, None

model, cats, anomalies_list, feature_names = load_all()

# --- 2. BỘ RÀO CHẮN LOGIC TỔNG LỰC (GIỮ NGUYÊN TÍNH CHẤT) ---
def validate_house(dt, ngang, stang, pngu, loai):
    errors, warnings = [], []
    dai = dt / ngang
    dt_su_dung = dt * stang 
    dt_san_phong = dt_su_dung / pngu  
    phong_tang = pngu / stang         
    
    # --- THÊM MỚI: RÀO CHẮN CHIỀU NGANG VÀ TỶ LỆ ---
    if ngang < 2.0: 
        errors.append(f"❌ Chiều ngang {ngang}m là quá hẹp, không đủ tiêu chuẩn xây dựng/sinh hoạt.")
    if dai > ngang * 6: 
        errors.append(f"❌ Nhà quá mỏng: Chiều dài ({dai:.1f}m) gấp hơn 6 lần chiều ngang ({ngang}m).")
    # ----------------------------------------------

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
    final_df['dien_tich_su_dung_log_scaled'] = np.log1p(dt * stang)
    final_df['tong_so_tang_log_scaled'] = np.log1p(stang)
    final_df['so_phong_ngu_log_scaled'] = np.log1p(pngu)
    final_df['chieu_ngang_log_scaled'] = np.log1p(ngang)
    
    if f"quan_huyen_{quan}_encoded" in feature_names: final_df[f"quan_huyen_{quan}_encoded"] = 1
    if f"loai_hinh_{loai}_encoded" in feature_names: final_df[f"loai_hinh_{loai}_encoded"] = 1
            
    return np.expm1(model.predict(final_df))[0]

# --- 4. MENU SIDEBAR ---
# 1. Khởi tạo biến trạng thái nếu chưa có
if 'selected_page' not in st.session_state:
    st.session_state.selected_page = "🏠 Trang chủ"

# 2. Hàm để đồng bộ Radio khi bấm chọn trên Radio
def sync_radio():
    st.session_state.selected_page = st.session_state.main_radio

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/609/609803.png", width=50)
    st.title("HCMC BẤT ĐỘNG SẢN")
    st.divider()

    # CỤM 1: CÔNG CỤ HỆ THỐNG
    st.markdown("### 🛠️ CÔNG CỤ HỆ THỐNG")
    options = ["🏠 Trang chủ", "💰 Dự đoán giá nhà", "🔍 Phát hiện bất thường"]
    
    # Tính toán vị trí Radio dựa trên trang đang hiển thị
    # Nếu đang ở Business Problem hoặc Task Assignment thì Radio giữ ở vị trí hiện tại của nó
    current_idx = 0
    if st.session_state.selected_page in options:
        current_idx = options.index(st.session_state.selected_page)
    
    # Dùng biến menu_info để Mục 5 của Yến vẫn đọc được dữ liệu
    menu_info = st.radio(
        "Chọn chức năng:",
        options,
        index=current_idx,
        key="main_radio",
        on_change=sync_radio,
        label_visibility="collapsed"
    )

    st.divider()

    # CỤM 2: THÔNG TIN DỰ ÁN
    st.markdown("### 📖 THÔNG TIN DỰ ÁN")
    if st.button("🏢 Business Problem", use_container_width=True):
        st.session_state.selected_page = "🏢 Business Problem"
        st.rerun()
        
    if st.button("📋 Task Assignment", use_container_width=True):
        st.session_state.selected_page = "📋 Task Assignment"
        st.rerun()
        
    st.divider()
    st.subheader("👨‍🏫 Giảng viên hướng dẫn:")
    st.info("**ThS. Khuất Thùy Phương**")
    st.subheader("👥 Nhóm thực hiện (Nhóm 3):")
    st.write("📍 Nguyễn Huỳnh Duy")
    st.write("📍 Ngô Thị Phương Yến")

    
# --- 5. XỬ LÝ NỘI DUNG TỪNG TAB ---
# Kiểm tra xem có đang chọn xem Giới thiệu dự án ở Menu bên dưới không
menu_info = st.session_state.selected_page
if menu_info == "🏢 Business Problem":
    st.title("🏢 Business Problem - Bài toán kinh doanh")
    
    st.markdown("### 1. Bối cảnh dự án")
    st.write("""
    Dữ liệu dự án được thu thập thực tế từ nền tảng **Nhà Tốt** tại 3 quận trọng điểm: 
    **Bình Thạnh, Gò Vấp và Phú Nhuận**. Việc xác định giá trị thực giúp minh bạch hóa thị trường bất động sản.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("🎯 **Bài toán 1: Dự báo giá**\n\nXây dựng mô hình Regression gợi ý giá bán hợp lý dựa trên đặc điểm nhà.")
    with col2:
        st.warning("🔍 **Bài toán 2: Bất thường**\n\nPhát hiện tin đăng ảo hoặc giá sai lệch quá xa so với thị trường thực tế.")
    
    st.divider()
    st.markdown("### 2. Phương pháp & Kết quả")
    st.success("🏆 **Mô hình tối ưu: XGBoost  (R² ≈ 0.7523)**")
    st.write("**Công thức tính Anomaly Score:**")
    st.latex(r"Score = 40\% \cdot Residual + 40\% \cdot IsolationForest + 20\% \cdot LogicRules")

elif menu_info == "📋 Task Assignment":
    st.title("📋 Task Assignment - Phân công nhiệm vụ")
    # --- THÔNG TIN TRƯỜNG LỚP & THÀNH VIÊN ---
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #ff4b4b;">
        <p style="margin: 0; font-weight: bold;">📍  DL07 — K311</p>
        <p style="margin: 0; font-weight: bold;">📚  Đồ án tốt nghiệp</p>
        <p style="margin: 0; font-weight: bold;">🏫  ĐH Khoa Học Tự Nhiên TP.HCM</p>
        <hr style="margin: 10px 0;">
        <p style="margin: 0; font-size: 1.1em;">👥 <b>Thành viên thực hiện:</b> Ngô Thị Phương Yến — Nguyễn Huỳnh Duy</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("") # Tạo khoảng cách nhỏ
    
    
    # 2. Bảng phân công chi tiết (Dựa theo thực tế code của Yến)
    task_details = {
        "Giai đoạn": [
            "PRE-PROCESSING", 
            "EDA", 
            "Dự đoán giá (Sklearn)", 
            "Dự đoán giá (Pyspark)", 
            "Phát hiện bất thường (Sklearn)", 
            "Phát hiện bất thường (Pyspark)",
            "GUI Development", 
        ],
        "Trạng thái": [
            "✅ 100%", "✅ 100%", "✅ 100%", 
            "✅ 100%", "✅ 100%", "✅ 100%", "✅ 100%"
        ],
        "Thành viên": [
            "Yến & Duy", "Yến & Duy", "Yến", 
            "Duy", "Yến", "Duy", "Yến & Duy"
        ]
    }
    
    # Hiển thị bảng theo phong cách chuyên nghiệp
    st.dataframe(
        pd.DataFrame(task_details), 
        use_container_width=True, 
        hide_index=True
    )

# NẾU KHÔNG XEM GIỚI THIỆU, HIỂN THỊ CÁC CHỨC NĂNG APP (GIỮ NGUYÊN CODE CỦA YẾN)
elif menu_info == "🏠 Trang chủ":
    st.header("Chào mừng đến với Hệ thống Định giá Bất động sản HCMC")
    try:
        st.image("banner_nhatot.png", use_container_width=True)
    except:
        st.error("⚠️ Vẫn không tìm thấy file banner_nhatot.png bro ơi!")
    st.markdown("""
    Hệ thống sử dụng mô hình **XGBoost** được huấn luyện trên dữ liệu thực tế tại 3 quận TP.HCM (Bình Thạnh, Gò Vấp, Phú Nhuận).
    - **Dự đoán giá**: Tính toán giá trị thị trường dựa trên thông số nhà.
    - **Phát hiện bất thường**: So sánh giá niêm yết với giá model dự đoán để tìm ra các trường hợp 'bất thường'.
    
    ⚠️ **Lưu ý quan trọng**: Model được tối ưu hóa chính xác cho 3 quận trên. Các khu vực khác như **Quận 1, Quận 7...** vẫn có thể nhập liệu nhưng kết quả chỉ mang tính chất **tham khảo** do sự khác biệt về đặc thù giá trị đất giữa các khu vực.
    """)

    st.divider()

    # --- HƯỚNG DẪN SỬ DỤNG CHI TIẾT ---
    st.subheader("📖 Hướng dẫn sử dụng nhanh")
    
    guide_steps = pd.DataFrame({
            "Menu tính năng": ["🏠 Trang chủ", "💰 Dự đoán giá nhà", "🔍 Phát hiện bất thường"],
            "Mục đích": [
                "Xem thông tin nhóm, hướng dẫn sử dụng và tải file mẫu CSV",
                "Tính toán giá trị thị trường dựa trên thông số nhà",
                "So sánh giá niêm yết với dự đoán để tìm nhà giá rẻ/giá ảo"
            ],
            "Cách thực hiện": [
                "Đọc hướng dẫn và chọn file mẫu bên dưới",
                "Nhập thông số hoặc Upload CSV ➡️ Nhấn '🚀 Dự đoán'",
                "Nhập thông số + Giá rao hoặc Upload CSV ➡️ Nhấn '🚀 Quét'"
            ]
        })
    st.table(guide_steps)

        # --- NÚT TẢI FILE MẪU ---
    st.write("📂 **Tải file mẫu để Upload hàng loạt (chọn loại phù hợp nhu cầu):**")
    col1, col2 = st.columns(2)

    with col1:
        # File mẫu cho Dự đoán
        predict_template = pd.DataFrame({
            "dien_tich_dat": [50.5],
            "chieu_ngang": [4.0],
            "tong_so_tang": [2],
            "so_phong_ngu": [2],
            "quan_huyen": ["Binh Thanh"],
            "loai_hinh": ["nha_ngo_hem"]
        })
        st.download_button(
            label="📥 Tải mẫu Dự đoán giá",
            data=predict_template.to_csv(index=False).encode('utf-8'),
            file_name="mau_du_doan_gia.csv",
            mime="text/csv",
            help="Dùng cho menu Dự đoán giá nhà!"
        )

    with col2:
        # File mẫu cho Bất thường (có thêm cột giá rao bán)
        anomaly_template = predict_template.copy()
        anomaly_template["gia_rao_ban"] = [8.5]
        st.download_button(
            label="📥 Tải mẫu Phát hiện bất thường",
            data=anomaly_template.to_csv(index=False).encode('utf-8'),
            file_name="mau_quet_bat_thuong.csv",
            mime="text/csv",
            help="QUAN TRỌNG: Phải có cột giá rao bán mới quét được bất thường nha!"
        )

elif menu_info == "💰 Dự đoán giá nhà":
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
            
        
        if st.button("🚀 Dự Đoán"):
            errs, warns = validate_house(dt, ngang, stang, pngu, loai)
            for w in warns: st.warning(w)
            if errs:
                for e in errs: st.error(e)
            else:
                gia = predict_price_single(dt, ngang, stang, pngu, quan, loai)
                st.success(f"💰 Giá dự đoán: **{gia:.2f} tỷ VNĐ** ({ (gia*1000/dt):.1f} tr/m2)")

    with tab2:
        st.subheader("📊 Định giá bất động sản hàng loạt")
        
        st.divider()
        
        # --- BƯỚC 1: TẢI FILE MẪU ---
        st.write("📂 **Bước 1: Tải file mẫu để định giá hàng loạt**")
        predict_template = pd.DataFrame({
            "dien_tich_dat": [50.5], "chieu_ngang": [4.0], "tong_so_tang": [2],
            "so_phong_ngu": [2], "quan_huyen": ["Binh Thanh"], "loai_hinh": ["nha_ngo_hem"]
        })
        st.download_button(
            label="📥 Tải file CSV mẫu (Dự đoán)",
            data=predict_template.to_csv(index=False).encode('utf-8'),
            file_name="mau_du_doan_gia.csv",
            mime="text/csv",
            key="dl_template_predict_tab2_step1"
        )
        
        st.divider()
        
        # --- BƯỚC 2: UPLOAD VÀ TÍNH TOÁN ---
        st.write("📤 **Bước 2: Upload file cần định giá**")
        st.info("💡 Hệ thống sẽ tự động tính toán giá thị trường dựa trên các thông số trong file.")
        
        file_pred = st.file_uploader("Chọn tệp CSV nhà cần định giá", type="csv", key="file_uploader_tab2_step2")
        
        if file_pred:
            df_input = pd.read_csv(file_pred)
            
            # Nút bấm để kích hoạt dự đoán
            if st.button("🚀 Bắt đầu quét dự đoán", key="btn_predict_tab2_final"):
                prices = []
                # CHẠY VÒNG LẶP (Giữ nguyên logic dự báo của bạn)
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
                df_input['Gia_Du_Bao (Ty)'] = prices
                
                st.success("✅ Đã tính toán xong toàn bộ danh sách!")
                
                # Hiển thị bảng kết quả
                st.subheader("📊 Kết quả định giá danh sách")
                st.dataframe(df_input, use_container_width=True)
                
                # NÚT LƯU FILE (Download Button) - Xuất hiện sau khi tính toán xong
                # Dùng utf-8-sig để Excel mở lên không bị lỗi font tiếng Việt
                csv_result = df_input.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 Tải file kết quả định giá (.csv)",
                    data=csv_result,
                    file_name="ket_qua_dinh_gia_hang_loat.csv",
                    mime="text/csv",
                    key="dl_result_final_tab2"
                )

# Chuyển sang Menu Phát hiện bất thường
elif menu_info == "🔍 Phát hiện bất thường":
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
                gia_du_doan = predict_price_single(dt_a, ngang_a, stang_a, pngu_a, quan_a, loai_a)
                lech = (gia_nhap - gia_du_doan) / gia_du_doan * 100
                
                st.write(f"Giá dự báo: {gia_du_doan:.2f} tỷ | Giá bạn nhập: {gia_nhap:.2f} tỷ")
                
                if abs(lech) > 30:
                    st.error(f"🚨 Bất thường về Giá: Lệch {lech:.1f}% so với thị trường!")
                    if lech > 0: st.write("👉 Đánh giá: Nhà này đang bị **quá đắt**.")
                    else: st.write("👉 Đánh giá: **'Giá hời'** hoặc có vấn đề pháp lý ngầm nên rẻ bất thường.")
                else:
                    st.success(f"✅ Bình thường: Giá lệch {lech:.1f}% (Nằm trong vùng giao dịch an toàn).")

    with tab4:
        st.subheader("🏠 Kho dữ liệu nhà bất thường (Hệ thống tự quét)")
        
        st.divider()
        
        # --- BƯỚC 1: TẢI FILE MẪU ---
        st.write("📂 **Bước 1: Tải file mẫu để quét hàng loạt**")
        anomaly_template = pd.DataFrame({
            "dien_tich_dat": [50.5], "chieu_ngang": [4.0], "tong_so_tang": [2],
            "so_phong_ngu": [2], "quan_huyen": ["Binh Thanh"], "loai_hinh": ["nha_ngo_hem"],
            "gia_rao_ban": [8.5]
        })
        st.download_button(
            label="📥 Tải file CSV mẫu (Quét bất thường)",
            data=anomaly_template.to_csv(index=False).encode('utf-8'),
            file_name="mau_quet_bat_thuong.csv",
            mime="text/csv",
            key="dl_template_anomaly_tab4" # Key duy nhất
        )
        
        st.divider()
        
        # --- BƯỚC 2: UPLOAD VÀ TRUY QUÉT ---
        st.write("📤 **Bước 2: Upload file cần kiểm tra**")
        st.info("💡 Lưu ý: File upload phải có cột **gia_rao_ban** để so sánh.")
        
        file_anom = st.file_uploader("Chọn file CSV quét hàng loạt", type="csv", key="uploader_anom_tab4")
        
        if file_anom:
            df_an = pd.read_csv(file_anom)
            
            if st.button("🔍 Bắt đầu truy quét", key="btn_run_anom_tab4"):
                results = []
                # Vòng lặp quét từng dòng
                for idx, row in df_an.iterrows():
                    # 1. Kiểm tra rào chắn logic
                    errs, _ = validate_house(
                        row['dien_tich_dat'], row['chieu_ngang'], 
                        row['tong_so_tang'], row['so_phong_ngu'], row['loai_hinh']
                    )
                    
                    # 2. Dự báo giá
                    gia_du_doan = predict_price_single(
                        row['dien_tich_dat'], row['chieu_ngang'], 
                        row['tong_so_tang'], row['so_phong_ngu'], 
                        row['quan_huyen'], row['loai_hinh']
                    )
                    
                    # 3. Tính độ lệch
                    gia_rao = row['gia_rao_ban']
                    diff = ((gia_rao - gia_du_doan) / gia_du_doan) * 100
                    
                    # 4. Phân loại trạng thái
                    status = "✅ Bình thường"
                    if errs:
                        status = f"🚨 Lỗi: {errs[0]}"
                    elif diff < -25:
                        status = "💎 GIÁ HỜI"
                    elif diff > 30:
                        status = "🚩 GIÁ CAO"
                    
                    results.append({
                            "STT": idx + 1,
                            "Quận": row['quan_huyen'],
                            "Diện tích": row['dien_tich_dat'],
                            "Ngang": row['chieu_ngang'],
                            "Tầng": row['tong_so_tang'],
                            "Phòng ngủ": row['so_phong_ngu'],
                            "Giá Rao (Tỷ)": round(float(gia_rao), 2),
                            "Giá dự đoán (Tỷ)": round(float(gia_du_doan), 2),
                            "Độ lệch (%)": f"{diff:.1f}%",
                            "Trạng thái": status
                    })
                
                # Hiển thị bảng kết quả
                st.subheader("📊 Kết quả phân tích danh sách")
                res_df = pd.DataFrame(results)
                st.dataframe(res_df, use_container_width=True)
                
                # --- NÚT LƯU FILE TẠI ĐÂY ---
                csv_anom = res_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Tải báo cáo bất thường (.csv)",
                    data=csv_anom,
                    file_name="ket_qua_quet_bat_thuong.csv",
                    mime="text/csv",
                    key="dl_result_anom_tab4"
                )
                # --------------------------------
                
                # Thống kê nhanh
                gia_hoi_count = len(res_df[res_df['Trạng thái'] == "💎 GIÁ HỜI"])
                if gia_hoi_count > 0:
                    st.success(f"🔥 Tuyệt vời! Tìm thấy **{gia_hoi_count}** căn có giá hời (rẻ hơn 25% so với thị trường)!")
                else:
                    st.write("Cơ hội đầu tư: Chưa tìm thấy căn nào có giá rẻ bất thường.")