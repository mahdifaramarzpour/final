# ============================================================
# وارد کردن کتابخانه‌های لازم
# ============================================================
import streamlit as st                     # فریم‌ورک رابط کاربری
from PIL import Image                      # باز کردن تصویر
import Model_load as Ml                    # فایل خودمون برای پیش‌بینی
import torch                               # PyTorch (داخل Model_load)
import matplotlib.pyplot as plt            # برای نمودار
import numpy as np                         # برای کار با آرایه و argmax
import os                                  # کار با فایل و مسیر
import glob                                # پیدا کردن فایل با الگو


# ============================================================
# تنظیمات اولیه صفحه (باید اولین دستور Streamlit باشد)
# ============================================================
st.set_page_config(
    page_title="MNIST Digit Classifier",   # عنوان تب مرورگر
    page_icon="🔢",                        # آیکون تب مرورگر
    layout="centered"                       # چیدمان وسط‌چین
)


# ============================================================
# عنوان و زیرعنوان بالای صفحه
# ============================================================
st.title("🔢 MNIST Classifier")            # عنوان اصلی
st.caption("Upload is under Work! ...")    # متن کوچک زیر عنوان



# ============================================================
# چیدمان دو ستونه: چپ = تصویر، راست = پیش‌بینی
# ============================================================
col1, col2 = st.columns([1, 1], gap="medium")


# ------------------------------------------------------------
# ستون چپ
# ------------------------------------------------------------
with col1:
    st.subheader("📤 Image")               # زیرعنوان ستون چپ

    # دو تب: کتابخانه‌ی نمونه‌ها + آپلود
    tab1, tab2 = st.tabs(["Library", "Upload"])

    # دو متغیر برای مسیر تصویر انتخاب‌شده
    selected_image_path = None             # مسیر تصویر انتخاب‌شده
    uploaded_file = None                   # فایل آپلودشده

    # ---------- تب ۱: انتخاب از کتابخانه ----------
    with tab1:
        sample_images_dir = "sample_images"    # پوشه‌ی نمونه‌ها

        # چک می‌کنیم پوشه وجود دارد
        if os.path.exists(sample_images_dir):

            # همه‌ی فایل‌های داخل پوشه را می‌گیریم
            image_files = glob.glob(os.path.join(sample_images_dir, "*"))

            # فقط فایل‌های تصویری را نگه می‌داریم
            image_files = [
                f for f in image_files
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
            ]

            if image_files:
                # لیست نام فایل‌ها برای selectbox
                image_names = [os.path.basename(f) for f in image_files]

                # منوی کشویی انتخاب تصویر
                selected_image_name = st.selectbox(
                    "Sample images:",                  # برچسب
                    ["Select..."] + image_names,       # گزینه‌ی خالی + نام‌ها
                    label_visibility="collapsed"       # برچسب مخفی
                )

                # اگر کاربر "Select..." را انتخاب نکرد
                if selected_image_name != "Select...":
                    # مسیر کامل فایل
                    selected_image_path = os.path.join(sample_images_dir, selected_image_name)

                    # نمایش تصویر انتخاب‌شده
                    image = Image.open(selected_image_path)
                    st.image(image, caption=selected_image_name, width=200)

                    # نمایش اندازه‌ی تصویر
                    st.caption(f"Size: {image.size}")
            else:
                st.warning("No sample images found.")
        else:
            st.warning("Sample images folder not found.")

    # ---------- تب ۲: آپلود تصویر ----------
    with tab2:
        # ویجت آپلود فایل
        uploaded_file = st.file_uploader(
            "Choose image file",                                    # برچسب
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],             # فرمت‌های مجاز
            help="Upload a handwritten digit image"                 # راهنما
        )

        # اگر کاربر فایلی آپلود کرد
        if uploaded_file is not None:
            # نمایش تصویر آپلودشده
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded", width=200)

            # ذخیره‌ی موقت چون predict مسیر فایل می‌گیرد
            with open("temp_image.png", "wb") as f:
                f.write(uploaded_file.getbuffer())

            # مسیر فایل موقت را برای ستون راست نگه می‌داریم
            selected_image_path = "temp_image.png"


# ------------------------------------------------------------
# ستون راست
# ------------------------------------------------------------
with col2:
    st.subheader("🎯 Predictions")         # زیرعنوان ستون راست

    # فقط اگر تصویری انتخاب شده باشد
    if selected_image_path is not None:

        # نمایش اسپینر هنگام پیش‌بینی
        with st.spinner("Analyzing..."):
            # صدا زدن تابع پیش‌بینی از Model_load
            prediction_result = Ml.predict_mnist_probabilities(selected_image_path)

        # اگر خروجی با "Error" شروع شد، خطا نشان بده
        if prediction_result.startswith("Error"):
            st.error(prediction_result)

        # وگرنه نتیجه را تجزیه کن
        else:
            # ---- تجزیه‌ی خروجی ----
            # خروجی به شکل "0: 0.12\n1: 0.05\n..." است
            lines = prediction_result.strip().split('\n')

            digits = []                    # لیست رقم‌ها
            probabilities = []             # لیست احتمال‌ها

            # هر خط را به رقم و احتمال شکاف می‌دهیم
            for line in lines:
                digit, prob = line.split(': ')
                digits.append(int(digit))
                probabilities.append(float(prob))

            # پیدا کردن رقمی که بیشترین احتمال دارد
            predicted_digit = digits[np.argmax(probabilities)]
            max_probability = max(probabilities)

            # ---- نمایش نتیجه ----
            st.success(
                f"**Predicted Digit: {predicted_digit}** "
                f"(Confidence: {max_probability:.1%})"
            )

            # ---- رسم نمودار میله‌ای احتمال‌ها ----
            fig, ax = plt.subplots(figsize=(6, 3))

            # میله‌ها با رنگ آبی روشن
            bars = ax.bar(digits, probabilities, color='lightblue', alpha=0.8)

            # میله‌ی رقم پیش‌بینی‌شده را قرمز کن
            bars[predicted_digit].set_color('red')

            # برچسب‌های محورها
            ax.set_xlabel('Digit', fontsize=10)
            ax.set_ylabel('Probability', fontsize=10)
            ax.set_title('Prediction Probabilities', fontsize=11)

            # تنظیمات محور افقی
            ax.set_xticks(digits)                 # نمایش ۰ تا ۹
            ax.grid(True, alpha=0.3)              # شبکه‌ی کم‌رنگ
            ax.tick_params(axis='both', labelsize=8)

            # نوشتن عدد احتمال بالای میله‌های مهم (بیشتر از ۵٪)
            for i, prob in enumerate(probabilities):
                if prob > 0.05:
                    ax.text(
                        i, prob + 0.01,
                        f'{prob:.2f}',
                        ha='center', va='bottom',
                        fontsize=7
                    )

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

            # ---- جدول جزئیات داخل expander ----
            with st.expander("📊 Detailed Probabilities", expanded=False):
                prob_data = {
                    'Digit': digits,
                    'Probability': [f"{prob:.3f}" for prob in probabilities],
                    'Percentage': [f"{prob:.1%}" for prob in probabilities],
                }
                st.dataframe(prob_data, use_container_width=True, hide_index=True)

    # اگر هیچ تصویری انتخاب نشده باشد
    else:
        st.info("👆 Select an image to see predictions")


# ============================================================
# بخش راهنما (داخل expander، پیش‌فرض بسته)
# ============================================================
with st.expander("💡 Usage Tips", expanded=False):
    st.markdown("""
    **For better results:**
    - Use clear, high-contrast digit images
    - Single digits work best
    - White backgrounds with dark digits preferred
    - Images are resized to 28x28 pixels

    **Adding sample images:**
    - Place images in `sample_images` folder
    - Refresh app to see new images
    - Use descriptive filenames
    """)


# ============================================================
# پاک‌سازی: حذف فایل موقت در پایان اجرای اسکریپت
# ============================================================

if os.path.exists("temp_image.png"):       # اگر فایل موقت هست
    try:
        os.remove("temp_image.png")        # حذفش کن
    except:
        pass                                # اگر نشد، بی‌خیال