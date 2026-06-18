# Workflow: Phân Khúc Khách Hàng Bằng Phương Pháp Clustering

> Tài liệu đặc tả quy trình (spec) — dùng để hướng dẫn AI hoặc lập trình viên sinh code rõ ràng, có cấu trúc module hóa, dễ đọc, dễ bảo trì và đúng chuẩn khoa học dữ liệu.

---

## Mục Lục

1. [Tổng Quan Dự Án](#1-tổng-quan-dự-án)
2. [Cấu Trúc Thư Mục Dự Án](#2-cấu-trúc-thư-mục-dự-án)
3. [Quy Ước Viết Code](#3-quy-ước-viết-code)
4. [Phase 0 — Thiết Lập & Nạp Dữ Liệu](#4-phase-0--thiết-lập--nạp-dữ-liệu)
5. [Phase 1 — Exploratory Data Analysis (EDA)](#5-phase-1--exploratory-data-analysis-eda)
6. [Phase 2 — Tiền Xử Lý Dữ Liệu](#6-phase-2--tiền-xử-lý-dữ-liệu)
7. [Phase 3 — Xác Định Số Cluster Tối Ưu](#7-phase-3--xác-định-số-cluster-tối-ưu)
8. [Phase 4 — Huấn Luyện Mô Hình Clustering](#8-phase-4--huấn-luyện-mô-hình-clustering)
9. [Phase 5 — Đánh Giá Mô Hình](#9-phase-5--đánh-giá-mô-hình)
10. [Phase 6 — Trực Quan Hóa Kết Quả](#10-phase-6--trực-quan-hóa-kết-quả)
11. [Phase 7 — Phân Tích & Đặt Tên Phân Khúc](#11-phase-7--phân-tích--đặt-tên-phân-khúc)
12. [Phase 8 — Tổng Kết & Báo Cáo](#12-phase-8--tổng-kết--báo-cáo)
13. [Checklist Hoàn Thành](#13-checklist-hoàn-thành)
14. [Thư Viện Cần Dùng](#14-thư-viện-cần-dùng)

---

## 1. Tổng Quan Dự Án

**Bài toán:** Phân khúc khách hàng của một cửa hàng bán lẻ dựa trên độ tuổi (Age), giới tính (Gender), thu nhập hàng năm (Annual Income) và điểm chi tiêu (Spending Score), nhằm xác định các nhóm khách hàng có hành vi tương tự nhau.

**Loại bài toán:** Unsupervised Learning — Clustering (không có nhãn/target có sẵn).

**Đầu vào:** File dữ liệu dạng `.csv` chứa các cột `CustomerID, Gender, Age, Annual Income, Spending Score` (tương tự bộ dữ liệu *Mall Customer Segmentation* phổ biến trên Kaggle — nếu giáo viên cung cấp file khác, chỉ cần đổi tên cột tương ứng).

**Đầu ra:**
- Notebook/script phân tích đầy đủ từ EDA → Clustering → Evaluation.
- Các biểu đồ trực quan lưu trong thư mục `reports/figures/`.
- Bảng so sánh hiệu năng giữa các thuật toán clustering.
- Mô tả đặc điểm (profile) và tên gợi ý cho từng phân khúc khách hàng.

**Nguyên tắc thiết kế workflow:** Mỗi phase độc lập, có input/output rõ ràng, có thể chạy lại từng phần mà không ảnh hưởng phase khác (giúp AI sinh code theo từng hàm/module riêng biệt).

---

## 2. Cấu Trúc Thư Mục Dự Án

```text
customer_segmentation/
├── data/
│   ├── raw/                  # Dữ liệu gốc, không chỉnh sửa
│   └── processed/            # Dữ liệu đã làm sạch/scale
├── notebooks/
│   └── customer_segmentation.ipynb
├── src/
│   ├── data_loader.py        # Nạp & kiểm tra dữ liệu
│   ├── eda.py                # Các hàm phân tích/vẽ biểu đồ EDA
│   ├── preprocessing.py      # Encode, scale, xử lý outlier
│   ├── clustering.py         # Huấn luyện các thuật toán clustering
│   ├── evaluation.py         # Tính các chỉ số đánh giá
│   └── visualization.py      # Vẽ biểu đồ cluster, dendrogram, PCA
├── reports/
│   └── figures/               # Lưu toàn bộ hình ảnh xuất ra
├── requirements.txt
└── README.md
```

> Nếu làm trên Jupyter Notebook đơn lẻ thay vì chia module `src/`, vẫn nên tách rõ từng phase bằng heading Markdown (`## Phase 0`, `## Phase 1`,...) tương ứng với các phần dưới đây.

---

## 3. Quy Ước Viết Code

Để AI/sinh viên sinh code dễ đọc, dễ kiểm tra và tái sử dụng, áp dụng các quy ước sau cho **mọi đoạn code** trong dự án:

- **Mỗi bước xử lý = một hàm riêng**, có tên mô tả rõ hành động (verb_noun), ví dụ: `load_dataset()`, `plot_feature_distributions()`, `scale_features()`, `train_kmeans()`.
- **Docstring đầy đủ** cho mỗi hàm: mô tả mục đích, tham số (`Args`), giá trị trả về (`Returns`).
- **Type hints** cho tham số và giá trị trả về (`def scale_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:`).
- **Không hard-code số liệu rải rác** — khai báo hằng số ở đầu file/notebook (ví dụ `RANDOM_STATE = 42`, `K_RANGE = range(2, 11)`).
- **In log tiến trình rõ ràng** ở mỗi bước (ví dụ `print("✔ Đã nạp dữ liệu: 200 dòng, 5 cột")`) để người đọc theo dõi pipeline chạy đến đâu.
- **Tách biệt phần tính toán và phần trực quan hóa** — hàm vẽ biểu đồ không nên đồng thời xử lý dữ liệu.
- **Lưu hình ảnh với tên file mô tả rõ nội dung**, ví dụ `reports/figures/elbow_method.png`, không dùng tên chung như `plot1.png`.
- **Comment giải thích "tại sao"**, không chỉ lặp lại "cái gì" (code đã tự nói lên việc nó làm gì).
- **Đặt `random_state` cố định** ở tất cả thuật toán có yếu tố ngẫu nhiên để đảm bảo kết quả tái lập được (reproducibility).

**Ví dụ khung hàm chuẩn (mẫu để AI tuân theo):**

```python
def plot_feature_distributions(df: pd.DataFrame, numeric_cols: list[str]) -> None:
    """
    Vẽ histogram + KDE cho từng biến số liên tục để quan sát phân phối.

    Args:
        df: DataFrame chứa dữ liệu khách hàng.
        numeric_cols: Danh sách tên cột số cần vẽ (vd: ['Age', 'Annual Income']).

    Returns:
        None. Hiển thị và lưu hình vào reports/figures/.
    """
    fig, axes = plt.subplots(1, len(numeric_cols), figsize=(5 * len(numeric_cols), 4))
    for ax, col in zip(axes, numeric_cols):
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Phân phối của {col}")
    plt.tight_layout()
    plt.savefig("reports/figures/feature_distributions.png", dpi=150)
    plt.show()
```

---

## 4. Phase 0 — Thiết Lập & Nạp Dữ Liệu

**Mục tiêu:** Chuẩn bị môi trường và đảm bảo dữ liệu được nạp đúng, kiểm tra tính hợp lệ ban đầu.

**Công việc cụ thể:**
- Import các thư viện cần thiết (xem mục 14).
- Đặt các hằng số cấu hình (`RANDOM_STATE`, đường dẫn dữ liệu, danh sách cột số).
- Đọc file dữ liệu bằng `pandas.read_csv()`.
- Kiểm tra nhanh: `df.shape`, `df.head()`, `df.dtypes`, `df.columns`.
- Đổi tên cột nếu cần cho ngắn gọn và nhất quán (vd: `"Annual Income (k$)"` → `Annual_Income`).

**Output mong đợi:** DataFrame `df` sạch về tên cột, đã xác nhận đúng số dòng/cột mong đợi, in log xác nhận.

---

## 5. Phase 1 — Exploratory Data Analysis (EDA)

Đây là phần quan trọng nhất để hiểu dữ liệu trước khi clustering — cần thực hiện **kỹ và đầy đủ**, không chỉ dừng ở thống kê mô tả.

### 5.1. Kiểm Tra Tổng Quan Dữ Liệu
- `df.info()` — kiểu dữ liệu từng cột.
- `df.describe()` — thống kê mô tả (mean, std, min, max, quartiles) cho biến số.
- `df.isnull().sum()` — kiểm tra missing values.
- `df.duplicated().sum()` — kiểm tra dòng trùng lặp.
- Kiểm tra giá trị bất thường trong cột phân loại: `df['Gender'].value_counts()`.

### 5.2. Phân Tích Đơn Biến (Univariate Analysis)
- Histogram + KDE cho `Age`, `Annual Income`, `Spending Score` để xem hình dạng phân phối (lệch trái/phải, đa đỉnh...).
- Boxplot cho từng biến số để phát hiện outlier trực quan.
- Countplot cho `Gender` để xem tỷ lệ nam/nữ trong dữ liệu.
- Ghi nhận nhận xét: biến nào lệch chuẩn, biến nào có outlier rõ.

### 5.3. Phân Tích Đa Biến (Bivariate / Multivariate Analysis)
- Scatter plot `Annual Income` vs `Spending Score` — đây là cặp biến quan trọng nhất cho bài toán phân khúc, thường lộ rõ các nhóm bằng mắt thường.
- Scatter plot `Age` vs `Spending Score` và `Age` vs `Annual Income`.
- `sns.pairplot(df, hue='Gender')` để quan sát tương tác giữa toàn bộ biến số, phân màu theo giới tính.
- Heatmap ma trận tương quan (`df.corr()`) giữa các biến số để kiểm tra đa cộng tuyến (multicollinearity).
- So sánh `Spending Score` trung bình giữa hai nhóm Gender (boxplot hoặc violin plot).

### 5.4. Phát Hiện & Xử Lý Outliers
- Dùng phương pháp IQR (`Q1 - 1.5*IQR`, `Q3 + 1.5*IQR`) hoặc Z-score để xác định outlier cho từng biến số.
- Quyết định giữ hay loại outlier: với bài toán phân khúc khách hàng, outlier (vd: khách thu nhập rất cao) thường **là thông tin hữu ích** chứ không phải nhiễu — nên ưu tiên giữ lại trừ khi rõ ràng là lỗi nhập liệu.

### 5.5. Tổng Hợp Insight Từ EDA
- Viết một đoạn Markdown tóm tắt 4-6 nhận xét chính từ EDA (vd: "Có khoảng 3-5 nhóm khách hàng lộ rõ qua scatter Income–Spending Score", "Age không có tương quan mạnh với Spending Score", "Dữ liệu cân bằng giữa hai giới tính"...).
- Những insight này sẽ định hướng cho việc chọn features và số cluster ở các bước sau.

**Output mong đợi:** Một bộ biểu đồ đầy đủ lưu tại `reports/figures/`, cùng phần tóm tắt insight bằng văn bản.

---

## 6. Phase 2 — Tiền Xử Lý Dữ Liệu

**Mục tiêu:** Chuẩn hóa dữ liệu để các thuật toán clustering (dựa trên khoảng cách) hoạt động chính xác.

**Công việc cụ thể:**
- **Xử lý missing values** (nếu có): loại dòng hoặc impute bằng median/mode tùy loại biến.
- **Chọn features dùng để clustering:**
  - Cách phổ biến nhất: chỉ dùng 2 biến số `Annual Income` và `Spending Score` (dễ trực quan hóa 2D).
  - Cách mở rộng: thêm `Age` để clustering 3D, cho góc nhìn đầy đủ hơn.
  - `Gender` là biến phân loại nhị phân — khuyến nghị **không đưa trực tiếp vào K-Means** (do K-Means dùng khoảng cách Euclidean, không phù hợp với biến nhị phân), mà giữ lại để **phân tích đặc điểm từng cluster sau khi đã chia nhóm** (Phase 7). Nếu muốn dùng Gender trong clustering, có thể thử encode rồi đánh giá riêng, ghi rõ đây là một thử nghiệm bổ sung.
- **Feature Scaling:** dùng `StandardScaler` (đưa về mean=0, std=1) — phù hợp hơn `MinMaxScaler` khi dữ liệu có outlier, vì MinMax dễ bị outlier kéo lệch khoảng giá trị.
- Lưu lại scaler đã fit để có thể inverse_transform khi cần diễn giải lại cluster center theo đơn vị gốc.

**Output mong đợi:** Ma trận `X_scaled` (numpy array hoặc DataFrame) sẵn sàng đưa vào các thuật toán clustering.

---

## 7. Phase 3 — Xác Định Số Cluster Tối Ưu

**Mục tiêu:** Tìm số lượng cluster (K) hợp lý trước khi huấn luyện mô hình chính thức.

**Công việc cụ thể:**
- **Elbow Method:** Tính WCSS (Within-Cluster Sum of Squares / inertia) cho K từ 2 đến 10, vẽ đường gấp khúc, tìm điểm "khuỷu tay" (elbow point).
- **Silhouette Score:** Tính silhouette score trung bình cho từng giá trị K, chọn K có điểm cao nhất (gần 1 là tốt).
- **Dendrogram (cho Hierarchical Clustering):** Vẽ dendrogram bằng `scipy.cluster.hierarchy.linkage` (method='ward'), quan sát khoảng cách hợp nhất lớn nhất để chọn điểm cắt hợp lý.
- So sánh kết quả gợi ý K từ cả 3 phương pháp trên — nếu chúng đồng thuận (vd: cả 3 đều gợi ý K=5), đó là lựa chọn đáng tin cậy.

**Output mong đợi:** Biểu đồ Elbow, biểu đồ Silhouette theo K, dendrogram, và một giá trị K được chọn kèm lý do giải thích rõ ràng.

---

## 8. Phase 4 — Huấn Luyện Mô Hình Clustering

**Mục tiêu:** Áp dụng và so sánh ít nhất 2-3 thuật toán clustering khác nhau theo yêu cầu đề bài.

**Các thuật toán cần triển khai:**

| Thuật toán | Tham số chính cần tinh chỉnh | Ghi chú |
|---|---|---|
| K-Means | `n_clusters`, `n_init`, `random_state` | Nhanh, cần chọn K trước, giả định cluster hình cầu |
| Hierarchical (Agglomerative) | `n_clusters`, `linkage` (ward/complete/average) | Không cần chọn K trước khi xem dendrogram, chậm hơn với dữ liệu lớn |
| DBSCAN | `eps`, `min_samples` | Tự phát hiện số cluster, phát hiện outlier (noise = -1), nhạy với tham số `eps` |
| K-Medoids *(tùy chọn)* | `n_clusters` | Bền với outlier hơn K-Means, dùng điểm thực tế làm tâm cluster |

**Công việc cụ thể:**
- Huấn luyện từng thuật toán trên `X_scaled` với tham số đã xác định ở Phase 3.
- Với DBSCAN: vẽ k-distance graph để tìm `eps` phù hợp thay vì chọn ngẫu nhiên.
- Gán nhãn cluster (`cluster_label`) trở lại vào DataFrame gốc cho từng thuật toán (vd: `df['cluster_kmeans']`, `df['cluster_hierarchical']`).

**Output mong đợi:** DataFrame gốc có thêm các cột nhãn cluster cho mỗi thuật toán, sẵn sàng cho bước đánh giá và trực quan hóa.

---

## 9. Phase 5 — Đánh Giá Mô Hình

**Mục tiêu:** Đánh giá định lượng chất lượng phân cụm của từng thuật toán.

**Các chỉ số cần tính cho mỗi thuật toán:**
- **Silhouette Coefficient** — đo độ tách biệt và độ gắn kết của cluster (từ -1 đến 1, càng cao càng tốt).
- **Davies-Bouldin Index** — đo tỷ lệ giữa độ phân tán trong cluster và khoảng cách giữa các cluster (càng thấp càng tốt).
- **Calinski-Harabasz Index** — tỷ lệ phương sai giữa các cluster so với trong cluster (càng cao càng tốt).

**Công việc cụ thể:**
- Tính cả 3 chỉ số cho từng thuật toán đã huấn luyện ở Phase 4.
- Tổng hợp vào **một bảng so sánh duy nhất** (DataFrame), ví dụ:

| Thuật toán | Số cluster | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---|---|---|---|
| K-Means | 5 | ... | ... | ... |
| Hierarchical | 5 | ... | ... | ... |
| DBSCAN | ... | ... | ... | ... |

- Đưa ra nhận xét: thuật toán nào cho kết quả tốt nhất theo các chỉ số, và có phù hợp với quan sát trực quan ở Phase 1 không.

**Output mong đợi:** Bảng so sánh hiệu năng + nhận xét lựa chọn thuật toán cuối cùng được sử dụng cho phân tích kinh doanh.

---

## 10. Phase 6 — Trực Quan Hóa Kết Quả

**Mục tiêu:** Biến kết quả số liệu thành hình ảnh trực quan, dễ hiểu cho người không chuyên.

**Công việc cụ thể:**
- Scatter plot 2D (`Annual Income` vs `Spending Score`), tô màu theo nhãn cluster, đánh dấu tâm cluster (centroid) bằng ký hiệu khác biệt (vd: dấu X).
- Nếu dùng từ 3 features trở lên: dùng PCA giảm chiều xuống 2D/3D để trực quan hóa, hoặc dùng scatter 3D trực tiếp (Age, Income, Spending).
- Vẽ lại dendrogram cuối cùng với đường cắt thể hiện số cluster đã chọn.
- So sánh side-by-side các biểu đồ scatter của K-Means / Hierarchical / DBSCAN trên cùng một hình (subplot) để dễ đối chiếu trực quan.

**Output mong đợi:** Bộ hình ảnh hoàn chỉnh, có tiêu đề và chú giải (legend) rõ ràng, lưu trong `reports/figures/`.

---

## 11. Phase 7 — Phân Tích & Đặt Tên Phân Khúc

**Mục tiêu:** Chuyển kết quả thống kê thành insight kinh doanh có ý nghĩa.

**Công việc cụ thể:**
- Group dữ liệu theo cluster đã chọn (`df.groupby('cluster_kmeans').mean()`), tính trung bình `Age`, `Annual Income`, `Spending Score` cho từng cluster.
- Tính thêm tỷ lệ Gender trong từng cluster (`groupby(['cluster', 'Gender']).size()`).
- Dựa trên đặc điểm trung bình, đặt tên gợi ý cho từng phân khúc, ví dụ:
  - Thu nhập cao – Chi tiêu cao → "Khách hàng VIP/cao cấp"
  - Thu nhập cao – Chi tiêu thấp → "Khách hàng tiết kiệm/cẩn trọng"
  - Thu nhập thấp – Chi tiêu cao → "Khách hàng chi tiêu mạnh tay dù thu nhập thấp"
  - Thu nhập thấp – Chi tiêu thấp → "Khách hàng phổ thông/ít tiềm năng"
  - Nhóm trung bình → "Khách hàng đại trà"
- Đề xuất ngắn gọn hướng tiếp cận marketing cho mỗi nhóm (vd: ưu đãi VIP, chương trình khuyến mãi kích thích chi tiêu...).

**Output mong đợi:** Bảng profile từng cluster (đặc điểm trung bình + tên gợi ý + đề xuất hành động).

---

## 12. Phase 8 — Tổng Kết & Báo Cáo

**Mục tiêu:** Tổng hợp toàn bộ quá trình thành báo cáo súc tích.

**Công việc cụ thể:**
- Tóm tắt: số cluster cuối cùng, thuật toán được chọn, lý do chọn.
- Liệt kê hạn chế của phân tích (vd: dữ liệu nhỏ, chưa có biến hành vi mua hàng thực tế, K-Means giả định cluster hình cầu nên có thể không phù hợp nếu dữ liệu có hình dạng phức tạp).
- Đề xuất hướng phát triển tiếp theo (vd: thu thập thêm dữ liệu giao dịch, thử thêm Gaussian Mixture Model, đánh giá độ ổn định cluster qua nhiều lần resample).

**Output mong đợi:** Phần kết luận bằng văn bản (Markdown cell cuối notebook hoặc mục riêng trong báo cáo).

---

## 13. Checklist Hoàn Thành

- [ ] Dữ liệu đã được nạp và kiểm tra (shape, dtypes, missing, duplicates).
- [ ] EDA đầy đủ: phân phối từng biến, scatter các cặp biến, heatmap tương quan, phát hiện outlier.
- [ ] Insight từ EDA được ghi lại rõ ràng bằng văn bản.
- [ ] Dữ liệu đã được scale (StandardScaler) trước khi clustering.
- [ ] Số cluster tối ưu được xác định bằng Elbow + Silhouette (+ Dendrogram).
- [ ] Tối thiểu 2-3 thuật toán clustering đã được huấn luyện và so sánh.
- [ ] Các chỉ số đánh giá (Silhouette, Davies-Bouldin, Calinski-Harabasz) đã được tính cho mỗi thuật toán.
- [ ] Kết quả cluster đã được trực quan hóa rõ ràng (scatter/dendrogram/PCA).
- [ ] Từng cluster đã được profile và đặt tên có ý nghĩa kinh doanh.
- [ ] Báo cáo tổng kết với hạn chế và hướng phát triển đã hoàn thành.
- [ ] Code tuân thủ quy ước ở Mục 3 (hàm có docstring, type hints, log tiến trình, tên file hình ảnh rõ nghĩa).

---

## 14. Thư Viện Cần Dùng

```text
pandas              # Xử lý dữ liệu dạng bảng
numpy                # Tính toán số học
matplotlib           # Vẽ biểu đồ cơ bản
seaborn              # Vẽ biểu đồ thống kê (histplot, pairplot, heatmap)
scikit-learn         # KMeans, AgglomerativeClustering, DBSCAN, StandardScaler, PCA, các metrics
scipy                # linkage, dendrogram (scipy.cluster.hierarchy)
yellowbrick          # (tùy chọn) KElbowVisualizer, SilhouetteVisualizer dựng sẵn
```
