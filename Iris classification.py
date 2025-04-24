import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.svm import SVC
from sklearn.decomposition import PCA

# 讀取鳶尾花資料集
iris = datasets.load_iris()
X, y = iris.data, iris.target
target_names = iris.target_names

# 70%/30% 訓練-測試切分
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y,
)

# 建立包含標準化與 SVC 的管線
svc_clf = make_pipeline(StandardScaler(), SVC(gamma="scale"))

# 訓練模型
svc_clf.fit(X_train, y_train)

# 模型預測與評估
y_pred = svc_clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc:.3f}")  # e.g. 0.933

# 混淆矩陣
cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
fig_cm, ax_cm = plt.subplots()
ConfusionMatrixDisplay(confusion_matrix=cm,
                       display_labels=target_names).plot(ax=ax_cm)
ax_cm.set_title("Confusion Matrix on Test Set")

# 用 PCA 將 4 維特徵降到 2 維，方便畫散點圖
pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test)

fig_scatter, ax_scatter = plt.subplots()
scatter = ax_scatter.scatter(
    X_test_pca[:, 0],
    X_test_pca[:, 1],
    c=y_pred,
    edgecolor="k",
    s=60,
)
ax_scatter.set_xlabel("PCA_1")
ax_scatter.set_ylabel("PCA_2")
ax_scatter.set_title("SVC Predictions (70/30 spilit)")

# 依照預測結果擷取 legend label
handles, _ = scatter.legend_elements(prop="colors")
ax_scatter.legend(
    handles=handles,
    labels=list(target_names),
    title="Predicted Species",
    loc="best",
)

plt.tight_layout()
plt.show()
