
import pandas as pd
from sklearn.datasets import load_iris

# ЗАВДАННЯ 1. Обираємо та завантажуємо набір даних Iris.
iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=[
        "Довжина чашолистка",
        "Ширина чашолистка",
        "Довжина пелюстки",
        "Ширина пелюстки"
    ]
)

print("Кількість об’єктів:", df.shape[0])
print("Кількість ознак:", df.shape[1])
# print(df.head())


# ЗАВДАННЯ 2. Обраний інструмент — Python.
# Підключаємо бібліотеки для графіків і кластеризації.
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ЗАВДАННЯ 3. Порівнюємо три числа кластерів: 2, 3 та 5.


# scaler = StandardScaler()
# X = scaler.fit_transform(df)

# colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]

# fig, axes = plt.subplots(
#     1, 3,
#     figsize=(10, 3.3),
#     sharex=True,
#     sharey=True,
#     layout="constrained"
# )

# for ax, k in zip(axes, [2, 3, 5]):
#     model = KMeans(n_clusters=k, random_state=42, n_init=10)
#     labels = model.fit_predict(X)

#     for cluster in range(k):
#         points = df[labels == cluster]

#         ax.scatter(
#             points.iloc[:, 2],
#             points.iloc[:, 3],
#             color=colors[cluster],
#             s=18,
#             alpha=0.7,
#             edgecolors="white",
#             linewidths=0.3
#         )

#     # Повертаємо координати центрів до початкового масштабу
#     centers = scaler.inverse_transform(model.cluster_centers_)

#     ax.scatter(
#         centers[:, 2],
#         centers[:, 3],
#         color="black",
#         marker="X",
#         s=70,
#         edgecolors="white",
#         linewidths=0.6,
#         label="Центри"
#     )

#     ax.set_title(f"k = {k}", fontsize=11)
#     ax.set_xlabel("Довжина пелюстки, см", fontsize=9)
#     ax.tick_params(labelsize=8)
#     ax.grid(alpha=0.2)
#     ax.set_axisbelow(True)
#     ax.legend(fontsize=7, loc="upper left")

# axes[0].set_ylabel("Ширина пелюстки, см", fontsize=9)

# plt.savefig("comparison_clusters.png", dpi=300, bbox_inches="tight")
# plt.show()



# ЗАВДАННЯ 4. Обираємо число кластерів методом «ліктя».
# Частина 1. Будуємо графік і обираємо k = 3.
# k_values = range(1, 11)
# inertia = []

# for k in k_values:
#     model = KMeans(n_clusters=k, random_state=42, n_init=10)
#     model.fit(X)
#     inertia.append(model.inertia_)

# fig, ax = plt.subplots(figsize=(5.5, 3.5), layout="constrained")

# ax.plot(
#     list(k_values), inertia,
#     marker="o",
#     markersize=5,
#     linewidth=2,
#     color="#0072B2"
# )

# ax.set_title("Метод «ліктя»", fontsize=12, pad=12)
# ax.set_xlabel("Кількість кластерів k", fontsize=10)
# ax.set_ylabel("Сума квадратів відстаней", fontsize=10)
# ax.set_xticks(list(k_values))
# ax.tick_params(labelsize=9)
# ax.set_ylim(0, max(inertia) * 1.12)
# ax.grid(alpha=0.25)
# ax.set_axisbelow(True)

# ax.scatter(3, inertia[2], color="#D55E00", s=65, zorder=3)
# ax.annotate(
#     "Обираємо k = 3",
#     xy=(3, inertia[2]),
#     xytext=(4.2, 260),
#     fontsize=10,
#     arrowprops=dict(arrowstyle="->", color="#D55E00")
# )

# fig.savefig(
#     "elbow_method.png",
#     dpi=300,
#     bbox_inches="tight",
#     pad_inches=0.2
# )

# plt.show()


# ЗАВДАННЯ 4 (продовження).
# Частина 2. Проводимо кластеризацію з обраним k = 3.
# scaler = StandardScaler()
# X = scaler.fit_transform(df)

# model = KMeans(n_clusters=3, random_state=42, n_init=10)
# labels = model.fit_predict(X)
# centers = scaler.inverse_transform(model.cluster_centers_)


# fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
# fig.subplots_adjust(
#     left=0.17, right=0.96,
#     bottom=0.18, top=0.75
# )

# colors = ["#0072B2", "#E69F00", "#009E73"]

# for cluster, color in enumerate(colors):
#     points = df[labels == cluster]

#     ax.scatter(
#         points.iloc[:, 2],
#         points.iloc[:, 3],
#         s=20,
#         alpha=0.7,
#         color=color,
#         edgecolors="white",
#         linewidths=0.3,
#         label=f"Кластер {cluster + 1}"
#     )

# ax.scatter(
#     centers[:, 2],
#     centers[:, 3],
#     marker="X",
#     s=70,
#     color="black",
#     label="Центри"
# )

# ax.set_title("Кластеризація Iris: k = 3", fontsize=11, pad=10)
# ax.set_xlabel("Довжина пелюстки, см", fontsize=9)
# ax.set_ylabel("Ширина пелюстки, см", fontsize=9)
# ax.tick_params(labelsize=8)
# ax.legend(fontsize=7, loc="upper left")
# ax.grid(alpha=0.2)
# ax.set_axisbelow(True)
# ax.margins(0.08)

# fig.savefig(
#     "clusters_k3.png",
#     dpi=300,
#     bbox_inches="tight",
#     pad_inches=0.2
# )

# plt.show()


# ЗАВДАННЯ 5. Порівнюємо Евклідову та Манхеттенську відстані.
# В обох випадках кількість кластерів однакова: k = 3.
import numpy as np

# Використовуємо всі 4 ознаки
scaler = StandardScaler()
X = scaler.fit_transform(df)
k = 3

# Частина 1. Евклідова відстань — k-середніх
euclidean_model = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)
euclidean_labels = euclidean_model.fit_predict(X)
euclidean_centers = euclidean_model.cluster_centers_


# Частина 2. Манхеттенська відстань — k-медіан
def k_medians(X, k, n_init=10):
    rng = np.random.default_rng(42)
    best_score = float("inf")
    best_labels = None
    best_centers = None

    for _ in range(n_init):
        centers = X[rng.choice(len(X), k, replace=False)].copy()

        for _ in range(300):
            # Сума абсолютних різниць за всіма ознаками
            distances = np.abs(
                X[:, None, :] - centers[None, :, :]
            ).sum(axis=2)

            labels = distances.argmin(axis=1)
            new_centers = centers.copy()

            for cluster in range(k):
                points = X[labels == cluster]
                if len(points) > 0:
                    new_centers[cluster] = np.median(points, axis=0)

            if np.allclose(centers, new_centers):
                centers = new_centers
                break

            centers = new_centers

        distances = np.abs(
            X[:, None, :] - centers[None, :, :]
        ).sum(axis=2)

        labels = distances.argmin(axis=1)
        score = distances[np.arange(len(X)), labels].sum()

        if score < best_score:
            best_score = score
            best_labels = labels.copy()
            best_centers = centers.copy()

    return best_labels, best_centers


manhattan_labels, manhattan_centers = k_medians(X, k)

fig, axes = plt.subplots(
    1, 2, figsize=(9, 4), dpi=100,
    sharex=True, sharey=True
)
fig.subplots_adjust(
    left=0.09, right=0.98,
    bottom=0.18, top=0.75, wspace=0.12
)

results = [
    ("Евклідова — k-середніх",
     euclidean_labels, euclidean_centers),
    ("Манхеттенська — k-медіан",
     manhattan_labels, manhattan_centers)
]

colors = ["#0072B2", "#E69F00", "#009E73"]

for ax, (title, labels, centers) in zip(axes, results):
    # Узгоджуємо кольори за довжиною пелюстки центрів
    order = np.argsort(centers[:, 2])

    for number, cluster in enumerate(order):
        points = df[labels == cluster]

        ax.scatter(
            points.iloc[:, 2], points.iloc[:, 3],
            color=colors[number],
            s=18, alpha=0.7,
            label=f"Кластер {number + 1}"
        )

    original_centers = scaler.inverse_transform(centers)

    ax.scatter(
        original_centers[:, 2], original_centers[:, 3],
        color="black", marker="X", s=65,
        label="Центри"
    )

    ax.set_title(title, fontsize=10, pad=10)
    ax.set_xlabel("Довжина пелюстки, см", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)
    ax.margins(0.08)

axes[0].set_ylabel("Ширина пелюстки, см", fontsize=9)

fig.savefig(
    "metrics_comparison.png",
    dpi=300, bbox_inches="tight", pad_inches=0.2
)
plt.show()
