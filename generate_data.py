"""
Практическое задание №7
Стохастический градиентный спуск по мини-батчам (MSGD)
Идентификация цифр MNIST
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ─────────────────────────────────────────────
# 3.1. Подключение библиотек и путей
# ─────────────────────────────────────────────

DATA_DIR = "data"          # папка с CSV-файлами
RESULTS_DIR = "results"    # папка для сохранения графиков
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 3.2. Определение класса многослойного персептрона
# ─────────────────────────────────────────────

class MLP:
    """
    Многослойный персептрон с поддержкой мини-батч SGD.

    Параметры
    ----------
    layer_sizes : list[int]
        Количество нейронов в каждом слое (включая входной и выходной).
        Пример: [784, 128, 64, 10]
    learning_rate : float
        Шаг обучения η.
    batch_size : int
        Размер мини-батча.
        1   → стохастический градиентный спуск (SGD)
        N   → стандартный градиентный спуск (GD, где N = len(X_train))
        2…N-1 → мини-батч SGD (MSGD)
    training_epochs : int
        Число эпох обучения.
    seed : int | None
        Зерно генератора случайных чисел (для воспроизводимости).
    """

    def __init__(
        self,
        layer_sizes: list,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        training_epochs: int = 50,
        seed: int = 42,
    ):
        if seed is not None:
            np.random.seed(seed)

        self.layer_sizes = layer_sizes
        self.lr = learning_rate
        self.batch_size = batch_size
        self.training_epochs = training_epochs

        self.n_layers = len(layer_sizes)

        # He-инициализация весов; смещения — нули
        self.weights = []
        self.biases = []
        for i in range(self.n_layers - 1):
            fan_in = layer_sizes[i]
            W = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / fan_in)
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(W)
            self.biases.append(b)

        # Истории метрик
        self.train_loss_history = []
        self.test_loss_history = []
        self.train_acc_history = []
        self.test_acc_history = []

    # ── Вспомогательные функции ──────────────────────────────────────────

    @staticmethod
    def relu(z):
        return np.maximum(0.0, z)

    @staticmethod
    def relu_derivative(z):
        return (z > 0).astype(float)

    @staticmethod
    def softmax(z):
        # Вычитаем максимум по строкам для численной устойчивости
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    @staticmethod
    def cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Категориальная кросс-энтропия."""
        eps = 1e-12
        return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))

    @staticmethod
    def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))

    # ── Прямой проход ────────────────────────────────────────────────────

    def forward(self, X: np.ndarray):
        """
        Возвращает активации всех слоёв.
        Скрытые слои — ReLU, выходной — Softmax.
        """
        activations = [X]
        zs = []
        a = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b
            zs.append(z)
            if i < self.n_layers - 2:   # скрытые слои
                a = self.relu(z)
            else:                        # выходной слой
                a = self.softmax(z)
            activations.append(a)
        return activations, zs

    # ── Обратный проход ──────────────────────────────────────────────────

    def backward(self, activations, zs, y_true: np.ndarray):
        """
        Вычисляет градиенты методом обратного распространения ошибки.
        Возвращает списки dW и db для каждого слоя.
        """
        m = y_true.shape[0]
        dW_list = [None] * (self.n_layers - 1)
        db_list = [None] * (self.n_layers - 1)

        # Градиент на выходном слое (Softmax + Cross-Entropy)
        delta = activations[-1] - y_true          # (m, n_out)

        for i in reversed(range(self.n_layers - 1)):
            dW = activations[i].T @ delta / m
            db = delta.mean(axis=0, keepdims=True)
            dW_list[i] = dW
            db_list[i] = db

            if i > 0:   # проталкиваем ошибку дальше
                delta = delta @ self.weights[i].T * self.relu_derivative(zs[i - 1])

        return dW_list, db_list

    # ── Обновление параметров ────────────────────────────────────────────

    def update_params(self, dW_list, db_list):
        for i in range(self.n_layers - 1):
            self.weights[i] -= self.lr * dW_list[i]
            self.biases[i]  -= self.lr * db_list[i]

    # ── Предсказание ─────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        activations, _ = self.forward(X)
        return activations[-1]

    # ── Обучение ─────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        verbose: bool = True,
    ):
        """
        Обучение модели методом мини-батч SGD.

        Обрабатываются все три случая:
          batch_size == 1          → SGD
          batch_size >= len(X_train) → GD (стандартный)
          иначе                   → MSGD
        В конце каждой эпохи, если последний батч неполный, он всё равно
        используется (стратегия 1 из теоретической части).
        """
        n = len(X_train)
        # Реальный batch_size: ограничиваем сверху размером выборки
        bs = min(self.batch_size, n)

        for epoch in range(1, self.training_epochs + 1):
            # Перемешиваем обучающую выборку в начале каждой эпохи
            perm = np.random.permutation(n)
            X_shuf = X_train[perm]
            Y_shuf = Y_train[perm]

            # Разбивка на батчи; последний батч может быть неполным (стратегия 1)
            start = 0
            while start < n:
                end = start + bs
                X_batch = X_shuf[start:end]
                Y_batch = Y_shuf[start:end]

                activations, zs = self.forward(X_batch)
                dW, db = self.backward(activations, zs, Y_batch)
                self.update_params(dW, db)

                start = end

            # Метрики по всей выборке в конце эпохи
            train_pred = self.predict(X_train)
            test_pred  = self.predict(X_test)

            tl = self.cross_entropy(train_pred, Y_train)
            vl = self.cross_entropy(test_pred,  Y_test)
            ta = self.accuracy(train_pred, Y_train)
            va = self.accuracy(test_pred,  Y_test)

            self.train_loss_history.append(tl)
            self.test_loss_history.append(vl)
            self.train_acc_history.append(ta)
            self.test_acc_history.append(va)

            if verbose and (epoch % max(1, self.training_epochs // 10) == 0 or epoch == 1):
                print(
                    f"Эпоха {epoch:4d}/{self.training_epochs}  "
                    f"Потери: train={tl:.4f}  test={vl:.4f}  "
                    f"Точность: train={ta:.4f}  test={va:.4f}"
                )

        return self


# ─────────────────────────────────────────────
# 3.3. Импорт исходных данных
# ─────────────────────────────────────────────

def load_mnist_csv(filepath: str) -> np.ndarray:
    """Загрузка MNIST из CSV (первый столбец — метка, остальные — пиксели)."""
    return np.loadtxt(filepath, delimiter=",", skiprows=0)


# ─────────────────────────────────────────────
# 3.4. Подготовка наборов данных
# ─────────────────────────────────────────────

def prepare_data(raw: np.ndarray, n_classes: int = 10):
    """
    raw[:, 0]  → метки классов
    raw[:, 1:] → пиксели (0…255) → нормализуем в [0, 1]
    Возвращает X (m, 784), Y (m, 10) — one-hot encoding.
    """
    labels = raw[:, 0].astype(int)
    X = raw[:, 1:] / 255.0

    Y = np.zeros((len(labels), n_classes))
    Y[np.arange(len(labels)), labels] = 1.0

    return X, Y


# ─────────────────────────────────────────────
# Вспомогательная функция: построение графиков
# ─────────────────────────────────────────────

def plot_training(model: MLP, title: str, save_path: str = None):
    epochs = range(1, len(model.train_acc_history) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=13)

    # Точность
    ax = axes[0]
    ax.plot(epochs, model.train_acc_history, label="Train accuracy")
    ax.plot(epochs, model.test_acc_history,  label="Test accuracy", linestyle="--")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Точность")
    ax.set_title("Точность (Accuracy)")
    ax.legend()
    ax.grid(True, alpha=0.4)

    # Потери
    ax = axes[1]
    ax.plot(epochs, model.train_loss_history, label="Train loss")
    ax.plot(epochs, model.test_loss_history,  label="Test loss", linestyle="--")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Потери (Cross-Entropy)")
    ax.set_title("Потери (Loss)")
    ax.legend()
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  График сохранён: {save_path}")
    plt.show()
    plt.close()


# ─────────────────────────────────────────────
# ОСНОВНОЙ БЛОК
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── Загрузка малых наборов данных ──────────────────────────────────────
    # Ожидаемые файлы: data/mnist_train_100.csv, data/mnist_test_10.csv
    # Если файлов нет — генерируем синтетические данные для демонстрации

    train_path = os.path.join(DATA_DIR, "mnist_train_100.csv")
    test_path  = os.path.join(DATA_DIR, "mnist_test_10.csv")

    if os.path.exists(train_path) and os.path.exists(test_path):
        print("Загрузка малых наборов данных из CSV…")
        raw_train = load_mnist_csv(train_path)
        raw_test  = load_mnist_csv(test_path)
    else:
        print(
            f"ВНИМАНИЕ: файлы {train_path} / {test_path} не найдены.\n"
            "Генерируются синтетические данные (100 обучающих, 10 тестовых)."
        )
        np.random.seed(0)
        # Случайные метки (0–9) + пиксели (0–255)
        fake_train = np.hstack([
            np.random.randint(0, 10, (100, 1)),
            np.random.randint(0, 256, (100, 784)),
        ]).astype(float)
        fake_test = np.hstack([
            np.random.randint(0, 10, (10, 1)),
            np.random.randint(0, 256, (10, 784)),
        ]).astype(float)
        raw_train = fake_train
        raw_test  = fake_test

    X_train, Y_train = prepare_data(raw_train)
    X_test,  Y_test  = prepare_data(raw_test)

    print(f"X_train: {X_train.shape}, Y_train: {Y_train.shape}")
    print(f"X_test:  {X_test.shape},  Y_test:  {Y_test.shape}")

    # ── Общие параметры сети ───────────────────────────────────────────────
    LAYER_SIZES      = [784, 64, 32, 10]
    LEARNING_RATE    = 0.05
    TRAINING_EPOCHS  = 100

    # ═════════════════════════════════════════════════════════════════════
    # ЗАДАНИЕ 1, 2: Обучение при batch_size = 1, 100 (малый датасет)
    # ═════════════════════════════════════════════════════════════════════
    experiments_small = [
        {"batch_size": 1,   "label": "SGD (batch_size=1)"},
        {"batch_size": 100, "label": "GD  (batch_size=100)"},
        {"batch_size": 10,  "label": "MSGD (batch_size=10)"},
        {"batch_size": 25,  "label": "MSGD (batch_size=25)"},
    ]

    print("\n" + "=" * 60)
    print("Малый набор данных (100 обуч., 10 тест.)")
    print("=" * 60)

    for exp in experiments_small:
        bs    = exp["batch_size"]
        label = exp["label"]
        print(f"\n── {label} ──")

        model = MLP(
            layer_sizes=LAYER_SIZES,
            learning_rate=LEARNING_RATE,
            batch_size=bs,
            training_epochs=TRAINING_EPOCHS,
        )
        model.fit(X_train, Y_train, X_test, Y_test, verbose=True)

        final_train_acc = model.train_acc_history[-1]
        final_test_acc  = model.test_acc_history[-1]
        print(f"  Итоговая точность — train: {final_train_acc:.4f}, test: {final_test_acc:.4f}")

        save_name = f"small_{bs}.png"
        plot_training(model, title=label, save_path=os.path.join(RESULTS_DIR, save_name))

    # ═════════════════════════════════════════════════════════════════════
    # ЗАДАНИЕ 3: Большой набор данных (если доступен)
    # ═════════════════════════════════════════════════════════════════════
    big_train_path = os.path.join(DATA_DIR, "mnist_train.csv")
    big_test_path  = os.path.join(DATA_DIR, "mnist_test.csv")

    if os.path.exists(big_train_path) and os.path.exists(big_test_path):
        print("\n" + "=" * 60)
        print("Большой набор данных MNIST")
        print("=" * 60)
        raw_big_train = load_mnist_csv(big_train_path)
        raw_big_test  = load_mnist_csv(big_test_path)
        X_big_train, Y_big_train = prepare_data(raw_big_train)
        X_big_test,  Y_big_test  = prepare_data(raw_big_test)
        print(f"X_big_train: {X_big_train.shape}, X_big_test: {X_big_test.shape}")

        # ── ЗАДАНИЕ 4: Произвольный batch_size (обработка неполных батчей) ──
        # Уже реализована в методе fit() — стратегия 1 из теоретической части.
        # Дополнительная демонстрация с batch_size=37 (не делитель 60000).

        experiments_big = [
            {"batch_size": 1,    "label": "SGD (batch_size=1)",    "epochs": 20},
            {"batch_size": 60000,"label": "GD  (batch_size=full)",  "epochs": 50},
            {"batch_size": 32,   "label": "MSGD (batch_size=32)",   "epochs": 50},
            {"batch_size": 128,  "label": "MSGD (batch_size=128)",  "epochs": 50},
            {"batch_size": 256,  "label": "MSGD (batch_size=256)",  "epochs": 50},
            {"batch_size": 37,   "label": "MSGD (batch_size=37, неполный батч)", "epochs": 50},
        ]

        # ── ЗАДАНИЕ 5: Подбор оптимальных гиперпараметров ──────────────────
        # На большом датасете нашему MSGD достаточно batch_size ≈ 128–256,
        # learning_rate ≈ 0.01–0.05, архитектура [784, 256, 128, 10].
        # Ниже запускаются все эксперименты с выводом сравнительного графика.

        best_acc  = 0.0
        best_label = ""

        acc_summary = {}
        for exp in experiments_big:
            bs     = exp["batch_size"]
            label  = exp["label"]
            epochs = exp["epochs"]
            print(f"\n── {label} ──")

            model = MLP(
                layer_sizes=[784, 256, 128, 10],
                learning_rate=0.05,
                batch_size=bs,
                training_epochs=epochs,
            )
            model.fit(X_big_train, Y_big_train, X_big_test, Y_big_test, verbose=True)

            fa = model.test_acc_history[-1]
            acc_summary[label] = fa
            print(f"  Итоговая точность теста: {fa:.4f}")

            save_name = f"big_{bs}.png"
            plot_training(model, title=label, save_path=os.path.join(RESULTS_DIR, save_name))

            if fa > best_acc:
                best_acc   = fa
                best_label = label

        print("\n── Итоговое сравнение на большом датасете ──")
        for lbl, acc in acc_summary.items():
            marker = " ← ЛУЧШИЙ" if lbl == best_label else ""
            print(f"  {lbl}: {acc:.4f}{marker}")

    else:
        print(
            f"\nБольшой датасет не найден ({big_train_path}).\n"
            "Поместите mnist_train.csv и mnist_test.csv в папку data/ для запуска заданий 3–5."
        )

    print("\nВыполнение завершено. Графики сохранены в папке:", RESULTS_DIR)