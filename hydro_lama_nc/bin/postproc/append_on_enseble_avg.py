from datetime import date
import numpy as np
import pandas as pd
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
import sys

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def aggregate_scalar_by_epochs(event_acc: event_accumulator.EventAccumulator, tag: str):
    """
    Извлекает данные для тега 'tag' из EventAccumulator и группирует их по эпохам.

    Предполагается, что в логгировании шаг (step) соответствует номеру батча,
    и вам нужно агрегировать эти шаги в эпохи.

    Args:
        event_acc: Загруженный EventAccumulator.
        tag: Имя скалярной переменной (например, 'train/loss').

    Returns:
        pandas.DataFrame: DataFrame с двумя колонками: 'epoch' и 'value'.
    """
    # 1. Извлекаем все события для нужного тега [citation:3][citation:5][citation:7]
    #    Каждое событие имеет атрибуты: step, value, wall_time
    events = event_acc.Scalars(tag)

    # 2. Вычисляем, сколько шагов (батчей) в одной эпохе.
    #    Это критически важно и должно быть известно из логики вашего обучения!
    #    Допустим, у вас 100 батчей на эпоху.
    batches_per_epoch = 100

    # 3. Группируем шаги в эпохи и вычисляем среднее
    data = []
    for i in range(0, len(events), batches_per_epoch):
        # Берем срез событий для текущей эпохи
        epoch_events = events[i : i + batches_per_epoch]
        # Если срез не пустой, вычисляем среднее арифметическое значений
        if epoch_events:
            mean_value = np.mean([e.value for e in epoch_events])
            epoch_num = i // batches_per_epoch
            data.append({"epoch": epoch_num, "value": mean_value})

    return pd.DataFrame(data)


# --- Пример использования ---
if __name__ == "__main__":
    # Путь к вашему лог-файлу TensorBoard
    log_path = "path/to/your/events.out.tfevents.*"
    # ── Output directory (date-based, same-day resume) ────────────────────
    outdir = str(Path(PROJECT_ROOT) / "outputs" / f"{date.today()}")

    # Создаем и загружаем аккумулятор
    # Важно! Настройте size_guidance, чтобы загрузить все данные [citation:3]
    ea = event_accumulator.EventAccumulator(
        log_path, size_guidance=event_accumulator.STORE_EVERYTHING_SIZE_GUIDANCE
    )
    ea.Reload()  # Обязательный шаг для загрузки всех данных [citation:1][citation:8]

    # Получаем усредненные данные для нужного тега
    # Например, для тега 'train/loss_step', который вы логгировали на каждом шаге
    if "train/loss_step" in ea.Tags()["scalars"]:
        df_epoch_loss = aggregate_scalar_by_epochs(ea, "train/loss_step")
        print(df_epoch_loss)
    else:
        print("Тег 'train/loss_step' не найден.")
        print("Доступные теги:", ea.Tags()["scalars"])

    # Далее вы можете сохранить df_epoch_loss в CSV или построить график
