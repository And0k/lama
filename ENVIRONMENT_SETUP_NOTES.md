# Примечания по окружению и установке

## Успешные результаты
- Виртуальное окружение: `/workspaces/lama/.venv`
- Python в окружении: `/workspaces/lama/.venv/bin/python` (Python 3.12.3)
- `pytest` доступен и тесты можно запускать.
- Созданы CPU-ориентированные тесты в `tests_cpu/`:
  - `tests_cpu/test_cpu_imports.py`
  - `tests_cpu/test_cpu_model_factory.py`
- `pytest -q tests_cpu` прошел успешно: `3 passed`.
- `pip` восстановлен в виртуальном окружении командой:
  - `/home/codespace/.python/current/bin/python3 -m venv --upgrade /workspaces/lama/.venv`

## Что нужно
- Использовать `pip` через виртуальный интерпретатор:
  - `/workspaces/lama/.venv/bin/python -m pip install <package>`
- Хранить тестовые зависимости в `requirements.txt` или в отдельном `requirements-dev.txt`.
- Перед установкой проверять состояние окружения:
  - `/workspaces/lama/.venv/bin/python -m pip --version`
