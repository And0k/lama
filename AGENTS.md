Не ломай существующие конфиги: добавляй новые YAML в configs/ вместо правки оригинальных, чтобы можно было откатиться.
Hydra и _target_: если используешь _target_, убедись, что путь корректен и модуль импортируется (проверь python -c "import <module>; print('ok')").
in_channels: вычисляй из dataset (лучше dataset[0]['image'].shape[0]) и только потом инстанцируй модель.
num_workers и netCDF: при ошибках открывай netCDF внутри __getitem__ или используйте worker_init_fn.
Тесты: всегда сначала batch_size=1, max_epochs=1.
Документируй изменения в configs/README.md — чтобы другие могли повторить.