# Локальные данные

Бинарные данные не входят в Git-репозиторий. Стандартный набор загружается из
[Zenodo record 10205004](https://zenodo.org/records/10205004):

```bash
uv run python scripts/download_data.py
uv run python scripts/download_data.py --check
```

Downloader получает метаданные `data.zip` через Zenodo API, проверяет checksum,
извлекает только ожидаемые файлы `data/human/` и валидирует массивы через
`numpy.load(..., mmap_mode="r")`.

## Ожидаемые файлы

| Файл | Форма |
|---|---|
| `human/connectivity.npy` | `(1015, 1015, 70)` |
| `human/consensus_0.npy` ... `human/consensus_5.npy` | `(1015, 1015)` |
| `human/coords.npy` | `(1015, 3)` |
| `human/cortical.npy` | `(1015,)` |
| `human/hemiid.npy` | `(1015,)` |
| `human/rsn_mapping.npy` | `(1015,)` |

Каталоги `task_cache/` и `null_models/` оставлены для локальных кэшей задач и
предварительно вычисленных нуль-моделей. Их содержимое также игнорируется Git.

Для другого расположения используйте:

```bash
uv run python scripts/download_data.py --output-dir /path/to/human
```

После этого задайте `CONN2RES_DATA_DIR=/path/to/human` в окружении запуска.
Для принудительной замены неполного или повреждённого набора передайте
`--force`.

Дополнительные индивидуальные многомасштабные коннектомы доступны в
[Zenodo record 2872624](https://zenodo.org/records/2872624), но не требуются
для стандартных сценариев этого репозитория.
