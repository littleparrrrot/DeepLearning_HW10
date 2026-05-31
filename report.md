# Report

## Track

Выбранный трек:

```text
A
```

## Что реализовано

- [+] dataset.py
- [+] processor.py
- [+] model.py
- [+] train.py
- [+] benchmark.py

## Конфигурация

```text
config path: configs/track_a_cpu.yaml
seed: 42
device: CPU
dtype: float32
max_steps: 3
batch size: 1
```

Для benchmark использовался конфиг:

```text
config path: configs/inference_math.yaml
toy: true
```

## Результаты

```text
public tests: pytest -q tests_public проходит без изменения публичных тестов 
fast-train: python -m hw.train --config configs/track_a_cpu.yaml --fast-train 
train loss: 8.5751 
benchmark: python -m hw.benchmark --config configs/inference_math.yaml --toy 
benchmark accuracy overall: 0.0 
benchmark accuracy by subject: 
    subject/geometry: 0.0 
    subject/plots: 0.0
```

## Использованные ресурсы

```text
CPU/GPU: CPU
VRAM: не использовалась
время обучения: несколько секунд в режиме --fast-train
```

## Анализ ошибок

Приведите 3 ошибки модели:

1. Модель не выдаёт корректный вариант ответа A/B/C/D. В треке A использовалась mocked-модель, поэтому генерация не является полноценным правильным рассуждением над задачей.
2. Модель не умеет реально интерпретировать графики и геометрические изображения. Vision encoder в этом треке используется только для проверки формы тензоров и прохождения данных через adapter
3. Accuracy на toy-dev равна 0.0, потому что adapter не обучался полноценно, а запуск --fast-train нужен только для проверки, что loss конечный и optimizer может делать шаги

## Комментарии

Самым сложным было согласовать все модули, правильно проследить, откуда приходят данные, в каком формате, и правильно их преобразовать для дальнейшей работы

В рамках этого трека я не обучала большую VLM и не использовала GPU (так как реализовывала локально на ноутбуке и не обладаю своей GPU), LoRA или MathVista. Основная цель была в корректной реализации базового трека и прохождении тестов

Если продолжать работу над этой задачей, то еще много чего можно улучшить: использовать GPU и пойти по более продвинутому треку, чтобы уже полноценно обучить модель и получить реальные предсказания


## Критерии оценивания

См. файл [`GRADING.md`](GRADING.md).
