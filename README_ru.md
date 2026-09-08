# PDF Q&A Agent

Интеллектуальный агент, отвечающий на вопросы пользователя по содержимому PDF-документа
(эссе, научная статья, отчёт). Автоматически:

- извлекает текст и структуру из PDF;
- обрабатывает документы разного качества — **text-based PDF** и **сканы** (постранично, в
  рамках одного документа могут смешиваться оба типа страниц);
- использует извлечённые данные для осмысленных, обоснованных цитатами ответов на вопросы
  пользователя, честно сообщая, если ответа в документе нет.

## Как это работает

Для каждой страницы PDF:

1. Сначала пробуем текстовый слой (`pypdfium2`) — дёшево и точно для text-based PDF.
2. Если текстового слоя нет или он короче `TEXT_LAYER_MIN_CHARS` (страница — скан или таблица,
   отрисованная как изображение), страница рендерится в PNG и распознаётся мультимодальной
   моделью Claude (VLM OCR) с промптом, сохраняющим структуру документа в markdown.
3. Извлечённый текст всех страниц собирается в единый контекст документа.
4. Вопрос пользователя вместе с контекстом отправляется модели через `client.messages.parse`
   с pydantic-схемой `AgentAnswer` — модель обязана вернуть структурированный
   ответ: `answer`, `answerable` (найден ли ответ в документе) и `citations` (страница + дословная
   цитата на каждый использованный факт).

Все сетевые вызовы к Anthropic API (и OCR, и сам вопрос-ответ) обёрнуты **retry с
экспоненциальным backoff и джиттером** (`pdf_qa_agent/retry.py`): временные сетевые ошибки,
таймауты, rate limit и 5xx ретраятся; ошибки валидации запроса — нет, т.к. повтор не поможет.

## Структура проекта

```
pdf_qa_agent/          пакет агента
  schemas.py             pydantic-модели (PageContent, AgentAnswer, Citation, ...)
  retry.py               retry/exponential backoff декоратор
  extraction.py          извлечение PDF: текстовый слой + VLM OCR fallback
  agent.py               PDFQAAgent: load() -> ask()
  cli.py                 CLI-обёртка (разовый вопрос или интерактивный режим)

eval/
  goldenset.json          14 вопросов к 3 реальным документам разного качества
  run_eval.py             прогоняет агента по golden set, считает метрики
  eval_report.json        (генерируется run_eval.py)

pdf-files/               исходные PDF из golden set (в .gitignore)
tests/                   pytest-тесты, без сети (все вызовы Anthropic API замоканы)
pdf_qa_agent.py          CLI-обёртка, сохраняющая обратную совместимость
```

## Установка

```bash
cd /Users/zhadyrazhan/Documents/llm-engineer/pdf-qa-agent
pip install -r requirements.txt
```

Нужен ключ Anthropic API: `export ANTHROPIC_API_KEY=...` (или активный профиль `ant auth login`).
Без ключа работают только тесты (`tests/`) — они не обращаются к сети.

## Использование

```bash
# разовый вопрос
python3 -m pdf_qa_agent.cli path/to/document.pdf -q "О чём этот документ?"

# интерактивный режим
python3 -m pdf_qa_agent.cli path/to/document.pdf
```

Программный интерфейс:

```python
from pdf_qa_agent.agent import PDFQAAgent

agent = PDFQAAgent()
agent.load("report.pdf")
result = agent.ask("Какая выручка компании за 2024 год?")
print(result.answer, result.answerable, result.citations)
```

## Тесты

```bash
python3 -m pytest tests/ -v
```

Тесты полностью офлайн (без `ANTHROPIC_API_KEY`):

- `test_retry.py` — retry/backoff и обработка временных и постоянных ошибок API;
- `test_extraction.py` — текстовый слой, OCR fallback, PNG-запрос к VLM и сборка контекста;
- `test_agent.py` — загрузка контекста, структурированный ответ, citations и retry;
- `test_schemas.py`, `test_goldenset.py` — pydantic-схемы и структура golden set.

## Eval / Golden set

`eval/goldenset.json` — 14 вопросов к трём реальным документам из
`pdf-files/`. Пути в golden set разрешаются относительно
`--pdf-root`, по умолчанию используется `pdf-files/` рядом с каталогом проекта.

Набор охватывает оба режима извлечения и оба исхода (`answerable` / `not answerable`):

| Документ | Тип | Что проверяется |
|---|---|---|
| `kaztelecom.pdf` | text-based финансовый отчёт со сканированными таблицами | смешанное извлечение и точные числа |
| `G25_Central_Asia_reading_(1)-1-4.pdf` | 4-страничный скан книги | VLM OCR на английском тексте и отказ вне контекста |
| `1_-_mu-stat-grafiki_-2002-10-11.pdf` | 2 страницы с диаграммами | подписи/данные диаграмм и отказ на отсутствующий показатель |

Три вопроса намеренно не имеют ответа в документах. Они проверяют, что агент не галлюцинирует,
а возвращает `answerable=false`.

Запуск:

```bash
python3 eval/run_eval.py
python3 eval/run_eval.py --ids book-author,charts-fig7-subject
python3 eval/run_eval.py --pdf-root /custom/path/to/pdf-files
```

Метрики сохраняются в `eval/eval_report.json` и выводятся в консоль:

- `answerable_accuracy` — доля совпадений флага `answerable` с ожидаемым значением;
- `keyword_accuracy` — для отвечаемых вопросов наличие хотя бы одного ожидаемого слова/числа;
- `citation_accuracy` — наличие хотя бы одной ожидаемой страницы в citations;
- `pass_rate` — доля вопросов, прошедших проверки answerability и keywords.

Скрипт завершается с ненулевым кодом, если `pass_rate < 1.0`, поэтому его можно использовать в CI.

## Известные ограничения

- Контекст документа собирается целиком, без chunking/embeddings-ретривала; для очень больших
  документов потребуется добавить retrieval.
- Качество OCR зависит от качества скана; плотные числовые таблицы — наиболее сложный случай.
- Зашифрованные, повреждённые или неподдерживаемые PDF могут не загрузиться.
