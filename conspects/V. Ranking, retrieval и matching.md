Полное руководство по построению поисковых систем, RAG-пайплайнов и рекомендательных движков. От классических алгоритмов до SOTA трансформерных архитектур 2025 года.

---

## Эволюция поисковых систем

Современный поиск — это не просто "найти документ по ключевым словам". Это сложная многоступенчатая система, где каждый этап оптимизирован под свою задачу. Если в 2000-х годах доминировал TF-IDF, то сегодня стандартом стала гибридная архитектура: Sparse (BM25) + Dense (Neural Embeddings) + Reranking (Cross-Encoders).[deepschool+1](https://blog.deepschool.ru/llm/rag-ot-pervoj-versii-k-rabochemu-resheniyu/)​

**Основная проблема:** Как из 100 миллионов документов за 50 миллисекунд выдать топ-10, который реально решит задачу пользователя? Ответ: двухстадийная архитектура.

---

## 1. ФУНДАМЕНТАЛЬНЫЕ КОНЦЕПЦИИ

### 1.1. Information Retrieval Models (Теория поиска)

Исторически существовало три подхода к моделированию релевантности:

#### A. Boolean Retrieval (Булева модель)

Самая ранняя модель. Документы представлены как множества термов. Запрос — это булево выражение.

- _Запрос:_ `(Python AND "machine learning") OR TensorFlow`
    
- _Результат:_ Все документы, удовлетворяющие логике (без ранжирования).
    
- **Используется:** Elasticsearch, правовые базы данных (точный поиск по статьям).
    
- _Минус:_ Нет понятия "более релевантный". Либо документ подходит (1), либо нет (0).
    

#### B. Vector Space Model (Векторная модель)

Документы и запросы — это векторы в пространстве термов.

- Каждый терм (слово) — это измерение.
    
- Вес терма в документе: TF-IDF.
    
- **Релевантность:** Косинусное расстояние между $\vec{q}$ и $\vec{d}$.  
    sim(q,d)=q⃗⋅d⃗∣∣q⃗∣∣⋅∣∣d⃗∣∣sim(q,d)=∣∣q∣∣⋅∣∣d∣∣q⋅d
    

#### C. Probabilistic Model (BM25)

Вопрос: "Какова вероятность, что документ $d$ релевантен запросу $q$?"

- BM25 — это функция ранжирования, основанная на вероятностной модели Robertson & Sparck Jones.
    
- Учитывает насыщение частоты терма (если слово встречается 100 раз, это не в 100 раз важнее, чем 1 раз).
    

---

### 1.2. Two-Stage Retrieval Architecture

Это золотой стандарт для масштабируемых систем (Google, Yandex, OpenAI RAG).[jina+1](https://jina.ai/ru/news/maximizing-search-relevancy-and-rag-accuracy-with-jina-reranker/)​

**Stage 1: Retrieval (Candidate Generation)**

- **Цель:** Быстро отфильтровать 99.99% неподходящих документов.
    
- **Вход:** Запрос $q$, корпус $D$ (10M-1B документов).
    
- **Выход:** Топ-1000 кандидатов.
    
- **Методы:** BM25, Bi-Encoders (Sentence Transformers), ANN Search (HNSW).
    
- **Метрика успеха:** Recall@1000. Если среди 1000 кандидатов есть все релевантные документы, то Recall = 100%.
    

**Stage 2: Ranking (Re-ranking)**

- **Цель:** Отсортировать 1000 так, чтобы топ-10 были идеальными.
    
- **Методы:** Cross-Encoders, LambdaMART, LLM-as-Ranker.
    
- **Метрика успеха:** NDCG@10, Precision@10.
    

**Почему нельзя использовать Cross-Encoder сразу?**  
Cross-Encoder требует Forward Pass для каждой пары (query, doc). Для 10M документов это 10M вызовов BERT (несколько часов). Bi-Encoder можно применить за 1 секунду, потому что эмбеддинги документов предпосчитаны.

---

## 2. RETRIEVAL METHODS (ПОИСК КАНДИДАТОВ)

### 2.1. Sparse Retrieval (Lexical Search)

#### TF-IDF (Term Frequency - Inverse Document Frequency)

Классическая формула взвешивания термов.  
w(t,d)=tf(t,d)⋅idf(t)w(t,d)=tf(t,d)⋅idf(t)  
Где:

- $tf(t, d)$ — частота терма $t$ в документе $d$ (normalized).
    
- $idf(t) = \log \frac{N}{df(t)}$ — обратная частота документа ($N$ — всего документов, $df(t)$ — в скольких встречается терм $t$).
    

**Интуиция:** Слово "the" встречается везде → низкий IDF → неважно. Слово "трансформер" встречается редко → высокий IDF → важно для поиска.

#### BM25 (Best Matching 25) [SOTA Sparse]

Улучшение TF-IDF с **saturation функцией**.  
score(q,d)=∑t∈qIDF(t)⋅tf(t,d)⋅(k1+1)tf(t,d)+k1⋅(1−b+b⋅∣d∣avgdl)score(q,d)=∑t∈qIDF(t)⋅tf(t,d)+k1⋅(1−b+b⋅avgdl∣d∣)tf(t,d)⋅(k1+1)  
Параметры:

- $k_1 \in [1.2, 2.0]$: Контролирует насыщение TF. Если $k_1 = 0$, TF игнорируется.
    
- $b \in $: Нормализация по длине документа. $b=1$ — полная нормализация, $b=0$ — игнорируем длину.
    
- $avgdl$: Средняя длина документа в корпусе.
    

**Когда BM25 лучше Dense:**[deepschool](https://blog.deepschool.ru/llm/rag-ot-pervoj-versii-k-rabochemu-resheniyu/)​

- Запросы с точными совпадениями (артикулы, коды, имена).
    
- Короткие документы (твиты, заголовки).
    
- Домены с большим количеством уникальных терминов (медицина, юриспруденция).
    

---

### 2.2. Dense Retrieval (Semantic Search)

#### Bi-Encoders (Two-Tower Architecture)

Энкодер запроса и энкодер документа работают **независимо**.

1. Запрос $q$ → $\vec{v}_q = Encoder_Q(q)$ (размерность 768 или 384).
    
2. Документ $d$ → $\vec{v}_d = Encoder_D(d)$ (предпосчитан заранее).
    
3. Релевантность: $score(q, d) = CosSim(\vec{v}_q, \vec{v}_d)$.
    

**Почему это быстро?**  
Мы храним эмбеддинги всех документов в векторной базе данных (Qdrant, Milvus). Поиск сводится к ANN (Approximate Nearest Neighbors) — одной операции.

**Топ модели (2024-2025):**

- **Sentence-BERT (SBERT):** Пионер. Обучен на NLI + STS датасетах.
    
- **E5 (Microsoft):** Prefix-based. Запрос начинается с `"query: "`, документ с `"passage: "`.
    
- **BGE (BAAI):** Китайская модель, SOTA на MTEB бенчмарке.
    
- **Voyage AI:** Проприетарная модель (через API), лучшая по качеству.
    

**Обучение Bi-Encoder:**

- **Contrastive Loss (InfoNCE):**  
    L=−log⁡exp⁡(sim(q,d+)/τ)exp⁡(sim(q,d+)/τ)+∑d−exp⁡(sim(q,d−)/τ)L=−logexp(sim(q,d+)/τ)+∑d−exp(sim(q,d−)/τ)exp(sim(q,d+)/τ)  
    Где $d^+$ — позитивный документ, $d^-$ — негативные (hard negatives критичны для качества).
    

---

#### Approximate Nearest Neighbors (ANN)

Как найти ближайших соседей среди 100 миллионов векторов?

**1. HNSW (Hierarchical Navigable Small World)** [Стандарт индустрии]

- Строит **иерархический граф**. На верхнем уровне — мало узлов, связи между далекими регионами. На нижнем — все точки, детальные связи.
    
- **Поиск:** Начинаем с верхнего слоя, "прыгаем" к ближайшему узлу, спускаемся на уровень ниже, повторяем.
    
- **Параметры:**
    
    - `ef_construction`: Размер динамического списка кандидатов при построении (больше = точнее, но медленнее строится).
        
    - `M`: Число связей каждого узла (обычно 16-64).
        
- **Используется:** Qdrant, Milvus, Weaviate.
    

**2. IVF (Inverted File Index)**

- Разбиваем пространство на $N$ ячеек (Voronoi cells) с помощью K-Means.
    
- Каждый вектор попадает в одну ячейку.
    
- **Поиск:** Ищем только в $k$ ближайших ячейках (обычно $k=10-50$).
    
- **Product Quantization (PQ):** Дополнительная компрессия. Вектор 768-dim сжимается до 64 байт.
    
- **Используется:** Faiss (Facebook AI Similarity Search).
    

**3. ScaNN (Google)**

- Комбинация ANN + квантизация + re-scoring.
    
- Сначала ищем грубо (квантованные векторы), затем уточняем (full-precision для топ-100).
    

---

### 2.3. Hybrid Search (Sparse + Dense)[jina](https://jina.ai/ru/news/maximizing-search-relevancy-and-rag-accuracy-with-jina-reranker/)​

**Проблема:** BM25 хорош для keyword matching, Dense — для семантики. Как их объединить?

#### Reciprocal Rank Fusion (RRF)

Простой и эффективный метод.  
RRF(d)=∑r∈{BM25,Dense}1k+rankr(d)RRF(d)=∑r∈{BM25,Dense}k+rankr(d)1  
Где $k=60$ (константа из статьи авторов), $rank_r(d)$ — позиция документа в списке метода $r$.

**Пример:**

- BM25: [doc1, doc5, doc3] → $rank_{BM25}(doc1) = 1$
    
- Dense: [doc3, doc1, doc7] → $rank_{Dense}(doc1) = 2$
    
- $RRF(doc1) = \frac{1}{60+1} + \frac{1}{60+2} \approx 0.032$
    

**Преимущества:**

- Не нужна нормализация скоров (работает только с рангами).
    
- Устойчив к выбросам (один метод ошибся — второй компенсирует).
    

#### Learned Fusion (LTR Approach)

Обучаем XGBoost/CatBoost с фичами:

- `bm25_score`
    
- `dense_score`
    
- `doc_length`
    
- `exact_match_count` (сколько слов запроса есть в документе)
    
- `query_doc_cosine`
    

Модель учится оптимальным весам для каждой фичи.

---

## 3. RANKING & RE-RANKING (ДОВОДКА ПОРЯДКА)

### 3.1. Neural Ranking Architectures

#### A. Cross-Encoders (BERT Reranker)[jina](https://jina.ai/ru/news/maximizing-search-relevancy-and-rag-accuracy-with-jina-reranker/)​

**Идея:** Подаем пару (query, document) как **одну последовательность** в BERT.

```text
[CLS] query tokens [SEP] document tokens [SEP]
```

Self-Attention видит взаимодействия между всеми словами запроса и документа (в отличие от Bi-Encoder, где они кодируются раздельно).

**Архитектура:**

- BERT → [CLS] token → Linear(1) → $score \in \mathbb{R}$
    

**Плюсы:**

- Самая высокая точность (state-of-the-art на MS MARCO).
    

**Минусы:**

- Медленно: нужен Forward Pass для каждой пары (query, doc).
    
- Нельзя предпосчитать: эмбеддинг зависит от обоих входов одновременно.
    

**Production Pipeline:**

1. Bi-Encoder: 10M docs → 1000 кандидатов (1 сек).
    
2. Cross-Encoder: 1000 кандидатов → топ-10 (0.5 сек).
    

#### B. ColBERT (Late Interaction) [SOTA Balance]

"Золотая середина" между Bi и Cross.

**Механизм:**

1. Запрос $q$ → Набор векторов токенов $E_q = {e_1^q, ..., e_n^q}$ (каждый токен → 128-dim вектор).
    
2. Документ $d$ → $E_d = {e_1^d, ..., e_m^d}$ (предпосчитан).
    
3. **MaxSim:** Для каждого токена запроса находим максимальную схожесть с любым токеном документа.  
    score(q,d)=∑i=1∣q∣max⁡j=1∣d∣CosSim(eiq,ejd)score(q,d)=∑i=1∣q∣maxj=1∣d∣CosSim(eiq,ejd)
    

**Интуиция:** Каждое слово запроса должно "найти себя" в документе. Если слово "Python" в запросе сильно схоже с "Python" в документе, это добавляет в скор.

**Преимущества:**

- Можно предпосчитать $E_d$ (как в Bi-Encoder).
    
- Точность выше, чем у Bi-Encoder (близка к Cross-Encoder).
    
- Компактное хранилище (каждый токен → 128 dim вместо full 768-dim вектора документа).
    

---

### 3.2. Learning to Rank (LTR)

Классический ML подход: обучаем модель (XGBoost, CatBoost, LightGBM) предсказывать релевантность.

#### Типы лоссов:

**1. Pointwise (Независимое предсказание)**

- Задача: Для каждого документа предсказать скор релевантности (0-5) или бинарную метку (0/1).
    
- Лосс: MSE или Binary Cross-Entropy.
    
- _Минус:_ Не учитывает порядок документов в списке. Если мы ошибочно дали релевантному документу скор 0.4, а нерелевантному 0.6, и оба оказались на правильных местах, модель думает, что всё ОК.
    

**2. Pairwise (Сравнение пар)**

- Задача: Для пары $(d_i, d_j)$ предсказать, что $d_i$ более релевантен, чем $d_j$.
    
- **RankNet:** Лосс на вероятность правильного порядка.  
    L=−log⁡σ(si−sj)L=−logσ(si−sj)  
    Где $s_i, s_j$ — скоры документов.
    
- **LambdaMART:** Улучшение RankNet. Градиенты взвешиваются на изменение NDCG при swap'е пары документов.
    

**3. Listwise (Оптимизация всего списка)**

- Задача: Оптимизировать метрику (NDCG, MAP) напрямую.
    
- **ListNet:** Cross-Entropy между истинным распределением позиций и предсказанным.
    
- **ApproxNDCG:** Дифференцируемая аппроксимация NDCG (можно считать градиенты).
    

**Фичи для LTR:**

- Query-dependent: Длина запроса, число стоп-слов.
    
- Document-dependent: Длина документа, PageRank, читабельность (Flesch-Kincaid).
    
- Query-Document: BM25 score, Cosine similarity, Exact match ratio, Jaccard similarity.
    

---

### 3.3. LLM-as-a-Reranker

**Идея:** Использовать GPT-4 / Claude 3.5 / Llama 3 для оценки релевантности.

**Промпт:**

```text
Query: "Как обучить нейросеть на PyTorch?" Document 1: [text...] Document 2: [text...] Rank these documents by relevance to the query. Output: [1, 2] or [2, 1].
```

**Результаты:**

- GPT-4 показывает качество выше Cross-Encoder на сложных запросах (reasoning, multi-hop).
    
- Понимает нюансы (сарказм, контекст, доменные знания).
    

**Проблемы:**

- **Дорого:** $0.01 за 1k токенов × 100 документов × 1000 запросов = $1000/день.
    
- **Медленно:** 2-5 секунд на batch.
    
- **Position bias:** Модель предпочитает документы, которые стоят первыми в промпте.
    

**Решение: Distillation**

1. Собираем датасет: запросы + документы + ранги от GPT-4.
    
2. Обучаем маленький Cross-Encoder (MiniLM, 33M параметров) предсказывать эти ранги.
    
3. Получаем 95% качества GPT-4 за 1% стоимости.
    

---

## 4. МЕТРИКИ КАЧЕСТВА (EVALUATION)

### 4.1. Offline Metrics[habr](https://habr.com/ru/articles/948786/)​

#### A. Binary Relevance

**Precision@K:**  
P@K=Relevant docs in top-KKP@K=KRelevant docs in top-K

- _Пример:_ Топ-10, из них 7 релевантны → $P@10 = 0.7$.
    

**Recall@K:**  
R@K=Relevant docs in top-KTotal relevant docsR@K=Total relevant docsRelevant docs in top-K

- _Пример:_ Всего 20 релевантных, в топ-10 нашли 7 → $R@10 = 0.35$.
    

**F1@K:** Гармоническое среднее Precision и Recall.

**MAP (Mean Average Precision):**  
Для каждого релевантного документа считаем Precision на его позиции, усредняем.  
AP=1∣Rel∣∑k=1NP@k⋅rel(k)AP=∣Rel∣1∑k=1NP@k⋅rel(k)  
Где $rel(k) = 1$, если документ на позиции $k$ релевантен.

---

#### B. Ranked Relevance (Порядок важен)

**MRR (Mean Reciprocal Rank):**  
MRR=1∣Q∣∑i=1∣Q∣1rankiMRR=∣Q∣1∑i=1∣Q∣ranki1  
Где $rank_i$ — позиция _первого_ релевантного документа для запроса $i$.

- _Используется:_ QA системы, где один правильный ответ (например, "Столица Франции?").
    
- _Интерпретация:_ Если первый правильный ответ на 2-м месте, вклад запроса = $0.5$. Если на 5-м → $0.2$.
    

**NDCG@K (Normalized Discounted Cumulative Gain)** [GOLD STANDARD]  
Учитывает:

1. Градации релевантности (не просто 0/1, а 0-5).
    
2. Позицию (ошибка на 1-м месте "дороже", чем на 10-м).
    

DCG@K=∑i=1K2reli−1log⁡2(i+1)DCG@K=∑i=1Klog2(i+1)2reli−1  
NDCG@K=DCG@KIDCG@KNDCG@K=IDCG@KDCG@K  
Где $IDCG$ — идеальный DCG (если бы документы были отсортированы идеально).

- _Пример:_ Релевантности на позициях.
    
- $DCG = \frac{2^3-1}{\log_2(2)} + \frac{2^2-1}{\log_2(3)} + ... = 7 + 1.89 + 4.42 + 0 + 0.38 = 13.69$
    
- $IDCG$ (если бы порядок был ) = 15.3
    
- $NDCG = 13.69 / 15.3 = 0.895$
    

---

### 4.2. Online Metrics (Production)

**CTR (Click-Through Rate):**  
CTR=ClicksImpressionsCTR=ImpressionsClicks  
Доля запросов, где пользователь кликнул на результат.

**Dwell Time:**  
Среднее время, проведенное на странице результата. Proxy для "нашел ли пользователь ответ".

**Zero Results Rate:**  
ZRR=Queries with 0 resultsTotal queriesZRR=Total queriesQueries with 0 results  
Критическая метрика. Если ZRR > 10%, нужен Query Expansion или Spell Correction.

**SERP Abandonment:**  
Пользователь не кликнул ни на один результат (возможно, нашел ответ в сниппете, или ничего не подошло).

---

## 5. ADVANCED TOPICS (ПРОДВИНУТЫЕ ТЕХНИКИ)

### 5.1. Hard Negatives Mining

**Проблема:** Если обучать Bi-Encoder на случайных негативах (берем random документы из корпуса), модель быстро переобучается. Она учится отличать "очевидно неподходящие" документы, но не учится тонким различиям.[deepschool](https://blog.deepschool.ru/llm/rag-ot-pervoj-versii-k-rabochemu-resheniyu/)​

**Hard Negative:** Документ, который:

- Похож на запрос (высокий BM25 score или cosine similarity).
    
- Но **НЕ релевантен** (нет в ground truth).
    

**Методы майнинга:**

1. **In-batch Negatives:** Другие документы в батче (если батч = 32 пары (q, d+), то для запроса $q_1$ негативами будут $d_2, ..., d_{32}$).
    
2. **BM25 Negatives:** Берем топ-100 по BM25, исключаем ground truth.
    
3. **ANN Negatives:** Используем текущую версию модели, ищем ближайших соседей к запросу, исключаем ground truth.
    

**Результат:** Качество модели (NDCG) растет на 5-15%.

---

### 5.2. Query Understanding

Перед тем как искать, нужно понять запрос.

**Spell Correction:**

- "айфон 15 про" → "iPhone 15 Pro"
    
- Методы: Levenshtein distance + frequency dictionary, Neural Spell Checker (T5-based).
    

**Query Segmentation:**

- "купить ноутбук asus 16gb" → `[action: купить] [category: ноутбук] [brand: asus] [spec: 16gb]`
    
- Используется для фильтров (в БД ищем только ноутбуки ASUS с RAM 16GB).
    

**Intent Classification:**

- _Navigational:_ Пользователь ищет конкретный сайт ("Apple официальный сайт").
    
- _Informational:_ Хочет узнать факт ("Что такое трансформер?").
    
- _Transactional:_ Готов купить ("купить iPhone 15").
    

---

### 5.3. Multi-modal Retrieval

**Задача:** Запрос может быть текст + картинка, а документы могут содержать видео + текст.

**Архитектура (CLIP-based):**

1. Image Encoder (Vision Transformer) → $\vec{v}_{img}$
    
2. Text Encoder (BERT) → $\vec{v}_{text}$
    
3. Проекция в общее пространство (размерность 512).
    
4. Поиск: $CosSim(\vec{v}_{query}, \vec{v}_{doc})$
    

**Примеры:**

- Поиск товара по фото (обратный поиск).
    
- "Покажи мне кроссовки как на этой картинке, но красные" (hybrid query).
    

---

### 5.4. RAG-Specific: Chunk Strategy[deepschool](https://blog.deepschool.ru/llm/rag-ot-pervoj-versii-k-rabochemu-resheniyu/)​

**Проблема:** Документ в 10k токенов не влезет в эмбеддинг. Нужно разрезать (chunking).

**Методы:**

1. **Fixed-size:** 512 токенов, overlap 50. Просто, но может разорвать смысловой блок.
    
2. **Semantic Chunking:** Режем по параграфам. Если embedding distance между параграфами скачет (> threshold), это граница чанка.
    
3. **Hierarchical (Parent-Child):**
    
    - Храним мелкие чанки (128 токенов) для поиска.
        
    - При retrieval возвращаем "родительский" чанк (512 токенов) для контекста LLM.
        

---

### 5.5. Context Compression для RAG

**Проблема:** Топ-10 чанков по 500 токенов = 5000 токенов контекста. Это дорого и может не влезть в окно LLM.[deepschool](https://blog.deepschool.ru/llm/rag-ot-pervoj-versii-k-rabochemu-resheniyu/)​

**Решения:**

1. **LongLLMLingua:** Модель-компрессор. Удаляет "лишние" токены, сохраняя суть.
    
    - "The company was founded in 1998 in California" → "Founded 1998 California".
        
2. **Extractive Summarization:** BERT-based модель выделяет key sentences.
    
3. **Reranking Context:** Cross-Encoder сортирует чанки, берем топ-3 вместо топ-10.
    

---

## 6. PRODUCTION & SCALABILITY

### 6.1. Vector Databases (Сравнение)

|Параметр|Qdrant|Milvus|Weaviate|Pinecone|
|---|---|---|---|---|
|**Язык**|Rust|C++/Go|Go|Managed (Python SDK)|
|**ANN**|HNSW|HNSW/IVF/PQ|HNSW|Проприетарный|
|**Гибридный поиск**|✅ (Sparse + Dense)|❌ (только Dense)|✅|❌|
|**Фильтрация**|✅ (по метаданным)|✅|✅|✅|
|**Масштабирование**|Vertical + Horizontal|Distributed (Kubernetes)|Horizontal|Auto|
|**Стоимость**|Open-source / Cloud|Open-source / Cloud|Open-source / Cloud|Дорого ($70+/мес)|

**Выбор:**

- Qdrant — если нужен гибридный поиск и фильтрация.
    
- Milvus — для огромных масштабов (> 1B векторов).
    
- Pinecone — если бюджет не ограничен и нужна простота.
    

---

### 6.2. Latency Optimization

**1. Model Quantization:**

- FP32 → INT8: скорость ×4, качество -1%.
    
- ONNX Runtime поддерживает квантизацию из коробки.
    

**2. Batch Inference:**

- Вместо обработки запросов по одному, группируем в батчи (8-32).
    
- GPU utilization растет с 20% до 80%.
    

**3. Cached Embeddings:**

- Для популярных запросов храним предпосчитанные эмбеддинги в Redis.
    

**4. Model Distillation:**

- BERT-large (340M) → DistilBERT (66M): скорость ×2, качество -3%.
    

---

## ИТОГОВАЯ СВОДКА (CHEAT SHEET)

### Когда что использовать?

|Задача|Метод Stage 1|Метод Stage 2|Почему?|
|---|---|---|---|
|**FAQ / RAG (малый корпус < 10k)**|BM25 + SBERT|Cross-Encoder|Простота, высокая точность [jina](https://jina.ai/ru/news/maximizing-search-relevancy-and-rag-accuracy-with-jina-reranker/)​|
|**Поиск по базе знаний (> 1M docs)**|BGE Bi-Encoder + HNSW|ColBERT|Баланс скорость/точность|
|**E-commerce (артикулы, точные совпадения)**|BM25|LambdaMART (LTR)|Keyword matching важен [deepschool](https://blog.deepschool.ru/llm/rag-ot-pervoj-versii-k-rabochemu-resheniyu/)​|
|**Семантический поиск (синонимы, перефразы)**|E5 Bi-Encoder|Cross-Encoder|Dense Retrieval сильнее|
|**Огромный масштаб (> 100M docs)**|Faiss IVF-PQ|Mini-Cross-Encoder (distilled)|Компрессия + скорость|

### Метрики для разных задач

|Задача|Ключевая метрика|Почему?|
|---|---|---|
|QA (один правильный ответ)|MRR|Важна позиция первого правильного|
|Поиск (несколько релевантных)|NDCG@10|Учитывает порядок и градации [habr](https://habr.com/ru/articles/948786/)​|
|RAG (нужны все факты)|Recall@K|Пропуск факта → галлюцинация LLM|
|E-commerce (продажи)|CTR, Revenue per Search|Бизнес-метрики важнее оффлайн|

---

### ЗАКЛЮЧЕНИЕ

Современный поиск — это сложная инженерная система, где каждый компонент решает свою подзадачу. BM25 не умер — он дополняет нейросетевые методы в гибридном поиске. Cross-Encoders дают максимальную точность, но применяются только на финальной стадии. ColBERT показывает, что можно найти "золотую середину" между скоростью Bi-Encoder и точностью Cross-Encoder.[jina+1](https://jina.ai/ru/news/maximizing-search-relevancy-and-rag-accuracy-with-jina-reranker/)​

Для продакшена критичны не только алгоритмы, но и инфраструктура: векторные базы, квантизация моделей, кеширование. A/B тесты показывают, что улучшение NDCG на 2% может дать рост CTR на 10%.

**Главный вывод:** Не существует "одного лучшего метода". Правильная архитектура — это гибридный пайплайн, где каждый этап оптимизирован под свои метрики и constraints.