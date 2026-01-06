Полное руководство по NLP. От классических алгоритмов до SOTA трансформерных архитектур.

---
## TF-IDF & ML

---
#### Вопрос: Напишите TF-IDF с нуля?

**Ответ:**

TF-IDF (Term Frequency – Inverse Document Frequency) — это метод векторизации текста, который представляет слова числовыми значениями, сохраняя контекст и значимость. Основная идея: частые слова получают высокие веса, редкие слова в корпусе — тоже.

**Формула:**

TF-IDF(t,d)=TF(t,d)×IDF(t,d)TF-IDF(t,d)=TF(t,d)×IDF(t,d)

где:

- **TF (Term Frequency)** — частота слова в документе:  
    TF(t,d)=количество слова t в документе dвсего слов в документе dTF(t,d)=всего слов в документе dколичество слова t в документе d
    
- **IDF (Inverse Document Frequency)** — обратная частота документа:  
    IDF(t)=log⁡(Ndf(t)+1)IDF(t)=log(df(t)+1N)
    

где N — количество документов в корпусе, df(t) — количество документов со словом t (+1 для избежания деления на 0).

**Пошаговая реализация:**

```python
import numpy as np
from nltk.tokenize import word_tokenize
from typing import List, Dict, Tuple

def create_counts(texts: List[str]) -> Tuple[List, set, int, Dict]:
    """Препроцессинг: токенизация, создание словаря индексов"""
    sentences = []
    word_set = set()
    
    # Разбиваем текст на слова, переводим в нижний регистр, оставляем только буквы
    for sent in texts:
        tokens = [token.lower() for token in word_tokenize(sent) if token.isalpha()]
        sentences.append(tokens)
        word_set.update(tokens)
    
    total_documents = len(sentences)
    index_dict = {word: idx for idx, word in enumerate(sorted(word_set))}
    
    return sentences, word_set, total_documents, index_dict

def count_dict(sentences: List[List[str]], word_set: set) -> Dict[str, int]:
    """Подсчитываем, в скольких документах встречается каждое слово"""
    return {
        word: sum(1 for sent in sentences if word in sent)
        for word in word_set
    }

def termfreq(document: List[str], word: str) -> float:
    """Вычисляем TF для слова в документе"""
    return document.count(word) / len(document) if document else 0.0

def inverse_doc_freq(word: str, total_documents: int, word_count: Dict[str, int]) -> float:
    """Вычисляем IDF для слова"""
    word_occurrence = word_count.get(word, 0) + 1  # +1 для сглаживания
    return np.log(total_documents / word_occurrence)

def tf_idf(
    sentence: List[str],
    vector_shape: int,
    index_dict: Dict[str, int],
    total_documents: int,
    word_count: Dict[str, int]
) -> np.ndarray:
    """Получаем TF-IDF вектор предложения"""
    tf_idf_vec = np.zeros(vector_shape)
    
    for word in set(sentence):  # Используем set для уникальных слов
        if word in index_dict:
            tf = termfreq(sentence, word)
            idf = inverse_doc_freq(word, total_documents, word_count)
            tf_idf_vec[index_dict[word]] = tf * idf
    
    return tf_idf_vec

def create_vectors(texts: List[str]) -> Tuple[np.ndarray, Dict[str, int]]:
    """Главная функция: преобразуем все тексты в TF-IDF векторы"""
    sentences, word_set, total_docs, index_dict = create_counts(texts)
    word_count = count_dict(sentences, word_set)
    
    vectors = [
        tf_idf(sent, len(word_set), index_dict, total_docs, word_count)
        for sent in sentences
    ]
    
    return np.array(vectors), index_dict

# Пример использования
if __name__ == "__main__":
    sample_texts = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',
    ]
    
    vectors, word2id = create_vectors(sample_texts)
    print(f"Shape: {vectors.shape}")  # (4, количество уникальных слов)
    print(f"First document TF-IDF:\n{vectors[0]}")
    print(f"Word to index mapping: {word2id}")
```

**Как это работает:**

1. **Препроцессинг** — токенизируем текст, создаем словарь уникальных слов
    
2. **Подсчет документов** — для каждого слова считаем, в скольких документах оно встречается
    
3. **TF расчет** — для каждого слова в документе делим количество появлений на длину документа
    
4. **IDF расчет** — берем логарифм отношения всех документов к документам со словом
    
5. **TF-IDF** — умножаем TF на IDF и помещаем результат в вектор по индексу слова
    

---

#### Вопрос: Что такое нормализация в TF-IDF?

**Ответ:**

Нормализация в TF-IDF решает проблему **длины документа**. В длинных документах TF значения естественно выше, потому что слов просто больше.

**Проблема без нормализации:**

- Слово "кот" встречается 10 раз в тексте из 1000 слов
    
- Слово "кот" встречается 5 раз в тексте из 500 слов
    

Без нормализации первый вариант получит выше TF, хотя относительная важность одинакова.

**Решение — L2 нормализация вектора:**

После подсчета всех TF-IDF значений вектор нормализуется:

normalized_vector=TF-IDF вектор∥TF-IDF вектор∥normalized_vector=∥TF-IDF вектор∥TF-IDF вектор

где $|\text{v}| = \sqrt{\sum_{i=1}^{n} v_i^2}$ — норма L2.

**Практическое преимущество:**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
# norm='l2' (по умолчанию) 
vectorizer = TfidfVectorizer(norm='l2') 
vectors = vectorizer.fit_transform(texts)

# После L2 нормализации можно использовать dot product вместо косинусной близости # 
cosine_similarity(vec1, vec2) = vec1 · vec2 (просто скалярное произведение!)
```

**Когда использовать:**

- **L2 нормализация** — когда сравниваете тексты по косинусной близости
    
- **Без нормализации** — если нужна абсолютная "тяжесть" слова в документе
    

---

#### Вопрос: Зачем знать про TF-IDF в наше время и как использовать в сложных моделях?

**Ответ:**

TF-IDF остается актуальным по нескольким причинам:

**1. Быстрая проверка гипотез**

Это один из самых быстрых методов векторизации. Перед использованием BERT или других нейросетей:

- Быстро проверяете гипотезу о классификации
    
- Если идея не работает на TF-IDF, то вряд ли работает на нейросетях
    
- Экономит вычислительные ресурсы на этапе исследования
    

**2. Extraction важных признаков**

Каждое значение в векторе = важность слова. Используется для:

- **Topic Modeling** — берете top-N слов по TF-IDF для каждого топика
    
- **Feature Importance** — какие слова определяют класс? TF-IDF дает ответ
    
- **Keyword Extraction** — автоматический поиск ключевых слов
    

**3. Комбинирование с нейросетями**

```python
# Пример: комбинируем TF-IDF + BERT
import numpy as np

# Преобразуем текст в TF-IDF вектор
tfidf_vec = tfidf_vectorizer.transform([text])  # размер: (1, vocab_size)

# Получаем эмбеддинг BERT
bert_vec = model.encode(text)  # размер: (1, 768)

# Конкатенируем векторы
combined = np.concatenate([
    tfidf_vec.toarray(),  # TF-IDF признаки
    bert_vec              # BERT эмбеддинги
], axis=1)

# Альтернатива: взвешенная сумма
# combined = 0.3 * tfidf_vec.toarray() + 0.7 * bert_vec

# Передаем в классификатор
prediction = classifier.predict(combined)
```

Это помогает моделе "видеть" как статистическую структуру слов, так и их глубокую семантику.

**4. Интерпретируемость**

В отличие от BERT, TF-IDF легко объяснить человеку: "вот эти слова определили класс"

---

#### Вопрос: Объясните, как работает Наивный Баес? Для чего вы можете его использовать?

**Ответ:**

Naive Bayes основан на **теореме Байеса**:

P(A∣B)=P(B∣A)⋅P(A)P(B)P(A∣B)=P(B)P(B∣A)⋅P(A)

где:

- **P(A|B)** — апостериорная вероятность (что нам нужно найти)
    
- **P(B|A)** — правдоподобие (likelihood)
    
- **P(A)** — априорная вероятность класса
    
- **P(B)** — полная вероятность доказательства
    

**Классический пример (задача Байеса о партиях):**

На склад поступило:

- Партия 1: 4000 изделий, 20% брака
    
- Партия 2: 6000 изделий, 10% брака
    

Взяли случайно изделие, оно оказалось **стандартным**. Найти вероятность, что оно из партии 1.

Решение:

P(партия 1)=400010000=0.4P(партия 1)=100004000=0.4

P(стандартное | партия 1)=1−0.2=0.8P(стандартное | партия 1)=1−0.2=0.8  
P(стандартное | партия 2)=1−0.1=0.9P(стандартное | партия 2)=1−0.1=0.9

P(стандартное)=0.8⋅0.4+0.9⋅0.6=0.32+0.54=0.86P(стандартное)=0.8⋅0.4+0.9⋅0.6=0.32+0.54=0.86

P(партия 1 | стандартное)=P(стандартное | партия 1)⋅P(партия 1)P(стандартное)=0.8⋅0.40.86=1643≈0.372P(партия 1 | стандартное)=P(стандартное)P(стандартное | партия 1)⋅P(партия 1)=0.860.8⋅0.4=4316≈0.372

**Применение в классификации текстов:**

Для класса **spam / not spam**:

P(spam | слова)=P(слова | spam)⋅P(spam)P(слова)P(spam | слова)=P(слова)P(слова | spam)⋅P(spam)

Предполагаем, что слова независимы (отсюда "Naive"):

P(слова | spam)=P(w1∣spam)⋅P(w2∣spam)⋅...⋅P(wn∣spam)P(слова | spam)=P(w1∣spam)⋅P(w2∣spam)⋅...⋅P(wn∣spam)

**Код:**

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

# Пример данных для обучения
texts = [
    'This is spam email with cheap offers',
    'Meeting tomorrow at 10 AM',
    'Buy now special price limited time',
    'Project deadline extended to Friday'
]
labels = [1, 0, 1, 0]  # 1 = spam, 0 = not spam

# Создаем и обучаем векторайзер
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Создаем и обучаем модель
model = MultinomialNB(alpha=1.0)  # alpha=1.0 - сглаживание Лапласа
model.fit(X, labels)

# Пример предсказания
test_text = ['buy cheap products now']
test_vec = vectorizer.transform(test_text)
prediction = model.predict(test_vec)

print(f"Предсказание: {'SPAM' if prediction[0] == 1 else 'NOT SPAM'}")  # [1] — spam

# Дополнительно: посмотрим вероятности классов
probabilities = model.predict_proba(test_vec)
print(f"Вероятности классов (не спам, спам): {probabilities[0].round(3)}")

# Выведем наиболее важные признаки для спама
feature_names = vectorizer.get_feature_names_out()
coefs = model.feature_log_prob_[1]  # Логарифмы вероятностей для класса спам
top_spam_words = sorted(zip(feature_names, coefs), 
                       key=lambda x: x[1], 
                       reverse=True)[:5]
print("Топ-5 слов, указывающих на спам:", [w[0] for w in top_spam_words])
```

**Когда использовать Naive Bayes:**

- ✅ Быстрая классификация текстов
    
- ✅ Мало данных для обучения
    
- ✅ Нужна интерпретируемость
    
- ❌ Слова явно зависимы (тогда лучше использовать логистическую регрессию или BERT)
    

**⚠️ Важная проблема — нулевая вероятность:**

Если слово X не встречалось в классе K, то P(X | K) = 0, и весь документ получает вероятность 0. Решение — **добавить сглаживание Лапласа** (параметр `alpha` в sklearn):

P(w∣класс)=count(w,класс)+αsum(все слова)+α⋅vocab sizeP(w∣класс)=sum(все слова)+α⋅vocab sizecount(w,класс)+α

---

#### Вопрос: Как может переобучиться SVM?

**Ответ:**

SVM (Support Vector Machine) может переобучиться несколькими способами:

**1. Сложная функция ядра (kernel function)**

```python
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Параметр C контролирует баланс между штрафом за ошибки и шириной разделяющей полосы
# Меньше C → больше игнорируются выбросы, шире margin
# Больше C → уже margin, точнее классификация на обучающих данных

# Слишком низкий C (0.001) - модель недообучается (слишком простые границы)
model_low_c = SVC(C=0.001, kernel='linear', random_state=42)

# Слишком высокий C (1000) - модель переобучается (подстраивается под шумы)
model_high_c = SVC(C=1000, kernel='linear', random_state=42)

# Оптимальный C (1.0) - баланс между точностью и обобщающей способностью
model_optimal = SVC(C=1.0, kernel='linear', random_state=42)

# Пример с синтетическими данными
np.random.seed(42)
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Простая линейная граница

# Добавим немного шума
y[:5] = 1 - y[:5]

# Разделим данные
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Обучим модели
models = {
    'Low C (0.001)': model_low_c,
    'High C (1000)': model_high_c,
    'Optimal C (1.0)': model_optimal
}

for name, model in models.items():
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name}:")
    print(f"  Точность на обучении: {train_acc:.3f}")
    print(f"  Точность на тесте: {test_acc:.3f}")
    print(f"  Количество опорных векторов: {sum(model.n_support_)}")
    print()
```

Чем выше степень полинома или чем более локально ядро (RBF с малым gamma), тем больше гибкость и риск переобучения.

**2. Выбросы в данных**

SVM минимизирует **margin** — расстояние между разделяющей гиперплоскостью и опорными объектами. Если в данных есть выбросы или шум:

Выброс становится опорным объектом-нарушителем (support vector) и напрямую влияет на построение разделяющей гиперплоскости.

**3. Параметр регуляризации C (слишком низкий)**

```python
from sklearn.svm import SVC

# C контролирует баланс: низкие штрафы за ошибки vs простота модели
model_low_c = SVC(C=0.001)   # Очень низкий C → переобучение (широкий margin)
model_high_c = SVC(C=1000)   # Очень высокий C → переобучение (узкий margin)
model_optimal_c = SVC(C=1.0)  # Условно оптимальное значение C
```

- **Низкий C** — модель может быть слишком жесткой или слишком мягкой
    
- **Высокий C** — мало ошибок на тренировке, но плохо на тесте
    

**Как избежать переобучения:**

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 0.1, 1],
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Используем лучшие параметры
best_model = grid_search.best_estimator_
```

---

#### Вопрос: Объясните методы предобработки текста (лемматизацию и стемминг). Какие алгоритмы знаете?

**Ответ:**

**Лемматизация vs Стемминг:**

|Аспект|Лемматизация|Стемминг|
|---|---|---|
|**Что делает**|Приводит слово к словарной форме (лемме)|Удаляет окончания, оставляя корень|
|**Результат**|Реальное слово|Может быть нереальным словом|
|**Скорость**|Медленнее (нужен словарь)|Быстрее (правила)|
|**Качество**|Выше, контекстно-зависимое|Ниже, механическое|
|**Пример**|running, runs → **run**|running → **runn**, runs → **run**|

**Лемматизация с NLTK:**

```python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
words = ['running', 'runs', 'ran', 'playing', 'played', 'better', 'best']

for word in words:
    print(f"{word} → {lemmatizer.lemmatize(word)}")

# При указании части речи результат точнее
print(lemmatizer.lemmatize('running', pos='v'))  # running → run
print(lemmatizer.lemmatize('better', pos='a'))   # better → good
```

**Стемминг с NLTK:**

```python
from nltk.stem import PorterStemmer, SnowballStemmer

porter = PorterStemmer()
snowball = SnowballStemmer('english')
words = ['running', 'runs', 'organization', 'organize', 'computer', 'computation']

print('Porter Stemmer:')
for word in words:
    print(f"{word} → {porter.stem(word)}")

print('\nSnowball Stemmer:')
for word in words:
    print(f"{word} → {snowball.stem(word)}")
```

**Результаты:**

Porter Stemmer: running → run runs → run organization → organ  ❌ (нереальное слово) organize → organ computer → comput  ❌ computation → comput

**Когда что использовать:**

|Задача|Метод|Причина|
|---|---|---|
|Классификация текстов|Лемматизация|Точность > скорость|
|Search engine|Стемминг|Скорость важна, точность не критична|
|Sentiment analysis|Лемматизация|Нужно контекстное понимание|
|Поиск по ключевым словам|Стемминг|Пользователь ввел "running", ищем и "runs"|
|Морфологический анализ|Лемматизация|Нужна часть речи и форма|

**На русском языке (pymorphy2):**

```python
import pymorphy2

morph = pymorphy2.MorphAnalyzer()
words = ['бегущий', 'бегут', 'бежал', 'организация', 'организовать']

for word in words:
    parsed = morph.parse(word)[0]
    print(f"{word} → {parsed.normal_form}")  # нормальная форма (лемма)
```

---

#### Вопрос: Какие метрики для близости текстов вы знаете?

**Ответ:**

Метрики близости делятся на две категории:

**1. Лексические метрики (работают со словами как с множествами):**

**Jaccard Similarity:**

Jaccard(A,B)=∣A∩B∣∣A∪B∣Jaccard(A,B)=∣A∪B∣∣A∩B∣

где A и B — множества слов в двух текстах.

```python
def jaccard_similarity(text1: str, text2: str) -> float:
    set1 = set(text1.split())
    set2 = set(text2.split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union else 0.0


text1 = 'the cat sat on the mat'
text2 = 'the dog sat on the rug'
print(jaccard_similarity(text1, text2))  # 0.375
```

**Проблема:** Не учитывает семантику, только точное совпадение слов.

---

**2. Семантические метрики (работают с векторами):**

**Cosine Similarity:**

cos⁡(θ)=A⃗⋅B⃗∣A⃗∣⋅∣B⃗∣=∑i=1nAi⋅Bi∑i=1nAi2⋅∑i=1nBi2cos(θ)=∣A∣⋅∣B∣A⋅B=∑i=1nAi2⋅∑i=1nBi2∑i=1nAi⋅Bi

```python
import numpy as np


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Косинусная близость (-1 до 1)."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


vec1 = np.array([1, 0])
print(cosine_similarity(vec1, np.array([1, 0])))   # 1.0
print(cosine_similarity(vec1, np.array([0, 1])))   # 0.0
print(cosine_similarity(vec1, np.array([-1, 0])))  # -1.0
```

**Euclidean Distance:**

d(A,B)=∑i=1n(Ai−Bi)2d(A,B)=∑i=1n(Ai−Bi)2

```python
import numpy as np


def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Евклидово расстояние в пространстве."""
    return np.sqrt(np.sum((vec1 - vec2) ** 2))


print(euclidean_distance(np.array([0, 0]), np.array([3, 4])))  # 5.0
```

**Manhattan Distance:**

d(A,B)=∑i=1n∣Ai−Bi∣d(A,B)=∑i=1n∣Ai−Bi∣

```python
import numpy as np


def manhattan_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Манхэттенское расстояние (как ходить по улицам города)."""
    return np.sum(np.abs(vec1 - vec2))


print(manhattan_distance(np.array([0, 0]), np.array([3, 4])))  # 7.0
```

**Сравнение метрик:**

|Метрика|Диапазон|Тип|Когда использовать|
|---|---|---|---|
|Jaccard|[askpython](https://www.askpython.com/python/examples/tf-idf-model-from-scratch)​|Лексическая|Точное совпадение слов, sparse тексты|
|Cosine Similarity|[-1, 1]|Семантическая|TF-IDF, embeddings, стандартная метрика|
|Cosine Distance||Семантическая|Когда нужно "расстояние" (1 - similarity)|
|Euclidean|[0, ∞]|Семантическая|Dense vectors, когда масштаб важен|
|Manhattan|[0, ∞]|Семантическая|Sparse vectors, вычислительно дешевле|

**Практический пример:**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

texts = [
    'I love machine learning',
    'I enjoy deep learning',
    'The weather is nice today',
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)
similarity_matrix = cosine_similarity(tfidf_matrix)
print(similarity_matrix)
```

---

#### Вопрос: Объясните разницу между косинусной близостью и косинусным расстоянием. Какое из этих значений может быть негативным?

**Ответ:**

**Cosine Similarity (косинусная близость):**

Это угол между двумя векторами. Значение показывает, **насколько похожи** векторы.

cos⁡(θ)=A⃗⋅B⃗∣A⃗∣⋅∣B⃗∣cos(θ)=∣A∣⋅∣B∣A⋅B

**Диапазон: [-1, 1]**

- **+1** — векторы идентичны (угол 0°)
    
- **0** — векторы ортогональны (угол 90°)
    
- **-1** — векторы противоположны (угол 180°)
    

**Интерпретация по диапазонам:**

| Cosine similarity | Пример пары | Интерпретация |
|---|---|---|
| +1.0 | "python" vs "python" | Абсолютно похожи |
| +0.8 | "machine learning" vs "deep learning" | Очень похожи |
| +0.5 | "university knowledge" vs "work experience" | Средняя схожесть |
| 0.0 | "university knowledge" vs "work" | Нет корреляции |
| -0.5 | "security" vs "vulnerability" | Противоположные концепции |
| -1.0 | "chatgpt is genius" vs "chatgpt is stupid" | Полностью противоположны |

**Когда бывает отрицательным:**

Отрицательное значение косинусной близости означает, что векторы указывают в **противоположных направлениях**. Это возможно, когда:

```python
import numpy as np

# Пример с противоположными векторами
vec1 = np.array([1, 0])      # вектор вправо
vec2 = np.array([-1, 0])     # вектор влево

cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
print(cos_sim)  # -1.0
```

---

**Cosine Distance (косинусное расстояние):**

Это **инверсия** косинусной близости. Показывает, **насколько непохожи** векторы.

cosine_distance=1−cos⁡(θ)cosine_distance=1−cos(θ)

**Диапазон: **

- **0** — векторы идентичны
    
- **1** — векторы ортогональны
    
- **2** — векторы противоположны
    

```python
import numpy as np


def cosine_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return 1 - similarity


vec1 = np.array([1, 0])
print(cosine_distance(vec1, np.array([1, 0])))   # 0.0
print(cosine_distance(vec1, np.array([0, 1])))   # 1.0
print(cosine_distance(vec1, np.array([-1, 0])))  # 2.0
```

**Может ли быть отрицательным?** ❌ Нет, потому что диапазон similarity [-1, 1], поэтому:

- min distance = 1 - 1 = 0
    
- max distance = 1 - (-1) = 2
    

---

**Сравнение:**

|Аспект|Similarity|Distance|
|---|---|---|
|**Диапазон**|[-1, 1]||
|**Может быть отрицательным**|✅ Да|❌ Нет|
|**Смысл**|Похожесть|Непохожесть|
|**Когда использовать**|Сравнение, рейтинги|Кластеризация (K-means ищет min distance)|
|**Интуиция**|Угол между векторами|Как далеко векторы друг от друга|

**Практическое использование:**

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity

vec1 = np.array([[1, 0, 0]])
vec2 = np.array([[0.5, 0.5, 0]])

sim = cosine_similarity(vec1, vec2)[0][0]
dist = cosine_distances(vec1, vec2)[0][0]

print(f'Similarity: {sim:.2f}')
print(f'Distance: {dist:.2f}')
print(f'1 - similarity = {1 - sim:.2f}')
```

**Когда отрицательное значение similarity имеет смысл:**

В NLP это редко, но может быть при работе с word embeddings, где противоположные концепции имеют отрицательную близость:

```python
# Предполагаем, что model — обученный Word2Vec
similarity_good_bad = model.wv.similarity('good', 'bad')      # ≈ -0.1 … -0.3
similarity_good_great = model.wv.similarity('good', 'great')  # ≈ 0.8

print(similarity_good_bad)
print(similarity_good_great)
```

---



##  Полный обзор метрик классификации и NLG

---

#### Вопрос: Объясните precision и recall разницу простыми словами и на что вы будете смотреть при отсутствии F1 score?

**Ответ:**

Confusion Matrix (матрица ошибок):

|                   | Предсказано **Положительно** | Предсказано **Отрицательно** |
|-------------------|------------------------------|-------------------------------|
| **Истина Положительно** | TP (true positive)             | FN (false negative)            |
| **Истина Отрицательно** | FP (false positive)            | TN (true negative)             |

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

Precision: Из всех людей, которых модель определила как беременных, сколько действительно беременны? Это показатель доверия к позитивному предсказанию модели. Высокий precision означает, что модель редко ошибается, когда говорит "беременна".

Recall: Из всех действительно беременных, сколько модель угадала? Это показатель полноты поиска беременных. Высокий recall означает, что модель не пропускает беременных.

Простые интерпретации:

- Precision: "Насколько мы можем доверять предсказанию о беременности?"
    
- Recall: "Какой процент беременных модель нашла?"
    

В идеальном классификаторе нет ни ложных положительных (беременных мужчин), ни ложных отрицательных (пропущенных беременных). Но на практике нужен выбор.

Когда F1 score отсутствует, смотрим на контекст задачи:

Если ошибка "ложная тревога" дорогая (FP) - смотрим на Precision. Примеры:

- Отправка спама в рассылку клиентов (дорого напугать)
    
- Диагностика редкого заболевания (дорого назначить ненужное лечение)
    
- Fraud detection в банке (дорого заблокировать легального клиента)
    

Стратегия: становимся консервативны, предсказываем "да" только при высокой уверенности.

```python
# Высокий порог вероятности → выше precision
if model_probability > 0.9:  # вместо 0.5
    predict = "positive"
```

Если ошибка "пропуск" дорогая (FN) - смотрим на Recall. Примеры:

- Медицинская диагностика рака (пропустить смерть пациента дорого)
    
- Поиск преступников в базе (пропустить опасного)
    
- Обнаружение дефектов в производстве (пропустить брак)
    

Стратегия: становимся либеральны, предсказываем "да" при малейшем подозрении.

```python
# Низкий порог вероятности → выше recall
if model_probability > 0.3:  # вместо 0.5
    predict = "positive"
```

Если затраты примерно одинаковые - ищем баланс через взвешенное среднее или анализируем cost matrix.

---

#### Вопрос: В каком случае вы будете наблюдать изменение specificity?

**Ответ:**

Sensitivity (чувствительность):

$$
\text{Sensitivity} = \frac{TP}{TP + FN} = \text{Recall}
$$

Это recall для положительного класса. Доля больных, правильно определенных как больные.

Specificity (специфичность):

$$
\text{Specificity} = \frac{TN}{TN + FP}
$$

Это recall для отрицательного класса. Доля здоровых, правильно определенных как здоровые.

Аналогия с медициной:

Sensitivity: "Если у человека COVID, тест его найдет?" = (правильно найденные больные) / (все больные)

Specificity: "Если у человека нет COVID, тест не ошибется?" = (правильно определенные здоровые) / (все здоровые)

Сравнение метрик:

|Метрика|Формула|Что ищет|
|---|---|---|
|Recall / Sensitivity|TP / (TP + FN)|Доля больных, найденных из всех больных|
|Specificity|TN / (TN + FP)|Доля здоровых, определенных из всех здоровых|
|Precision|TP / (TP + FP)|Доля верных из всех предсказаний "болен"|

Ключевое различие: Recall и Specificity делят на действительное количество в данных, Precision делит на количество примеров, которые дала модель.

Когда смотреть на Specificity:

Specificity критична, когда важно не ошибиться в отрицательном классе (не определить здоровых как больных).

Практические примеры:

Лекарство с опасными побочными эффектами: ошибка дать лекарство здоровому приводит к серьезному вреду. Максимизируем specificity.

Одобрение кредита: ошибка одобрить неплатежеспособного приводит к потере денег. Максимизируем specificity.

Прием в элитное заведение: ошибка принять неподходящего приводит к репутационному урону. Максимизируем specificity.

Изменение specificity происходит, когда:

- Меняется порог классификации
    
- Меняется распределение данных
    
- Меняется качество модели на отрицательном классе
    

---

#### Вопрос: Когда вы будете смотреть на macro, а когда на micro метрики? Почему существует weighted метрика?

**Ответ:**

Это касается мультиклассовой классификации - когда классов больше 2 (например: кот, собака, птица).

Три типа усреднения:

**Macro-average (макро-усреднение)**

$$
\text{Macro-F1} = \frac{1}{K} \sum_{i=1}^{K} F1_i
$$

Рассчитываем метрику отдельно для каждого класса, потом берем простое среднее.

```python
# Пример: 3 класса (кот, собака, птица)
F1_cat = 0.95
F1_dog = 0.90
F1_bird = 0.50  # редкий класс, модель хуже предсказывает

macro_f1 = (F1_cat + F1_dog + F1_bird) / 3
print(f"Macro-F1: {macro_f1:.2f}")
```

Когда использовать:

- Все классы одинаково важны
    
- Хотите оценить производительность на редких классах
    
- Интересует сбалансированная оценка
    

Преимущество: видны проблемы с редкими классами (F1_bird = 0.50 вы заметите).

**Micro-average (микро-усреднение)**

$$
\text{Micro-F1} = \frac{\sum_{i=1}^{K} TP_i}{\sum_{i=1}^{K} (TP_i + FP_i)}
$$

Суммируем TP, FP, FN по всем классам, потом один раз вычисляем метрику.

```python
# Того же примера
tp_total = 950 + 900 + 50
tp_fp_total = 1000 + 1000 + 100

micro_f1 = tp_total / tp_fp_total
print(f"Micro-F1: {micro_f1:.2f}")
```

Когда использовать:

- Частые классы важнее (автоматически получают больший вес)
    
- Хотите видеть общее качество на всех данных
    
- Классы имеют естественный дисбаланс
    

Характеристика: Micro-average доминируется частыми классами, потому что их примеров больше.

Сравнение Macro vs Micro:

**Сравнение Macro vs Micro (пример):**

- **Macro:**
  - F1_cat = 0.95, F1_dog = 0.90, F1_bird = 0.50
  - Среднее: (0.95 + 0.90 + 0.50) / 3 = 0.78 → видно, что модель плохо работает с редким классом "птица".
- **Micro:**
  - Суммируем все TP и FP: 1900 / 2100 = 0.90 → общая точность кажется высокой за счёт частых классов.

**Weighted-average (взвешенное усреднение)**

$$
\text{Weighted-F1} = \sum_{i=1}^{K} w_i \cdot F1_i, \qquad w_i = \frac{N_i}{\sum_{j=1}^{K} N_j}
$$

где $w_i = \frac{\text{количество примеров класса } i}{\text{всего примеров}}$

```python
# Того же примера
weights = {
    "cat": 900 / 1900,
    "dog": 900 / 1900,
    "bird": 100 / 1900,
}

weighted_f1 = (
    weights["cat"] * 0.95
    + weights["dog"] * 0.90
    + weights["bird"] * 0.50
)
print(f"Weighted-F1: {weighted_f1:.3f}")
```

Когда использовать:

- Классы несбалансированы (разное количество примеров)
    
- Хотите учесть реальное распределение данных
    
- Нужен честный компромисс между macro и micro
    

Таблица сравнения:

|Метрика|Когда|Результат|Реагирует на дисбаланс|
|---|---|---|---|
|Macro|Все классы равноправны|Показывает слабые места|Нет, игнорирует|
|Micro|Частые классы важнее|Отражает общее качество|Да, доминирует частота|
|Weighted|Учитываем дисбаланс|Компромисс с реальностью|Да, учитывает пропорции|

Практический пример:

```python
from sklearn.metrics import f1_score

y_true = [0, 0, 0, 0, 0, 1, 1, 2]  # 0 = здоров, 1 = болезнь A, 2 = болезнь B
y_pred = [0, 0, 0, 0, 1, 1, 1, 0]

f1_macro = f1_score(y_true, y_pred, average='macro')       # 0.54
f1_micro = f1_score(y_true, y_pred, average='micro')       # 0.62
f1_weighted = f1_score(y_true, y_pred, average='weighted') # 0.61

print(f"Macro: {f1_macro:.2f}, Micro: {f1_micro:.2f}, Weighted: {f1_weighted:.2f}")
```

---

#### Вопрос: Что такое perplexity? С чем мы можем ее считать?

**Ответ:**

Perplexity (недоумение) - это метрика, которая показывает, насколько модель сомневается при генерации текста.

Аналогия:

Вы хорошо знаете язык: читаете предложение → сразу знаете, какое следующее слово → Perplexity низкая → модель уверена.

Вы новичок в языке: читаете предложение → нужно перебрать 100 вариантов слов и проверить в словаре → Perplexity высокая → модель не уверена.

Правило: Чем ниже perplexity, тем лучше модель.

Математическое определение:

$$
\text{Perplexity} = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \log P(w_i \mid w_{1:i-1})\right)
$$

где:

- N - количество слов в тексте
    
- P(w_i | w_{1:i-1}) - вероятность слова i, учитывая предыдущие слова
    

Или через кросс-энтропию:

$$
\text{Perplexity} = e^{\text{Cross-Entropy}}
$$

Интерпретация: Perplexity = среднее количество слов, из которых модель выбирает на каждом шаге.

- Perplexity = 10 → модель выбирает из 10 слов в среднем
    
- Perplexity = 100 → модель выбирает из 100 слов
    
- Perplexity = 1 → идеальная модель (выбирает правильно всегда)
    

Практический расчет на Python:

```python
import math
import torch
from torch.nn import CrossEntropyLoss

# Пример: предсказания модели для 5 слов
logits = torch.tensor([
    [1.0, 2.0, 3.0],  # слово 1: вероятности [0.09, 0.24, 0.67]
    [2.0, 1.0, 3.0],  # слово 2
    [1.0, 1.0, 1.0],  # слово 3
    [3.0, 1.0, 1.0],  # слово 4
    [1.0, 3.0, 2.0],  # слово 5
])

labels = torch.tensor([0, 1, 2, 0, 2])

loss_fn = CrossEntropyLoss()
ce_loss = loss_fn(logits, labels)

perplexity = math.exp(ce_loss.item())
print(f"Perplexity: {perplexity:.4f}")
```

Или через вероятности напрямую:

```python
import math

# Вероятности правильного слова на каждом шаге
probabilities = [0.8, 0.9, 0.5, 0.7, 0.6]

log_probs = [math.log(p) for p in probabilities]
avg_log_prob = sum(log_probs) / len(log_probs)

perplexity = math.exp(-avg_log_prob)
print(f"Perplexity: {perplexity:.4f}")  # ≈ 1.48
```

Интерпретация значений:

|Perplexity|Качество модели|Пример|
|---|---|---|
|1.0|Идеально (невозможно)|Модель всегда угадывает|
|2-3|Отличное|Модель BERT на тесте|
|5-10|Хорошее|Перевод документов|
|50-100|Среднее|Генерация случайного текста|
|1000+|Плохое|Угадывание из всего словаря|

Perplexity для разных задач:

```python
# Языковая модель (Language Model)
perplexity_lm = math.exp(cross_entropy_loss)

# Machine Translation
perplexity_mt = math.exp(cross_entropy_loss)

# Text Summarization
perplexity_sum = math.exp(cross_entropy_loss)

# Machine Classification
# Здесь чаще используют accuracy, F1 и другие метрики
```

Важные характеристики:

Intrinsic метрика - не зависит от конкретной задачи, просто "качество языка"

Высокая perplexity не гарантирует плохое качество на практике. Примеры:

- Модель с низкой perplexity может хорошо генерировать текст, но плохо переводить
    
- Причина: perplexity показывает, как модель предсказывает следующее слово, но не показывает качество смысла
    

---

#### Вопрос: Что такое метрика BLEU?

**Ответ:**

BLEU (Bilingual Evaluation Understudy) - метрика для оценки машинного перевода. Сравнивает машинно переведенный текст с эталонным переводом, учитывая точность слов и грамматическую правильность.

Основная идея: считаем, сколько n-gram'ов (последовательностей из N слов) в сгенерированном переводе совпадают с эталоном.

Пример:

```
Эталон: "The cat sat on the mat" Перевод: "A cat sat on the mat" Unigrams (1-gram): the, cat, sat, on, the, mat Совпадения: cat, sat, on, the, mat (5 из 6) → precision = 5/6 Bigrams (2-gram): "the cat", "cat sat", "sat on", "on the", "the mat" Совпадения: "cat sat", "sat on", "on the", "the mat" (4 из 5) → precision = 4/5
```

Формула для N-gram precision:

pn=∑n-gram∈переводCountmatch(n-gram)∑n-gram∈переводCount(n-gram)pn=∑n-gram∈переводCount(n-gram)∑n-gram∈переводCountmatch(n-gram)

где:

- Count_match = количество n-gram'ов перевода, совпадающих с эталоном
    
- Count = всего n-gram'ов в переводе
    

```python
# Пример расчёта unigram precision
translation = ['the', 'cat', 'sat']
reference = ['the', 'cat', 'is']

matches = 2  # 'the', 'cat'
total = len(translation)
p_1 = matches / total
print(f"p1: {p_1:.2f}")  # 0.67
```

Brevity Penalty (штраф за краткость):

Если перевод короче эталона, даем штраф. Иначе модель может выбросить слова и получить 100% точность.

$$
BP =
\begin{cases}
1, & \text{если длина перевода} \ge \text{ длины эталона} \\
e^{1 - \frac{\text{длина эталона}}{\text{длина перевода}}}, & \text{иначе}
\end{cases}
$$

```python
# Пример brevity penalty
reference_len = 10
translation_len = 5

bp = math.exp(1 - reference_len / translation_len)
print(f"BP: {bp:.3f}")  # 0.368
```

Финальная формула BLEU:

$$
\text{BLEU} = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)
$$

где:

- BP = brevity penalty
    
- p_n = precision для n-gram'ов размера n
    
- w_n = вес для n-gram'ов (обычно одинаковый, например 0.25 для N=4)
    

```python
import math

# Пример расчёта BLEU-4
p_1 = 0.95
p_2 = 0.90
p_3 = 0.80
p_4 = 0.75
BP = 0.98

weights = [0.25, 0.25, 0.25, 0.25]
score = BP * math.exp(
    weights[0] * math.log(p_1)
    + weights[1] * math.log(p_2)
    + weights[2] * math.log(p_3)
    + weights[3] * math.log(p_4)
)
print(f"BLEU: {score:.4f}")  # ≈ 0.86
```

На практике с библиотеками:

```python
from nltk.translate.bleu_score import sentence_bleu

# Эталонный перевод (может быть несколько вариантов)
references = [
    ['the', 'cat', 'sat', 'on', 'the', 'mat'],
    ['a', 'cat', 'is', 'sitting', 'on', 'the', 'mat'],
]

# Сгенерированный перевод
candidate = ['the', 'cat', 'sat', 'on', 'the', 'mat']

# Вычисляем BLEU-4 (все n-gram'ы до 4)
bleu_score = sentence_bleu(references, candidate, weights=(0.25, 0.25, 0.25, 0.25))
print(f"BLEU-4: {bleu_score:.4f}")  # 1.0 (идеальное совпадение)
```

Диапазон BLEU и интерпретация:

|BLEU|Качество|Интерпретация|
|---|---|---|
|0.00-0.10|Ужасное|Перевод почти не совпадает|
|0.10-0.30|Плохое|Много ошибок в словах|
|0.30-0.50|Среднее|Синтаксис близок, но смысл сдвинут|
|0.50-0.70|Хорошее|Большинство слов верно|
|0.70-0.90|Отличное|Близко к человеческому переводу|
|0.90-1.00|Идеальное|Совпадает с эталоном|

На практике: Google Translate (2016) показывает ~0.45-0.50, State-of-the-art (2020) ~0.70+

Ограничения BLEU:

Чувствительна к токенизации - разные системы токенизируют по-разному, сложно сравнивать

Не работает для морфологически сложных языков (русский, финский) - одно слово может иметь много форм

Не учитывает синонимы - если выбрали синоним вместо точного совпадения, BLEU = 0

Не коррелирует с человеческой оценкой для некоторых случаев

```python
# Пример проблемы с синонимами
reference = ['the', 'big', 'dog']
candidate = ['the', 'large', 'dog']  # "large" = синоним "big"

bleu = sentence_bleu([reference], candidate)
print(bleu)  # 0.67 (не видит близость синонимов)
```

---

#### Вопрос: Объясните разницу между разными видами ROUGE метрики?

**Ответ:**

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) - метрика для оценки суммаризации текстов. Сравнивает сгенерированный summary с эталонным, считая совпадающие n-gram'ы и последовательности.

Три основных типа ROUGE:

**ROUGE-N (N-gram overlap)**

Считает, сколько N-gram'ов из эталона совпадают с сгенерированным summary.

$$
\text{ROUGE-N}_{\text{recall}} = \frac{\text{совпадающие N-gram'ы}}{\text{все N-gram'ы в эталоне}}
$$

$$
\text{ROUGE-N}_{\text{precision}} = \frac{\text{совпадающие N-gram'ы}}{\text{все N-gram'ы в summary}}
$$

$$
\text{ROUGE-N}_{F1} = \frac{2 \cdot \text{Recall} \cdot \text{Precision}}{\text{Recall} + \text{Precision}}
$$

ROUGE-1: Unigram'ы (отдельные слова)

ROUGE-2: Bigram'ы (пары слов)

Пример ROUGE-1:


```text
Эталон: "I really loved reading the Hunger Games" Summary: "I really really loved reading reading the Hunger Games" Unigrams эталона: {I, really, loved, reading, the, Hunger, Games} = 7 Unigrams summary: {I, really, really, loved, reading, reading, the, Hunger, Games} = 9 Совпадающие: {I, really, loved, reading, the, Hunger, Games} = 7 ROUGE-1 Recall = 7 / 7 = 1.0 (нашли все слова из эталона) ROUGE-1 Precision = 7 / 9 = 0.78 (78% слов в summary совпадают с эталоном) ROUGE-1 F1 = 2 × (1.0 × 0.78) / (1.0 + 0.78) = 0.875
```

Проблема: Recall идеален, но summary содержит повторения! ROUGE-N не видит проблемы.

**ROUGE-L (Longest Common Subsequence)**

Ищет самую длинную общую подпоследовательность (LCS) между эталоном и summary.

Важно: Слова должны быть в одном порядке, но не обязательно подряд.

$$
\text{ROUGE-L}_{\text{recall}} = \frac{\text{LCS}(summary, reference)}{\text{длина reference}}
$$

$$
\text{ROUGE-L}_{\text{precision}} = \frac{\text{LCS}(summary, reference)}{\text{длина summary}}
$$

$$
\text{ROUGE-L}_{F1} = \frac{2 \cdot \text{Recall} \cdot \text{Precision}}{\text{Recall} + \text{Precision}}
$$

Пример ROUGE-L:

```text
Reference: "I really loved reading the Hunger Games" Summary:   "I really really loved reading reading the Hunger Games" LCS: "I really loved reading the Hunger Games" (длина = 7) ROUGE-L Recall = 7 / 7 = 1.0 ROUGE-L Precision = 7 / 9 = 0.78 ROUGE-L F1 = 0.875
```

Этот же результат, но ROUGE-L видит структуру (порядок слов важен).

Расширение: ROUGE-W

Дает больший вес последовательным совпадениям:

```
"I loved reading" (подряд) → вес выше, чем "I ... reading" (с пропусками)
```

**ROUGE-S (Skip-gram)**

Считает skip-gram'ы - это N-gram'ы, где слова могут быть разделены другими словами, но остаются в одном порядке.

Параметр N: сколько слов можно пропустить.

```
Пример с N=2 (можно пропустить до 2 слов): Фраза: "Police killed the gunman" Skip-gram'ы: - ("police", "killed") подряд - ("police", "the") 1 слово пропущено - ("police", "gunman") 2 слова пропущены - ("killed", "the") подряд - ("killed", "gunman") 1 слово пропущено - ("the", "gunman") подряд Итого: 6 skip-gram'ов
```

Потом считаем, сколько из них совпадают с эталоном.

```python
def get_skip_grams(words, n=2):
    """Генерирует skip-gram'ы"""
    skip_grams = set()
    for i in range(len(words)):
        for j in range(i + 1, min(i + n + 1, len(words) + 1)):
            skip_grams.add((words[i], words[j]))
    return skip_grams


words = 'Police killed the gunman'.split()
skip_grams = get_skip_grams(words, n=2)
print(skip_grams)
# {('police', 'killed'), ('police', 'the'), ('police', 'gunman'),
#  ('killed', 'the'), ('killed', 'gunman'), ('the', 'gunman')}
```

Сравнение видов ROUGE:

|ROUGE|Что считает|Чувствителен к|Когда использовать|
|---|---|---|---|
|ROUGE-1|Слова (unigrams)|Выбор словаря|Общее качество|
|ROUGE-2|Пары слов (bigrams)|Грамматику|Локальный контекст|
|ROUGE-L|Порядок слов|Структуру|Значимость слов|
|ROUGE-S|Skip-gram'ы|Гибкость порядка|Синтаксическую близость|

Практический расчет на Python:

```python
from rouge_score import rouge_scorer

reference = 'I really loved reading the Hunger Games'
candidate = 'I really really loved reading reading the Hunger Games'

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = scorer.score(reference, candidate)

print(scores['rouge1'])  # Recall=1.0, Precision=0.78, Fmeasure=0.875
print(scores['rouge2'])  # Recall=0.83, Precision=0.65, Fmeasure=0.73
print(scores['rougeL'])  # Recall=1.0, Precision=0.78, Fmeasure=0.875
```

Результаты на примере:

```python
import evaluate

rouge = evaluate.load('rouge')
predictions = [
    'I really really loved reading reading the Hunger Games',
    'Police killed the gunman',
]
references = [
    'I really loved reading the Hunger Games',
    'the gunman killed the police',
]

results = rouge.compute(predictions=predictions, references=references)
print(results)  # {'rouge1': 0.88, 'rouge2': 0.71, 'rougeL': 0.66, 'rougeLsum': 0.66}
```

Ограничения ROUGE:

Не видит синонимов - "good" и "great" считаются разными

Не учитывает смысл - может совпадать по словам, но отличаться по значению

Чувствительна к tokenization - "U.S.A" vs "USA" считаются разными

Хорошо коррелирует с человеческой оценкой для суммаризации

---

#### Вопрос: В чем отличие BLEU от ROUGE?

**Ответ:**

Обе метрики основаны на n-gram overlaps, но предназначены для разных задач:

BLEU (Bilingual Evaluation Understudy):

- Задача: Машинный перевод (MT)
    
- Ориентация: Precision-based (точность перевода)
    
- Что сравнивает: Машинный перевод vs эталонный перевод
    
- Штраф: Brevity penalty (штраф за короткий перевод)
    

BLEU = что из сгенерированного совпадает с эталоном? = Precision ориентированная метрика

ROUGE (Recall-Oriented Understudy for Gisting Evaluation):

- Задача: Суммаризация (Abstractive summarization)
    
- Ориентация: Recall-based (полнота информации)
    
- Что сравнивает: Сгенерированный summary vs эталонный summary
    
- Штраф: Нет штрафа за длину
    

ROUGE = сколько из эталона попало в сгенерированное? = Recall ориентированная метрика

Сравнение на числах:


`Эталон: "The quick brown fox" Вывод:  "The quick brown fox ran very fast" BLEU: - Считает precision: 4/7 = 0.57 (4 слова совпадают из 7 сгенерированных) - Применяет brevity penalty (нет, потому что длина > эталона) - BLEU = 0.57 ← высокий штраф за лишние слова! ROUGE: - Считает recall: 4/4 = 1.0 (все слова эталона найдены) - ROUGE = 1.0 ← высокий score, хотя есть лишние слова!`

Почему разные ориентации?

ПЕРЕВОД (BLEU → Precision):

Задача: Переводить ровно то, что дано

Плохо: Переводить лишнее (hallucination) - "Hello" → "Hello dear friend good morning" (добавили)

Хорошо: Совпадение с эталоном (precision)

СУММАРИЗАЦИЯ (ROUGE → Recall):

Задача: Извлечь самую важную информацию

Хорошо: Найти все ключевые моменты (recall), даже если summary не совпадает слово-в-слово

Плохо: Пропустить важную информацию (low recall)

Таблица сравнения:

|Аспект|BLEU|ROUGE|
|---|---|---|
|Задача|Перевод|Суммаризация|
|Считает|Precision|Recall|
|Вопрос|Сколько из вывода верно?|Сколько из эталона найдено?|
|Штраф за длину|Brevity penalty|Нет|
|N-grams|Обычно 1-4|1, 2, L, S|
|Синонимы|Не видит|Не видит|
|Корреляция с человеком|Средняя|Хорошая|

Когда использовать какую:

```python
# Оцениваем машинный перевод → BLEU
from nltk.translate.bleu_score import corpus_bleu

bleu = corpus_bleu(references, hypotheses)

# Оцениваем суммаризацию → ROUGE
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'])
scores = scorer.score(reference_summary, generated_summary)

# Абстрактивное резюме → ROUGE
# Извлекающее резюме → ROUGE + precision
# Диалоговые системы → BLEU (точность ответа)
# Вопрос-ответ → ROUGE + precision (полнота + правильность)
```

Практический пример:

```python
# Machine Translation example
reference_translation = 'The cat is on the mat'
mt_output = 'The cat is on the mat and sleeping'
# BLEU: 5 слов совпадают из 8 → precision = 5 / 8 = 0.625
# ROUGE: 5 слов совпадают из 6 → recall = 5 / 6 = 0.833

# Summarization example
reference_summary = 'Apple launched new iPhone with better camera'
generated_summary = 'Apple released iPhone with improved camera and battery'
# BLEU: ≈ 0.5
# ROUGE-1: ≈ 0.71
# Выводы:
# BLEU строже (штрафует за лишние слова)
# ROUGE мягче (оценивает покрытие содержания)
```


## Word2Vec: Основы обучения и эмбеддинги

---

#### Вопрос: Объясните как учится Word2Vec? Какая функция потерь? Что максимизируется?

**Ответ:**

Word2Vec - это метод получения плотных векторных представлений слов (эмбеддинги). Основная идея: слова, которые встречаются в похожих контекстах, должны иметь похожие векторы.

Существует две архитектуры:

**CBOW (Continuous Bag of Words)**

Предсказывает целевое слово по его контексту (окружающим словам).

Пример:


```
Предложение: "The quick brown fox jumps" Окно = 2 Контекст [The, quick, brown, jumps] → Предсказать "fox"
```

**Skip-gram**

Предсказывает контекст (окружающие слова) по целевому слову.


```
Предложение: "The quick brown fox jumps" Окно = 2 Слово "fox" → Предсказать [brown, jumps] (и еще context window слова)
```

**Функция потерь для Skip-gram**

Функция потерь - это Negative Log Likelihood:

Loss=−log⁡P(wcontext∣wtarget)Loss=−logP(wcontext∣wtarget)

где P(w_context | w_target) вычисляется через softmax:

P(wc∣wt)=exp⁡(vwcT⋅vwt)∑w=1Vexp⁡(vwT⋅vwt)P(wc∣wt)=∑w=1Vexp(vwT⋅vwt)exp(vwcT⋅vwt)

где:

- v_wc, v_wt - векторные представления слов
    
- V - размер словаря
    

Проблема: softmax требует суммирования по всему словарю (V может быть 100,000+), это медленно.

**Что максимизируется?**

Максимизируется вероятность контекстных слов при заданном целевом слове:

Maximize: ∑t=1T∑−m≤j≤m,j≠0log⁡P(wt+j∣wt)Maximize: ∑t=1T∑−m≤j≤m,j=0logP(wt+j∣wt)

где:

- T - длина предложения
    
- m - размер окна контекста
    
- Ищем такие векторы, чтобы слова с похожим контекстом были близко в пространстве
    

**Эквивалентно:**

Минимизируем функцию потерь (negative log probability):

Minimize: −∑t=1T∑−m≤j≤m,j≠0log⁡P(wt+j∣wt)Minimize: −∑t=1T∑−m≤j≤m,j=0logP(wt+j∣wt)

**Интуиция:**

Мы хотим, чтобы:

- P(контекстное слово | целевое слово) была высокой для реальных контекстных слов
    
- P(случайное слово | целевое слово) была низкой для случайных слов
    

Это заставляет близкие слова иметь похожие векторы.

**Пример на Python:**

```python
from gensim.models import Word2Vec

sentences = [
    'the quick brown fox jumps over the lazy dog',
    'the dog sat on the mat',
    'a quick brown fox',
]

tokenized = [sent.split() for sent in sentences]

model = Word2Vec(
    sentences=tokenized,
    vector_size=100,  # размерность векторов
    window=2,         # размер окна контекста
    min_count=1,      # минимальная частота слова
    workers=4,        # количество потоков
    epochs=10,        # количество эпох
    sg=1,             # 1 = Skip-gram, 0 = CBOW
)

vector_fox = model.wv['fox']
similar = model.wv.most_similar('fox', topn=5)
print(similar)
```

---

#### Вопрос: Какие способы получения эмбеддингов знаете? Когда какие будут лучше?

**Ответ:**

Существует несколько основных подходов к получению эмбеддингов:

**1. Count-based (TF-IDF, Co-occurrence matrices)**

Идея: слова встречаются вместе → близкие в смысле.

```python
# TF-IDF матрица 
from sklearn.feature_extraction.text import TfidfVectorizer 
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(texts)  # shape: (n_docs, 1000)`
```

Когда использовать:

- Быстро нужны эмбеддинги
    
- Мало данных для обучения
    
- Нужна интерпретируемость (какие слова важны)
    

Минусы: Очень high-dimensional (vocab_size), sparse, не учитывают семантику хорошо

**2. Prediction-based (Word2Vec, FastText, GloVe)**

Идея: обучить нейросеть предсказывать слово по контексту, эмбеддинги - это веса скрытого слоя.

```python
from gensim.models import Word2Vec
model = Word2Vec(sentences, vector_size=300, window=5)
embeddings = model.wv  # shape: (vocab_size, 300) - плотные векторы
```

Когда использовать:

- Есть большой текстовый корпус
    
- Нужны плотные эмбеддинги (300-500 размерность)
    
- Нужна скорость обучения
    

Минусы: Static (одно значение для всех контекстов)

**3. Contextual (BERT, ELMo, GPT)**

Идея: эмбеддинг зависит от контекста, в котором слово встречается.

```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
inputs = tokenizer("hello world", return_tensors="pt")
outputs = model(**inputs)
embeddings = outputs.last_hidden_state  # shape: (1, 2, 768) - contextual
```

Когда использовать:

- Высокий бюджет вычислений
    
- Нужна максимальная точность
    
- Контекст слова важен (например, "bank" в разных контекстах)
    

Минусы: Медленный inference, требует GPU

**4. Sparse (Bag of Words, One-hot)**

```python
# One-hot encoding
vocab = ['the', 'cat', 'sat', 'on', 'mat']
word = 'cat'
one_hot = [0, 1, 0, 0, 0]  # shape: (5,)
```

Когда использовать:

- Baseline для быстрого прототипирования
    
- Простые задачи
    

Минусы: Не выражает семантику, огромные векторы

**5. Hybrid approaches (Doc2Vec, Sentence-BERT)**

```python
from gensim.models import Doc2Vec
from sentence_transformers import SentenceTransforme
# Doc2Vec - расширение Word2Vec на уровне документов
doc_model = Doc2Vec(documents, vector_size=300, epochs=40)
# Sentence-BERT - эффективная версия BERT для предложений
sent_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = sent_model.encode(["hello world"])  # shape: (1, 384)
```

Когда использовать:

- Нужны эмбеддинги на уровне документов/предложений
    
- Есть GPU для inference
    

**Сравнение методов:**

|Метод|Размер|Скорость обучения|Скорость inference|Качество|Интерпретируемость|
|---|---|---|---|---|---|
|TF-IDF|10k-100k|Очень быстро|Очень быстро|Низкое|Высокая|
|Word2Vec|100-500|Быстро|Очень быстро|Среднее|Средняя|
|FastText|100-500|Быстро|Очень быстро|Среднее-высокое|Средняя|
|GloVe|50-300|Быстро|Очень быстро|Среднее|Средняя|
|ELMo|1024|Медленно|Медленно|Высокое|Низкая|
|BERT|768-1024|Медленно|Медленно|Очень высокое|Низкая|
|Sentence-BERT|384-768|Медленно|Быстро|Высокое|Низкая|

**Практический выбор:**

```python
from gensim.models import FastText
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel

# Быстрый прототип (день): TF-IDF или Word2Vec
quick_embed = TfidfVectorizer()

# Production, средний бюджет: FastText или GloVe
fast_embed = FastText(sentences)

# Production, нет бюджетных ограничений: Sentence-BERT
quality_embed = SentenceTransformer('all-mpnet-base-v2')

# Для задач NLU (понимание текста): BERT
understanding = AutoModel.from_pretrained('bert-base-uncased')
```

---

#### Вопрос: В чем отличие между static и contextual эмбеддингов?

**Ответ:**

**Static эмбеддинги (Word2Vec, GloVe, FastText)**

Одно векторное представление для каждого слова, независимо от контекста.

```text
Слово "bank": В предложении: "I went to the bank to withdraw money" Вектор: [0.2, 0.5, -0.3, 0.1, ...] В предложении: "The river bank is beautiful" Вектор: [0.2, 0.5, -0.3, 0.1, ...]  ← ТОТ ЖЕ ВЕКТОР!
```

Свойства:

- Один вектор на слово (размерность: vocab_size × embedding_dim)
    
- Быстро: просто lookup из таблицы
    
- Простой: легко использовать
    

Минусы: Не учитывает полисемию (одно слово - разные значения)

**Contextual эмбеддинги (BERT, ELMo, GPT)**

Векторное представление зависит от контекста - окружающих слов.

```text
Слово "bank": В предложении: "I went to the bank to withdraw money" Вектор: [0.15, 0.6, -0.2, 0.05, ...] В предложении: "The river bank is beautiful" Вектор: [0.35, 0.3, -0.5, 0.2, ...]  ← ДРУГОЙ ВЕКТОР!
```

Свойства:

- Вектор генерируется динамически для каждого слова и каждого контекста
    
- Медленнее: нужно прогнать нейросеть
    
- Сложнее: требует GPU для efficiency
    

Плюсы: Учитывает полисемию и контекстуальное значение

**Пример на Python:**

```python
# Static эмбеддинги — Word2Vec
from gensim.models import Word2Vec

model = Word2Vec(sentences)
bank_vec1 = model.wv['bank']
bank_vec2 = model.wv['bank']

similarity = (bank_vec1 == bank_vec2).all()
print(similarity)  # True — вектор одинаковый в любом контексте
```

```python
# Contextual эмбеддинги — BERT
import torch
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

sent1 = 'I went to the bank to withdraw money'
sent2 = 'The river bank is beautiful'

inputs1 = tokenizer(sent1, return_tensors='pt')
inputs2 = tokenizer(sent2, return_tensors='pt')

outputs1 = model(**inputs1)
outputs2 = model(**inputs2)

tokens1 = tokenizer.tokenize(sent1)
tokens2 = tokenizer.tokenize(sent2)

idx1 = tokens1.index('bank')
idx2 = tokens2.index('bank')

bank_embedding1 = outputs1.last_hidden_state[0, idx1]
bank_embedding2 = outputs2.last_hidden_state[0, idx2]

are_same = torch.allclose(bank_embedding1, bank_embedding2)
print(are_same)  # False — контекстные векторы различаются
```

**Сравнение:**

|Аспект|Static|Contextual|
|---|---|---|
|Полисемия|Не учитывает|Учитывает|
|Скорость обучения|Быстро|Медленно|
|Скорость inference|Очень быстро (lookup)|Медленно (forward pass)|
|Размер модели|Маленький|Большой (сотни МБ)|
|Использование памяти|Низкое|Высокое|
|Качество на сложных задачах|Среднее|Очень высокое|

**Когда что использовать:**

```python
# Static — для простых задач и быстрого обучения
if task_complexity == 'simple' and budget_time == 'low':
    embeddings = Word2Vec(sentences)

# Contextual — для сложных задач, требующих глубокого понимания
elif task_complexity == 'complex' and has_gpu:
    embeddings = SentenceTransformer('all-mpnet-base-v2')

# Компромиссный вариант
else:
    embeddings = FastText(sentences)
```

---

#### Вопрос: Какие две основные архитектуры вы знаете и какая из них учится быстрее?

**Ответ:**

Две основные архитектуры Word2Vec:

**1. CBOW (Continuous Bag of Words)**

Архитектура: Контекст → Целевое слово

```text
Входные данные: [w_{t-2}, w_{t-1}, w_{t+1}, w_{t+2}]  (окружающие слова)
                        ↓
             Embedding Layer (lookup для каждого слова)
                        ↓
        Average/Sum векторов контекста
                        ↓
        Dense слой + Softmax
                        ↓
Выход: вероятность целевого слова w_t
```

Пример:

```text
Предложение: "The quick brown fox jumps"
Window = 2
CBOW: [The, quick, brown, jumps] → "fox"
```

**2. Skip-gram**

Архитектура: Целевое слово → Контекст

```text
Входные данные: w_t  (целевое слово)
                 ↓
        Embedding Layer (lookup)
                 ↓
        Dense слой + Softmax
                 ↓
Выход: вероятности контекстных слов [w_{t-2}, w_{t-1}, w_{t+1}, w_{t+2}]
```

Пример:

```text
Предложение: "The quick brown fox jumps"
Window = 2
Skip-gram: "fox" → [brown, jumps, The, quick]
```

**Скорость обучения:**

Skip-gram учится медленнее, потому что предсказывает несколько слов (контекст).

CBOW учится быстрее, потому что предсказывает только одно слово (целевое).

На практике разница не велика, но Skip-gram часто работает лучше на качество.

```python
# CBOW — быстрее, но хуже качество на больших корпусах
model_cbow = Word2Vec(sentences, sg=0, window=5, workers=4)
# Примерное время на 1M слов: ~30 сек

# Skip-gram — медленнее, но лучше качество
model_sg = Word2Vec(sentences, sg=1, window=5, workers=4)
# Примерное время на 1M слов: 2-3 мин
```

**Сравнение:**

|Аспект|CBOW|Skip-gram|
|---|---|---|
|Архитектура|Контекст → Слово|Слово → Контекст|
|Скорость обучения|Быстро|Медленно (в 3-5 раз)|
|Качество|Среднее|Высокое|
|На частых словах|Хорошо|Лучше|
|На редких словах|Хуже|Лучше|
|Рекомендация|Малые корпусы|Большие корпусы|

**Когда что использовать:**

```python
if corpus_size < 1_000_000:
    model = Word2Vec(sentences, sg=0)  # маленький корпус → CBOW
else:
    model = Word2Vec(sentences, sg=1)  # большой корпус → Skip-gram
```

---

#### Вопрос: В чем разница между Glove, ELMO, FastText и Word2Vec?

**Ответ:**

Все это методы для получения word embeddings, но они отличаются подходом и качеством.

**Word2Vec (2013)**

Метод: Prediction-based (CBOW / Skip-gram)

```python
from gensim.models import Word2Vec

model = Word2Vec(sentences, vector_size=300, window=5)
embeddings = model.wv['word']  # shape: (300,)
```

Плюсы:

- Очень быстро обучается
    
- Легко использовать (просто lookup)
    
- Основа для современных методов
    

Минусы:

- Static (не учитывает контекст)
    
- Слабо на редких словах (неизвестные слова = случайный вектор)
    

---

**FastText (2016)**

Расширение Word2Vec: слово разбивается на символьные n-grams.

```python
from gensim.models import FastText

model = FastText(sentences, vector_size=300, window=5)
embeddings = model.wv['word']  # shape: (300,)

# Неизвестное слово всё равно получает вектор
unknown_word = model.wv['unknown_word123']
```

Идея:

```text
Слово "running" разбивается на 3-grams: <ru, run, unn, nni, nin, ing, ng>
Эмбеддинг "running" = среднее по эмбеддингам всех n-grams
```

Плюсы:

- Обрабатывает неизвестные слова (через n-grams)
    
- Лучше на опечатках и морфологии
    

Минусы:

- Медленнее, чем Word2Vec
    
- Все равно static
    

---

**GloVe (2014)**

Глобальные векторы. Гибрид между count-based и prediction-based.

```python
# GloVe требует отдельного обучения, чаще используют pre-trained модели
embeddings = load_glove_vectors('glove.6B.300d.txt')
word_vector = embeddings['word']  # shape: (300,)
```

Идея:

- Строит матрицу co-occurrence слов
    
- Факторизует эту матрицу
    
- Минимизирует разницу между скалярным произведением векторов и логарифмом co-occurrence
    

Плюсы:

- Комбинирует глобальную статистику с локальным контекстом
    
- Часто работает лучше, чем Word2Vec на benchmark'ах
    

Минусы:

- Дороже обучать
    
- Static
    

---

**ELMo (2018)**

Embeddings from Language Models. Contextual embeddings.

```python
from allennlp.commands.elmo import ElmoEmbedder

elmo = ElmoEmbedder()
sentence = 'The cat sat on the mat'
embeddings = elmo.embed_sentence(sentence.split())  # shape: (3, 1024, seq_len)
```

Идея:

- Обучает bidirectional LSTM на языковое моделирование
    
- Эмбеддинги = функция от всех слоев LSTM
    

Плюсы:

- Contextual (зависит от контекста)
    
- Очень хорошее качество
    

Минусы:

- Медленно обучается и использует
    
- Требует GPU
    

---

**BERT (2018)**

Bidirectional Encoder Representations from Transformers. State-of-the-art contextual embeddings.

```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

inputs = tokenizer('The cat sat on the mat', return_tensors='pt')
outputs = model(**inputs)
embeddings = outputs.last_hidden_state  # shape: (1, 7, 768)
```

Идея:

- Transformer architecture (вместо LSTM)
    
- Masked language modeling (маскирует 15% слов и учится их предсказывать)
    
- Next sentence prediction (предсказывает, является ли вторая фраза продолжением первой)
    

Плюсы:

- Лучшее качество (state-of-the-art)
    
- Contextual
    
- Fine-tune friendly (легко адаптировать под задачу)
    

Минусы:

- Самый медленный
    
- Требует много памяти
    

---

**Сравнение всех методов:**

|Метод|Год|Static/Contextual|Скорость обучения|Скорость inference|Качество|Память|
|---|---|---|---|---|---|---|
|Word2Vec|2013|Static|Очень быстро|Очень быстро|Среднее|Маленькая|
|GloVe|2014|Static|Быстро|Очень быстро|Среднее-высокое|Маленькая|
|FastText|2016|Static|Быстро|Очень быстро|Среднее|Маленькая|
|ELMo|2018|Contextual|Медленно|Медленно|Высокое|Средняя|
|BERT|2018|Contextual|Очень медленно|Медленно|Очень высокое|Большая|

**Практический выбор:**

```python
# Быстрый прототип (день)
embeddings = FastText(sentences, vector_size=300)

# Production с средним бюджетом
glove_vectors = load_glove('glove.6B.300d')

# Высокое качество, есть GPU
embeddings = SentenceTransformer('all-mpnet-base-v2')

# NLU задачи (классификация, NER)
model = AutoModel.from_pretrained('bert-base-uncased')
```

---

#### Вопрос: Что такое negative sampling и зачем он нужен? Какие еще трюки у word2vec знаете и как можете применять у себя?

**Ответ:**

**Negative Sampling (отрицательная выборка)**

Проблема стандартного Word2Vec:

Softmax требует нормализации по всему словарю:

P(wc∣wt)=exp⁡(vwcT⋅vwt)∑w=1Vexp⁡(vwT⋅vwt)P(wc∣wt)=∑w=1Vexp(vwT⋅vwt)exp(vwcT⋅vwt)

Если V = 100,000, то на каждом шаге нужно вычислить 100,000 экспонент. Очень медленно!

**Решение - Negative Sampling:**

Вместо softmax по всему словарю, обучаем бинарную классификацию:

- Настоящее слово (контекст) - label=1
    
- Случайные слова - label=0
    

```python
# Классический softmax (медленно)
P(w_c | w_t) = exp(score) / sum(exp(all_scores))
# V операций
# Negative sampling (быстро)
# Берем 1 позитивный пример и k негативных
# Обучаем логистическую регрессию на этих k+1 примерах`

Математика:

L=−log⁡(σ(vwT⋅vt))−∑i=1klog⁡(σ(−vwiT⋅vt))L=−log(σ(vwT⋅vt))−∑i=1klog(σ(−vwiT⋅vt))

где:

- Первый term: максимизируем скалярное произведение с реальным контекстом
    
- Второй term: минимизируем скалярное произведение со случайными словами
    

**Пример на Python:**

```python
from gensim.models import Word2Vec
# Negative sampling включен по умолчанию в Skip-gram
model = Word2Vec(sentences, sg=1, vector_size=300, window=5, negative=5, workers=4)
# Без negative sampling (медленно):
model_slow = Word2Vec(sentences, sg=1, vector_size=300, window=5, negative=0, workers=1)
```


Скорость:
```text
Обучение на 1M слов: - С negative sampling (k=5): 30 сек - Без negative sampling: 5+ минут
```
---

**Другие трюки Word2Vec:**

**1. Subsampling (субдискретизация) частых слов**

Проблема: слова типа "the", "a", "is" встречаются везде, мало информации.

Решение: случайно удаляем частые слова с вероятностью:

P(discard)=1−tf(w)P(discard)=1−f(w)t

где:

- t - порог (обычно 10^-5)
    
- f(w) - частота слова
    

```python
model = Word2Vec(
    sentences,
    sg=1,
    sample=1e-5,  # Subsampling порог
)

# Теперь слово "the" может быть пропущено с вероятностью ~80%
# Слово "cat" пропускается с вероятностью 0%
```

**2. Hierarchical Softmax (иерархический softmax)**

Вместо softmax по всему словарю, используем бинарное дерево (Huffman tree).

Каждое слово - лист дерева. Вероятность = произведение вероятностей на пути до листа.

```python
model = Word2Vec(
    sentences,
    sg=1,
    hs=1,        # включаем hierarchical softmax
    negative=0,  # отключаем negative sampling
)

# Скорость: немного медленнее, чем negative sampling, но быстрее softmax
# Использование памяти: выше, чем при negative sampling
```

**3. Phrase detection (обнаружение фраз)**

Некоторые слова часто встречаются вместе и должны рассматриваться как одно слово.

```python
from gensim.models.phrases import Phrases

# Находим частые фразы 
phrases = Phrases(sentences, min_count=5, threshold=100) 
bigram = phrases[sentences]  # Заменяет "new york" → "new_york" 
# Обучаем Word2Vec на измененных предложениях 
model = Word2Vec(bigram, sg=1, vector_size=300) # Теперь "new_york" - один вектор, а не два 
embedding = model.wv['new_york']`
```

**4. Dynamic context window (динамическое окно контекста)**

Вместо фиксированного окна (window=5), случайно выбираем размер окна.


```python
# Фиксированное окно = 5
model_fixed = Word2Vec(
    sentences,
    sg=1,
    window=5,
    shrink_windows=False,  # окно всегда = 5
)

# Динамическое окно (случайно от 1 до 5)
model_dynamic = Word2Vec(
    sentences,
    sg=1,
    window=5,
    shrink_windows=True,
)

# Динамическое окно часто работает лучше, особенно для редких слов
```

---

**Практическое применение:**

```python
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases

# 1. Подготовка данных
sentences = [sent.split() for sent in raw_texts]

# 2. Обнаружение фраз
phrases = Phrases(sentences, min_count=5, threshold=100)
sentences = [phrases[sent] for sent in sentences]

# 3. Обучение Word2Vec с оптимальными параметрами
model = Word2Vec(
    sentences=sentences,
    sg=1,              # Skip-gram (лучше для качества)
    vector_size=300,   # стандартная размерность
    window=5,          # окно контекста
    min_count=5,       # игнорируем редкие слова
    negative=5,        # negative sampling
    sample=1e-5,       # subsampling частых слов
    epochs=5,
    workers=4,
    shrink_windows=True,
)

# 4. Использование обученной модели
similar_words = model.wv.most_similar('word', topn=10)
analogy = model.wv.most_similar(positive=['king', 'woman'], negative=['man'])
```

---

#### Вопрос: Что такое dense и sparse эмбеддинги? Приведите примеры.

**Ответ:**

**Sparse (Разреженные) эмбеддинги**

Вектор в основном состоит из нулей, только несколько позиций = 1.

Пример 1: One-hot encoding

```python
vocab = ['cat', 'dog', 'bird', 'fish'] 
# Слово "cat" cat_embedding = [1, 0, 0, 0]  # shape: (4,) 
# Слово "dog" dog_embedding = [0, 1, 0, 0]  # shape: (4,) 
# 75% нулей → sparse!
```

Пример 2: Bag of Words (BoW)

```
Sentence: "The cat sat on the mat"
BoW embedding (TF counts):
    {'the': 2, 'cat': 1, 'sat': 1, 'on': 1, 'mat': 1, 'dog': 0, 'bird': 0, ...}
Vector form: [2, 1, 1, 1, 1, 0, 0, 0, 0, ...]  # shape: (10 000 vocab size)
# 99% значений равны нулю → очень sparse!
```

Пример 3: TF-IDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)  # shape: (100 docs, 10000 vocab) # На каждый документ в среднем 100 ненулевых значений из 10000 # 99% нулей → sparse
```

Свойства sparse:

- Очень high-dimensional (размер = vocab size)
    
- Интерпретируемые (какие слова важны)
    
- Fast на CPU
    
- Неэффективны для нейросетей
    

---

**Dense (Плотные) эмбеддинги**

Вектор содержит значения на всех позициях, мало нулей.

Пример 1: Word2Vec

```python
from gensim.models import Word2Vec

model = Word2Vec(sentences, vector_size=300)
cat_embedding = model.wv['cat']  # shape: (300,)
dog_embedding = model.wv['dog']  # shape: (300,)

print(cat_embedding[:5])  # [0.123, -0.456, 0.789, 0.234, ...]
print(dog_embedding[:5])  # [0.145, -0.423, 0.801, 0.256, ...]
```

Пример 2: BERT

```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

inputs = tokenizer('cat', return_tensors='pt')
outputs = model(**inputs)

embedding = outputs.last_hidden_state[0, 0]  # shape: (768,)
print(embedding[:5])  # [0.234, -0.567, 0.891, ...]
```

Свойства dense:

- Low-dimensional (300-1000)
    
- Не интерпретируемые (что значит позиция 42?)
    
- Semantic (похожие слова имеют похожие векторы)
    
- Good для нейросетей
    

---

**Сравнение:**

|Аспект|Sparse|Dense|
|---|---|---|
|Размерность|10k-1M|100-1000|
|Заполнение нулями|99%+|0-5%|
|Интерпретируемость|Высокая|Низкая|
|Скорость (CPU)|Быстро|Медленно|
|Скорость (GPU/нейросеть)|Медленно|Быстро|
|Захват семантики|Плохо|Хорошо|
|Размер памяти|Большой|Маленький|

**Таблица с примерами:**

|Метод|Type|Пример|
|---|---|---|
|One-hot|Sparse|[askpython](https://www.askpython.com/python/examples/tf-idf-model-from-scratch)​ (vocab=4)|
|Bag of Words|Sparse|[2, 1, 1, 0, 0, ...] (vocab=10k)|
|TF-IDF|Sparse|[0.5, 0.3, 0.2, 0, ...] (vocab=10k)|
|Co-occurrence|Sparse|word-word матрица (vocab × vocab)|
|Word2Vec|Dense|[0.123, -0.456, 0.789, ...] (300,)|
|FastText|Dense|[0.145, -0.423, 0.801, ...] (300,)|
|GloVe|Dense|[0.234, -0.567, 0.891, ...] (300,)|
|BERT|Dense|[0.234, -0.567, 0.891, ...] (768,)|

**Когда что использовать:**

```python
if need_interpretability and not has_gpu:
    embeddings = TfidfVectorizer()  # Sparse, легко объяснять
elif need_quality and has_gpu:
    embeddings = AutoModel.from_pretrained('bert-base-uncased')  # Dense, max качество
elif need_balance:
    embeddings = FastText(sentences)  # Dense, хорошее качество при умеренной цене
else:
    embeddings = Word2Vec(sentences)
```

---

#### Вопрос: Почему может быть важна размерность эмбеддинга?

**Ответ:**

Размерность эмбеддинга - это количество элементов в векторе (например, 300 для Word2Vec, 768 для BERT).

**Маленькая размерность (50-100)**

Плюсы:

- Быстрое обучение
    
- Малое использование памяти
    
- Быстрый inference
    
- Меньше параметров → меньше риск переобучения
    

Минусы:

- Не хватает "места" для представления сложной семантики
    
- Потеря информации
    
- Худшее качество на сложных задачах
    

```python
# Маленькая размерность
model_small = Word2Vec(sentences, vector_size=50, epochs=10)
print('Обучение: 10 сек, память: 1 MB, качество: низкое на сложных задачах')
```

**Средняя размерность (300-500)**

Плюсы:

- Балансирует качество и скорость
    
- Стандарт в индустрии (GloVe, FastText, Word2Vec default)
    
- Хорошее качество на большинстве задач
    

Минусы:

- Может быть недостаточно для очень сложных моделей
    

```python
# Стандартная размерность
model_standard = Word2Vec(sentences, vector_size=300, epochs=10)
print('Обучение: 1 минута, память: ~10 MB, качество: хорошее на большинстве задач')
```

**Большая размерность (768-2048)**

Плюсы:

- Больше информации в каждом векторе
    
- Лучше на очень сложных задачах
    
- Используется в BERT, GPT
    

Минусы:

- Медленно
    
- Много памяти
    
- Риск переобучения с маленькими корпусами
    
- Может быть избыточно для простых задач
    

```python
# Большая размерность
large_configs = {
    'BERT-base': 768,
    'GPT-2': 1024,
    'GPT-3': 2048,
}

for model_name, dim in large_configs.items():
    print(f'{model_name}: vector_size={dim}')

print('Обучение: часы, память: 100+ MB, качество высокое, но может быть избыточно')
```

---

**Как выбрать размерность?**

Эмпирическое правило (из гугл Paper):

dim≈4⋅log⁡10(V)dim≈4⋅log10(V)

где V - размер словаря.

```python
import math

vocab_sizes = [10_000, 100_000, 1_000_000]

for vocab_size in vocab_sizes:
    recommended_dim = 4 * math.log10(vocab_size)
    print(
        f"V = {vocab_size:,} → dim ≈ {recommended_dim:.0f}"
    )

# Пример выбора диапазона
# 10k слов → ~133 → выбираем 150-200
# 100k слов → ~177 → выбираем 200-300
# 1M слов → ~221 → выбираем 300-500
```

---

**Проблема: Curse of Dimensionality (проклятие размерности)**

При росте размерности:

> Объём пространства растёт экспоненциально:
> - dim = 1 → отрезок (10 точек)
> - dim = 10 → гиперкуб (10^10 точек)
> - dim = 100 → гиперкуб (10^100 точек)
>
> Нужно экспоненциально больше данных, чтобы заполнить пространство!

На практике:

- 50 dim: нужно 1000 примеров на слово
    
- 300 dim: нужно 10000 примеров на слово
    
- 768 dim: нужно 100000 примеров на слово
    

```python
def choose_vector_size(corpus_size: int) -> int:
    """Предлагаем размерность эмбеддинга в зависимости от размера корпуса."""
    if corpus_size < 50_000:
        return 100  # малый корпус (~100 слов, 10k примеров)
    if corpus_size < 10_000_000:
        return 300  # средний корпус (~10k слов, 1M примеров)
    return 500      # большой корпус (>100k слов, 10B+ примеров)


print(choose_vector_size(40_000))
print(choose_vector_size(5_000_000))
print(choose_vector_size(50_000_000))
```

---

**Эмпирические наблюдения:**

```python
# Benchmark: классификация текстов
accuracy_by_dim = {
    50: 0.75,
    100: 0.82,
    200: 0.85,
    300: 0.86,   # прирост замедляется
    500: 0.865,  # небольшой прирост
    1000: 0.867, # почти без изменений
}

for dim, score in accuracy_by_dim.items():
    print(f'vector_size={dim:<4} → accuracy={score:.3f}')
```

---

**Общая рекомендация:**

```python
# Быстрое решение (день)
quick = Word2Vec(sentences, vector_size=100)

# Production (неделя)
production = Word2Vec(sentences, vector_size=300)

# State-of-the-art (месяц)
sota = AutoModel.from_pretrained('bert-base-uncased')
```

---

#### Вопрос: Какие проблемы могут возникнуть при обучении Word2Vec на коротких текстовых данных, и как можно с ними справиться?

**Ответ:**

Обучение Word2Vec на коротких текстах (например, заголовки, комментарии, твиты) имеет специфические проблемы.

**Проблема 1: Малый контекст**

Проблема:

> Обычный текст: "The quick brown fox jumps over the lazy dog" — при окне 5 вокруг слова много контекста.  
> Короткий текст: "quick fox jumps" — то же окно выходит за границы, информации мало.

На коротких текстах модель видит мало контекста для каждого слова.

Решение:

```python
# Уменьшаем окно контекста для коротких текстов
model = Word2Vec(
    short_sentences,
    sg=1,
    window=2,      # вместо 5
    min_count=1,   # учитываем редкие слова
    epochs=20,     # больше эпох для малых корпусов
)
# Меньшее окно лучше соответствует длине текстов
```

**Проблема 2: Редкие слова и спарсовность**

Проблема:

> Большой корпус (100M слов):
> - "cat" встречается 10 000 раз → хороший embedding.
> - "fluffy_cat_breed" встречается 5 раз → embedding плохой.
>
> Малый корпус (1M слов):
> - "cat" встречается 100 раз → средний embedding.
> - "fluffy_cat_breed" не встречается → embedding отсутствует.

Решение:

```python
# 1. FastText использует символьные n-grams и умеет работать с редкими словами
from gensim.models import FastText

fasttext_model = FastText(
    short_sentences,
    sg=1,
    vector_size=300,
    window=2,
    min_count=1,
    epochs=20,
)
unknown = fasttext_model.wv['unknown_word_xyz']  # не вызывает ошибку

# 2. Расширяем данные синонимами
augmented_texts = augment_with_synonyms(short_texts, synonyms)

# 3. Используем pre-trained embeddings и дообучаем
from gensim.downloader import load

pretrained = load('word2vec-google-news-300')
```

**Проблема 3: Недостаточно данных для обучения**

Проблема:

> Рекомендация: vector_size = 4 × log10(vocab_size).  
> - 1 000 слов → dim ≈ 16.  
> - 10 000 слов → dim ≈ 53.  
> - 100 000 слов → dim ≈ 133.  
> Часто берут dim = 300, что может переобучать малый корпус.

Решение:

```python
# 1. Уменьшаем размерность для малых корпусов
small_model = Word2Vec(
    short_sentences,
    vector_size=100,
    window=2,
    min_count=2,
    epochs=20,
)

# 2. Более сильная регуляризация через subsampling и negative sampling
regularized_model = Word2Vec(
    short_sentences,
    vector_size=300,
    sample=1e-4,
    negative=15,
    epochs=20,
)

# 3. Дообучение предобученной модели
pretrained = Word2Vec.load('pretrained_model.bin')
pretrained.train(short_sentences, epochs=5, total_examples=pretrained.corpus_count)
```

**Проблема 4: Низкая частота слов**

Проблема:

> На коротких текстах многие слова встречаются 1–2 раза:
> - "apple" → 50 вхождений.  
> - "orange" → 2 вхождения.  
> - "banana" → 1 вхождение.  
> Качество embedding обратно пропорционально частоте.

Решение:

```python
# 1. Понижаем min_count и увеличиваем epochs
rare_words_model = Word2Vec(
    short_sentences,
    min_count=1,
    epochs=30,
)

# 2. Используем символьные n-grams (FastText)
fasttext_char = FastText(
    short_sentences,
    min_count=1,
    word_ngrams=1,
)

# 3. Обнаруживаем фразы, чтобы увеличить частоты
phrases = Phrases(short_sentences, min_count=2, threshold=10)
augmented = [phrases[sent] for sent in short_sentences]
phrase_model = Word2Vec(augmented, min_count=1, epochs=20)
```

**Проблема 5: Нестабильное обучение**

Проблема:

> На маленьком корпусе обучение нестабильно: loss прыгает, качество колеблется, разные запуски дают разные результаты.

Решение:

```python
# 1. Увеличиваем batch size и фиксируем seed
stable_model = Word2Vec(
    short_sentences,
    batch_words=50_000,
    epochs=20,
    seed=42,
)

# 2. Уменьшаем learning rate
low_lr_model = Word2Vec(
    short_sentences,
    alpha=0.01,
    min_alpha=0.0001,
    epochs=20,
)

# 3. Больше эпох с плавным снижением шага
long_train_model = Word2Vec(
    short_sentences,
    epochs=50,
    alpha=0.025,
    min_alpha=0.00025,
)
```

---

**Комплексное решение для коротких текстов:**

```python
from gensim.models import FastText
from gensim.models.phrases import Phrases

# 1. Подготовка данных
sentences = [text.split() for text in short_texts]

# 2. Обнаружение фраз
phrases = Phrases(sentences, min_count=2, threshold=10)
sentences = [phrases[sent] for sent in sentences]

# 3. Обучение FastText, оптимизированного под короткие тексты
model = FastText(
    sentences=sentences,
    sg=1,
    vector_size=200,
    window=2,
    negative=15,
    sample=1e-4,
    min_count=1,
    epochs=30,
    alpha=0.025,
    min_alpha=0.0001,
    workers=4,
    seed=42,
)

# 4. Использование эмбеддингов
similar = model.wv.most_similar('fox', topn=5)
embedding = model.wv['fox']  # shape: (200,)
```

Итоговые рекомендации для коротких текстов:

```python
# Рекомендации для коротких текстов:
# - вместо Word2Vec → FastText (лучше для редких слов)
# - вместо window=5 → window=2
# - вместо vector_size=300 → 100-200
# - epochs увеличиваем до 20-50
# - min_count=1, чтобы не терять редкие слова
# - subsampling=1e-4 для агрессивного удаления шумовых слов
```


## Рекуррентные и сверточные сети в NLP

---

#### Вопрос: Сколько обучающих параметров в простой 1-слойной RNN?

**Ответ:**

Для простой "Vanilla" RNN формулы обновления скрытого состояния выглядят так:

ht=tanh⁡(Wxhxt+Whhht−1+bh)ht=tanh(Wxhxt+Whhht−1+bh)  
yt=Whyht+byyt=Whyht+by

Где:

- $x_t \in \mathbb{R}^{d_{in}}$ — входной вектор (размерность input_dim)
    
- $h_t \in \mathbb{R}^{d_{hidden}}$ — скрытое состояние (размерность hidden_dim)
    
- $y_t \in \mathbb{R}^{d_{out}}$ — выходной вектор (размерность output_dim)
    

**Подсчет параметров:**

1. **Input-to-Hidden матрица ($W_{xh}$):**  
    Размер: `(hidden_dim, input_dim)`  
    Параметров: `hidden_dim * input_dim`
    
2. **Hidden-to-Hidden матрица ($W_{hh}$):**  
    Размер: `(hidden_dim, hidden_dim)`  
    Параметров: `hidden_dim * hidden_dim`
    
3. **Hidden Bias ($b_h$):**  
    Размер: `(hidden_dim)`  
    Параметров: `hidden_dim`
    

**Итого для скрытого слоя:**  
N=h⋅i+h⋅h+h=h(i+h+1)N=h⋅i+h⋅h+h=h(i+h+1)

Если считать с выходным слоем (для классификации, например):

4. **Hidden-to-Output матрица ($W_{hy}$):**  
    Размер: `(output_dim, hidden_dim)`  
    Параметров: `output_dim * hidden_dim`
    
5. **Output Bias ($b_y$):**  
    Размер: `(output_dim)`  
    Параметров: `output_dim`
    

**Пример на Python:**

```python
import torch
import torch.nn as nn  # [web:77]


def create_rnn(
    input_dim: int,
    hidden_dim: int,
    bias: bool = True,
) -> nn.RNN:
    """
    Создаёт RNN с заданными параметрами.
    
    Формула количества параметров:
    W_xh: hidden_dim × input_dim
    W_hh: hidden_dim × hidden_dim  
    bias_ih: hidden_dim (input bias)
    bias_hh: hidden_dim (hidden bias)
    
    Итого: hidden_dim × (input_dim + hidden_dim + 2)
    """
    rnn = nn.RNN(
        input_size=input_dim,
        hidden_size=hidden_dim,
        bias=bias,
    )
    
    return rnn


def count_parameters(model: nn.Module) -> int:
    """Подсчитывает общее количество параметров модели."""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


def print_param_breakdown(rnn: nn.RNN) -> None:
    """Выводит разбивку параметров по тензорам."""
    print("Разбивка параметров RNN:")
    for name, param in rnn.named_parameters():
        print(f"  {name}: {param.numel():4d}")
    
    total = count_parameters(rnn)
    print(f"\nИтого параметров: {total:,}")


def main() -> None:
    # Пример: input_dim=10, hidden_dim=20
    input_dim = 10
    hidden_dim = 20
    
    rnn = create_rnn(input_dim, hidden_dim)
    
    # Теоретический расчёт:
    theory_params = hidden_dim * (input_dim + hidden_dim + 2)
    print(f"Теоретически: {theory_params:,}")
    
    # Фактический подсчёт:
    actual_params = count_parameters(rnn)
    print(f"Фактически:    {actual_params:,}")
    assert theory_params == actual_params
    
    print_param_breakdown(rnn)
    
    # Проверка формулы на разных размерах
    print("\nПроверка на разных размерах:")
    for input_d, hidden_d in [(5, 10), (100, 50), (3, 128)]:
        rnn_test = create_rnn(input_d, hidden_d)
        params_test = count_parameters(rnn_test)
        theory_test = hidden_d * (input_d + hidden_d + 2)
        print(f"input={input_d}, hidden={hidden_d} → {params_test:,} (✓ {theory_test:,})")


if __name__ == "__main__":
    main()
```

---

#### Вопрос: Как обучается RNN?

**Ответ:**

RNN обучается с помощью алгоритма **BPTT (Backpropagation Through Time)**.

**Процесс:**

1. **Unrolling (Развертывание):**  
    Сеть разворачивается во времени. Если длина последовательности $T=5$, мы как бы создаем 5 копий сети, связанных друг с другом.
    
    $x_1 \xrightarrow{} h_1 \xrightarrow{} h_2 \dots \xrightarrow{} h_5$
    
2. **Forward Pass (Прямой проход):**  
    Вычисляем скрытые состояния $h_t$ и выходы $y_t$ для каждого шага $t$ от 1 до $T$.  
    Считаем Loss на каждом шаге (если нужно) или в конце.
    
    Ltotal=∑t=1TL(yt,targett)Ltotal=∑t=1TL(yt,targett)
    
3. **Backward Pass (Обратный проход):**  
    Градиенты текут от последнего момента времени $T$ к первому $1$.  
    Градиент ошибки для весов $W_{hh}$ — это сумма градиентов на каждом временном шаге (так как веса одни и те же для всех шагов).
    
    ∂L∂W=∑t=1T∂Lt∂W∂W∂L=∑t=1T∂W∂Lt
    
---

#### Вопрос: Какие проблемы есть в RNN?

**Ответ:**

**1. Vanishing Gradient (Затухание градиента)**  
Самая большая проблема. При обратном проходе градиент умножается на матрицу весов $W_{hh}$ много раз (по числу шагов времени).

- Если собственные числа матрицы < 1, градиент стремится к 0.
    
- Сеть перестает учиться на далеких зависимостях (забывает начало предложения).
    

**2. Exploding Gradient (Взрыв градиента)**

- Если собственные числа > 1, градиент растет экспоненциально.
    
- Веса становятся `NaN` или бесконечностью.
    

**3. Отсутствие параллелизации**

- Нельзя вычислить $h_t$, пока не посчитан $h_{t-1}$.
    
- Обучение медленное по сравнению с CNN или Transformer, которые могут обрабатывать всю последовательность параллельно.
    

**4. Short-term memory**

- Из-за затухания градиента Vanilla RNN плохо помнит контекст длиннее 10-20 шагов.
    

---

#### Вопрос: Какие виды RNN сетей вы знаете? Объясните разницу между GRU и LSTM?

**Ответ:**

**Виды RNN:**

1. **Vanilla RNN:** Простая ячейка с $\tanh$.
    
2. **LSTM (Long Short-Term Memory):** Сложная ячейка с 3 гейтами и отдельной памятью (Cell state).
    
3. **GRU (Gated Recurrent Unit):** Упрощенная версия LSTM с 2 гейтами.
    
4. **Bidirectional RNN:** Два слоя — один читает слева направо, другой справа налево.
    

**Разница LSTM vs GRU:**

|Характеристика|LSTM|GRU|
|---|---|---|
|**Гейты**|3 (Input, Forget, Output)|2 (Reset, Update)|
|**Память**|Cell state ($c_t$) + Hidden state ($h_t$)|Только Hidden state ($h_t$)|
|**Параметры**|Больше (сложнее учить)|Меньше (быстрее учить)|
|**Скорость**|Медленнее|Быстрее|
|**Данные**|Лучше на очень длинных последовательностях|Лучше на небольших датасетах|

**Структура LSTM:**

- **Forget Gate:** Что забыть из старой памяти?
    
- **Input Gate:** Что записать нового в память?
    
- **Output Gate:** Что показать наружу (в скрытое состояние)?
    

**Структура GRU:**

- **Reset Gate:** Насколько игнорировать прошлое при расчете нового кандидата?
    
- **Update Gate:** Баланс между старым состоянием и новым кандидатом (аналог Forget + Input).
    

**Код на Python (сравнение):**

```python
import torch
import torch.nn as nn


def create_lstm(
    input_size: int = 10,
    hidden_size: int = 20,
    bias: bool = True,
) -> nn.LSTM:
    """Создаёт LSTM с 4 врата (input, forget, cell, output)."""
    return nn.LSTM(input_size, hidden_size, bias=bias)


def create_gru(
    input_size: int = 10,
    hidden_size: int = 20,
    bias: bool = True,
) -> nn.GRU:
    """Создаёт GRU с 3 врата (reset, update, new)."""
    return nn.GRU(input_size, hidden_size, bias=bias)


def count_parameters(model: nn.Module) -> int:
    """Подсчитывает общее количество параметров."""
    return sum(p.numel() for p in model.parameters())


def print_param_breakdown(model: nn.Module, model_name: str) -> None:
    """Детальная разбивка параметров."""
    print(f"\n{model_name} — разбивка параметров:")
    for name, param in model.named_parameters():
        print(f"  {name:20}: {param.shape:15} = {param.numel():4d}")
    
    total = count_parameters(model)
    print(f"  {'Итого':20}: {'':15} = {total:4d}")


def compare_rnn_variants(input_size: int = 10, hidden_size: int = 20) -> None:
    """
    Сравнивает количество параметров RNN, LSTM, GRU.
    
    Формулы (с bias):
    RNN: 1 × (input_size + hidden_size + 2) × hidden_size
    GRU: 3 × (input_size + hidden_size + 2) × hidden_size  
    LSTM: 4 × (input_size + hidden_size + 2) × hidden_size
    """
    
    # Создаём модели
    rnn = nn.RNN(input_size, hidden_size)
    lstm = create_lstm(input_size, hidden_size)
    gru = create_gru(input_size, hidden_size)
    
    # Подсчёт параметров
    rnn_params = count_parameters(rnn)
    lstm_params = count_parameters(lstm)
    gru_params = count_parameters(gru)
    
    # Проверка формул
    gate_factor = {"RNN": 1, "LSTM": 4, "GRU": 3}
    base = hidden_size * (input_size + hidden_size + 2)
    
    print(f"\n{'='*60}")
    print(f"СРАВНЕНИЕ RNN/LSTM/GRU (input={input_size}, hidden={hidden_size})")
    print(f"{'='*60}")
    
    for model_type, params in [("RNN", rnn_params), ("LSTM", lstm_params), ("GRU", gru_params)]:
        theory = gate_factor[model_type] * base
        print(f"{model_type:4}: {params:5,} параметров (теория: {theory:5,}) ✓")
    
    print(f"\nLSTM / GRU = {lstm_params / gru_params:.1f}x больше параметров")
    print_param_breakdown(lstm, "LSTM")
    print_param_breakdown(gru, "GRU")


def main() -> None:
    compare_rnn_variants()
    
    # Тестируем на разных размерах
    sizes = [(5, 10), (100, 128), (20, 50)]
    print("\n" + "="*60)
    for input_size, hidden_size in sizes:
        print(f"\nРазмер input={input_size}, hidden={hidden_size}:")
        lstm = create_lstm(input_size, hidden_size)
        gru = create_gru(input_size, hidden_size)
        print(f"  LSTM: {count_parameters(lstm):,} | GRU: {count_parameters(gru):,}")


if __name__ == "__main__":
    main()
```

---

#### Вопрос: Какие параметры мы можем тюнить в такого вида сетях?

**Ответ:**

Основные гиперпараметры для настройки RNN/LSTM/GRU:

1. **Hidden Size (Размер скрытого слоя):**  
    Определяет "емкость" памяти. Больше = лучше память, но риск переобучения.  
    _Типичные значения:_ 128, 256, 512, 1024.
    
2. **Number of Layers (Количество слоев):**  
    Stacked RNN (многослойные). Позволяют учить более сложные абстракции.  
    _Типичные значения:_ 1-3 (глубокие RNN учатся плохо из-за градиентов).
    
3. **Bidirectional (Двунаправленность):**  
    `True/False`. Если важно видеть будущий контекст (для классификации текста, NER), используем `True`.  
    Удваивает количество параметров.
    
4. **Dropout:**  
    Для регуляризации. Обычно применяется между слоями RNN (не внутри рекурсии, хотя есть и Recurrent Dropout).
    
5. **Batch Size & Sequence Length:**  
    Длина последовательности при обучении (BPTT length). Влияет на то, как далеко назад учится сеть.
    

---

#### Вопрос: Что такое затухающие градиенты для RNN? И как вы решаете эту проблему?

**Ответ:**

**Проблема:**  
При обратном распространении ошибки градиент проходит через операцию умножения на одну и ту же матрицу весов $W$ много раз ($t$ раз).  
∂L∂h0∝Wt∂h0∂L∝Wt  
Если веса маленькие (собственные числа < 1), то $W^t \to 0$. Градиент исчезает, веса в начале сети не обновляются.

**Решения:**

1. **Архитектурные изменения:**  
    Использовать **LSTM** или **GRU**. Их механизм гейтов (особенно Forget gate = 1) позволяет градиенту течь беспрепятственно через "Gradient Superhighway".
    
2. **Gradient Clipping (для взрыва градиента):**  
    Если норма градиента превышает порог (например, 1.0), мы его обрезаем.
    
    ```python
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    ```
    
3. **Инициализация весов:**  
    Использовать Orthogonal initialization или Xavier initialization.  
    Инициализировать bias гейта забывания (forget gate bias) положительным числом (например, 1.0), чтобы в начале обучения сеть помнила всё.
    
4. **Функции активации:**  
    Использовать **ReLU** вместо $\tanh$ или Sigmoid (но осторожно, может привести к взрыву градиента).
    
5. **Skip Connections (Residual connections):**  
    Добавлять связи через слои, как в ResNet.
    

---

#### Вопрос: Зачем в NLP Convolutional Neural Network и как вы его можете использовать? С чем вы можете сравнить CNN в рамках парадигмы attention?

**Ответ:**

**Зачем CNN в NLP:**  
CNN отлично находят **локальные паттерны** (n-граммы), независимо от их позиции в тексте.

- Фильтр размером 2 видит биграммы.
    
- Фильтр размером 3 видит триграммы.
    

**Применение:**

- **Классификация текстов:** Определение тональности, темы. Быстро и эффективно находит ключевые фразы ("отличный фильм", "ужасный сервис").
    
- **Character-level модели:** Работа с опечатками, морфологией (например, в FastText или ELMo char-CNN).
    

**CNN vs Attention:**

**CNN:**

- **Local Attention:** Смотрит только на фиксированное окно (размер ядра).
    
- **Hard-coded:** Веса фильтра фиксированы после обучения, окно всегда одно и то же.
    
- **Position-invariant:** Находит паттерн "not good" в начале или конце одинаково.
    

**Self-Attention:**

- **Global Attention:** Каждое слово смотрит на _все_ остальные слова.
    
- **Dynamic:** Веса внимания зависят от самих данных (query-key interaction).
    
- **Flexible:** Может обратить внимание на слово в начале и в конце одновременно.
    

**Сравнение:**  
CNN можно рассматривать как **фиксированный локальный Multi-Head Attention**.  
Каждый фильтр CNN — это как одна "голова", которая смотрит только на соседей.

**Пример TextCNN на Python:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """
    TextCNN для классификации текста (Kim, 2014).
    Использует несколько сверточных фильтров разного размера для захвата n-грамм.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int,
        num_filters: int = 100,
        filter_sizes: list[int] = [3, 4, 5],
        dropout: float = 0.5,
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Сверточные слои для разных размеров фильтров
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim)) 
            for k in filter_sizes
        ])
        
        # Полносвязный слой (num_filters * len(filter_sizes))
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        embedded = embedded.unsqueeze(1)  # [batch_size, 1, seq_len, embed_dim]
        
        # Свертка → ReLU → MaxPool1d по временной оси
        conv_outs = []
        for conv in self.convs:
            conved = F.relu(conv(embedded))  # [batch, num_filters, seq_len-k+1, 1]
            pooled = F.max_pool1d(conved.squeeze(3), conved.size(2)).squeeze(2)  # [batch, num_filters]
            conv_outs.append(pooled)
        
        # Конкатенация по фильтрам
        cat_features = torch.cat(conv_outs, dim=1)  # [batch, num_filters * num_convs]
        cat_features = self.dropout(cat_features)
        
        return self.fc(cat_features)  # [batch, num_classes]


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Подсчёт параметров модели."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    breakdown = {
        "embedding": sum(p.numel() for p in model.embedding.parameters()),
        "convs": sum(p.numel() for conv in model.convs for p in conv.parameters()),
        "fc": sum(p.numel() for p in model.fc.parameters()),
        "total": total_params,
    }
    return breakdown


def main() -> None:
    # Создаём модель
    model = TextCNN(
        vocab_size=10_000,
        embed_dim=300,
        num_classes=2,  # бинарная классификация
    )
    
    # Подсчёт параметров
    params = count_parameters(model)
    print("Параметры TextCNN:")
    for name, count in params.items():
        print(f"  {name:10}: {count:,}")
    
    # Тест forward pass
    batch_size, seq_len = 32, 100
    x = torch.randint(0, 10_000, (batch_size, seq_len))
    output = model(x)
    print(f"\nForward pass: input [{x.shape}] → output [{output.shape}] ✓")
    
    # Пример батча
    print(f"\nПример работы:")
    sample_input = torch.randint(0, 10_000, (1, 20))
    logits = model(sample_input)
    probs = F.softmax(logits, dim=1)
    print(f"  Input:  {sample_input.shape}")
    print(f"  Logits: {logits}")
    print(f"  Probs:  {probs}")


if __name__ == "__main__":
    main()
```

## Attention and Transformer Architecture

---

#### Вопрос: Как считаете attention? (доп. для какой задачи его предложили? и почему?)

**Ответ:**

Механизм Attention изначально предложили для задачи **машинного перевода** (Bahdanau et al., 2014), чтобы решить проблему "бутылочного горлышка" в Seq2Seq моделях.  
В классической Seq2Seq (Encoder-Decoder RNN) весь смысл предложения сжимался в **один вектор фиксированной длины** (context vector). Для длинных предложений это работало плохо.

Attention позволяет декодеру "смотреть" на **все** скрытые состояния энкодера на каждом шаге генерации, выбирая наиболее важные части.

**Как считать (Scaled Dot-Product Attention):**

В основе лежат 3 вектора: **Query (Q)**, **Key (K)**, **Value (V)**.

- **Query:** что мы ищем? (текущее состояние декодера или токена)
    
- **Key:** что мы предлагаем? (ключи всех токенов энкодера)
    
- **Value:** что мы отдаем? (значения/информация токенов энкодера)
    

**Формула:**

Attention(Q,K,V)=softmax(QKTdk)VAttention(Q,K,V)=softmax(dkQKT)V

1. **MatMul ($QK^T$):** Считаем скалярное произведение Query с каждым Key. Получаем "очки схожести" (scores).
    
2. **Scale ($\frac{1}{\sqrt{d_k}}$):** Делим на корень из размерности ключей. Это нужно, чтобы при больших размерностях скалярное произведение не взрывалось, уводя softmax в зону с нулевыми градиентами.
    
3. **Softmax:** Превращаем очки в вероятности (веса внимания), сумма = 1.
    
4. **MatMul ($... \times V$):** Умножаем веса на Value. Получаем взвешенную сумму значений.
    

---

#### Вопрос: Сложность attention? Сравните с сложностью в RNN?

**Ответ:**

Пусть $n$ — длина последовательности, $d$ — размерность вектора.

**Self-Attention:**

- Сложность: $O(n^2 \cdot d)$
    
- Почему: Нам нужно перемножить каждый токен с каждым ($n \times n$ матрица внимания).
    
- Параллелизация: $O(1)$ (все токены обрабатываются одновременно).
    

**RNN:**

- Сложность: $O(n \cdot d^2)$
    
- Почему: На каждом шаге мы умножаем матрицу весов ($d \times d$) на вектор состояния.
    
- Параллелизация: $O(n)$ (нельзя, так как $h_t$ зависит от $h_{t-1}$).
    

**Сравнение:**

- Если $n < d$ (короткие предложения, большие эмбеддинги): Attention быстрее.
    
- Если $n > d$ (очень длинные тексты): RNN вычислительно эффективнее, но Attention параллелится лучше.
    

Именно квадратичная сложность по длине последовательности ($n^2$) — главная проблема Трансформеров для длинных текстов.

---

#### Вопрос: Сравните RNN и Attention? В каком случае будете использовать Attention, а когда RNN?

**Ответ:**

**RNN:**

- **Последовательная обработка:** слово за словом.
    
- **Память:** Сжата в скрытое состояние (hidden state).
    
- **Проблемы:** Забывает начало длинного текста (vanishing gradient), не параллелится.
    
- **Плюсы:** Эффективна для бесконечных потоков данных, малая память при генерации.
    

**Attention (Transformer):**

- **Параллельная обработка:** видит все предложение сразу.
    
- **Память:** Прямой доступ к любому слову в прошлом (расстояние = 1 шаг).
    
- **Проблемы:** Квадратичная память $O(n^2)$.
    
- **Плюсы:** Идеально ловит длинные зависимости, очень быстро учится.
    

**Когда что использовать:**

- **Transformer:** Практически всегда для NLP (BERT, GPT). Перевод, классификация, генерация текста.
    
- **RNN (LSTM/GRU):**
    
    - Для Time Series (временных рядов), где важна рекуррентная природа.
        
    - Для streaming (потоковой обработки) на слабых устройствах.
        
    - Когда последовательность очень длинная, а память ограничена (Linear RNNs, RWKV).
        

---

#### Вопрос: Напишите attention с нуля.

**Ответ:**

```python
import torch
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Классическая Scaled Dot-Product Attention (Vaswani et al., 2017).
    
    Args:
        query, key, value: [batch_size, num_heads, seq_len, d_k]
        mask: булева маска (True=attend, False=ignore) или float маска
        dropout_p: вероятность dropout для attention weights
        is_causal: применять треугольную маску для autoregressive моделей
    
    Returns:
        output: [batch_size, num_heads, seq_len, d_v]
        attention_weights: [batch_size, num_heads, seq_len, seq_len]
    """
    d_k = query.size(-1)
    
    # 1. Q @ K^T → scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 2. Маскирование
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    if is_causal:
        # Треугольная маска для causal attention
        seq_len = scores.size(-1)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device))
        scores = scores.masked_fill(causal_mask == 0, -1e9)
    
    # 3. Softmax
    attention_weights = F.softmax(scores, dim=-1)
    
    # 4. Dropout (только при обучении)
    if dropout_p > 0.0 and torch._C._get_tracing_state():
        attention_weights = F.dropout(attention_weights, p=dropout_p)
    
    # 5. Attention @ V
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights


# 🔥 Современный вариант с PyTorch 2.0+ встроенной функцией
def scaled_dot_product_attention_modern(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Использует torch.nn.functional.scaled_dot_product_attention (PyTorch 2.0+).
    Автоматически выбирает оптимальный backend (FlashAttention, etc.).
    """
    return F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )


def demo_attention() -> None:
    """Демонстрация работы attention."""
    batch_size, num_heads, seq_len, d_k = 2, 8, 10, 64
    
    # Случайные Q, K, V
    query = torch.randn(batch_size, num_heads, seq_len, d_k)
    key = torch.randn(batch_size, num_heads, seq_len, d_k)
    value = torch.randn(batch_size, num_heads, seq_len, d_k)
    
    print("🎯 Scaled Dot-Product Attention Demo")
    print(f"Input shapes: Q={query.shape}, K={key.shape}, V={value.shape}")
    
    # Обычная attention
    output1, weights1 = scaled_dot_product_attention(query, key, value)
    print(f"Classic:    output={output1.shape}, weights={weights1.shape}")
    
    # Causal attention (decoder-only)
    output2, _ = scaled_dot_product_attention(query, key, value, is_causal=True)
    print(f"Causal:     output={output2.shape}")
    
    # PyTorch 2.0+ (быстрее и оптимизировано)
    output3 = scaled_dot_product_attention_modern(query, key, value, is_causal=True)
    print(f"PyTorch 2+: output={output3.shape}")
    
    # Проверка эквивалентности
    torch.testing.assert_close(output2, output3, rtol=1e-4, atol=1e-4)
    print("✅ PyTorch 2.0+ ≡ Classic implementation")


if __name__ == "__main__":
    demo_attention()
```

---

#### Вопрос: Объясните маскирование в attention.

**Ответ:**

Маскирование (Masking) используется для двух целей:

1. **Padding Mask (Маска паддинга):**  
    В батче предложения разной длины выравниваются паддингами (специальными токенами `<pad>`).  
    Мы не хотим, чтобы механизм внимания учитывал эти пустые токены.  
    _Решение:_ Присвоить score = $-\infty$ для позиций паддинга перед softmax.
    
2. **Look-ahead Mask (Causal Mask, Маска будущего):**  
    Используется в **декодере** (GPT, Transformer Decoder).  
    При генерации текста модель не должна видеть будущие слова. Слово на позиции $t$ может смотреть только на $1...t$.  
    _Решение:_ Верхнетреугольная матрица из $-\infty$.
    
---

#### Вопрос: Какая размерность у матриц self-attention?

**Ответ:**

Пусть:

- `B` = Batch size
    
- `L` = Sequence length (длина последовательности)
    
- `D` = Embedding dimension (размер скрытого слоя)
    
- `H` = Number of heads (количество голов)
    
- `d_k` = `D / H` (размерность одной головы)
    

**Входы:**

- $Q, K, V$: `[B, L, D]` (до разделения на головы)
    

**Внутри Multi-Head:**

- Разделяем на головы: `[B, H, L, d_k]`
    

**Матрица Attention Scores ($QK^T$):**

- Результат умножения `[B, H, L, d_k]` на `[B, H, d_k, L]`
    
- Размерность: **`[B, H, L, L]`**
    
- Это квадратная матрица `seq_len x seq_len` для каждой головы.
    

**Выход:**

- После умножения на $V$: `[B, H, L, d_k]`
    
- После конкатенации голов: `[B, L, D]`
    

---

#### Вопрос: В чем разница между BERT и GPT в рамках подсчета attention?

**Ответ:**

Главная разница — в **направлении** внимания (маске).

**BERT (Bidirectional Encoder Representations from Transformers):**

- Использует **Encoder**.
    
- Внимание: **Bidirectional (двунаправленное)**.
    
- Каждый токен видит **все** токены (слева и справа).
    
- Цель: Понять контекст (для классификации, поиска ответов).
    
- Маска: Только Padding mask.
    

**GPT (Generative Pre-trained Transformer):**

- Использует **Decoder**.
    
- Внимание: **Unidirectional (однонаправленное / Causal)**.
    
- Каждый токен видит только **предыдущие** токены.
    
- Цель: Генерация следующего слова.
    
- Маска: Look-ahead mask (треугольная).
    

---

#### Вопрос: Какая размерность у эмбедингового слоя в трансформере?

**Ответ:**

Размерность матрицы эмбеддингов: **`(Vocab_Size, Embedding_Dim)`**

- **Vocab_Size (размер словаря):** Обычно 30k - 50k токенов (WordPiece / BPE).
    
- **Embedding_Dim ($d_{model}$):** Размер вектора.
    
    - BERT-base: 768
        
    - BERT-large: 1024
        
    - GPT-3: 12288
        

В трансформере эта матрица часто используется дважды (Weight Tying):

1. На входе (Input Embedding).
    
2. На выходе перед Softmax (Output Projection), чтобы получить вероятности слов.
    

---

#### Вопрос: Почему эмбеддинги называются контекстуальными? Как это работает?

**Ответ:**

**Static Embeddings (Word2Vec, GloVe):**  
Слово "bank" всегда имеет один и тот же вектор, независимо от того, "река" это или "финансы".  
E(bank)=[0.1,−0.5,… ]E(bank)=[0.1,−0.5,…]

**Contextual Embeddings (BERT, GPT):**  
Эмбеддинг слова зависит от его окружения. Механизм Self-Attention смешивает информацию о слове с информацией о его соседях.  
На выходе из BERT вектор для слова "bank" будет разным:

1. "Bank of the river": вектор близок к "water", "nature".
    
2. "Bank deposit": вектор близок к "money", "finance".
    

**Как это работает:**  
Вектор слова $x_i$ проходит через слои Self-Attention, где он обновляется как взвешенная сумма всех других векторов $x_j$.  
xinew=∑αijxjxinew=∑αijxj  
Так слово впитывает контекст.

---

#### Вопрос: Что используется в трансформере layer norm или batch norm и почему?

**Ответ:**

В Трансформерах используется **Layer Normalization**.

**Почему Layer Norm?**

1. **Независимость от батча:** В NLP размер батча может быть маленьким (из-за ограничений памяти с длинными текстами), а Batch Norm плохо работает на малых батчах.
    
2. **Переменная длина:** Тексты имеют разную длину. Batch Norm требует фиксированной структуры признаков по батчу, что сложно для последовательностей. Layer Norm нормализует каждый сэмпл отдельно (вдоль размерности features), что идеально для NLP.
    

**Batch Norm:** Считает среднее/дисперсию по вертикали (по батчу).  
**Layer Norm:** Считает среднее/дисперсию по горизонтали (по фичам одного токена).

---

#### Вопрос: Зачем в трансформерах PreNorm и PostNorm?

**Ответ:**

Это два варианта расположения Layer Normalization относительно блока Residual Connection.

**Post-Norm (Оригинальный Transformer):**  
xt+1=Norm(xt+Sublayer(xt))xt+1=Norm(xt+Sublayer(xt))

- Нормализация стоит **после** сложения с residual connection.
    
- **Проблема:** Градиенты могут быть нестабильными в начале обучения, требуется "warm-up" (постепенное повышение learning rate).
    

**Pre-Norm (GPT-2, современные модели):**  
xt+1=xt+Sublayer(Norm(xt))xt+1=xt+Sublayer(Norm(xt))

- Нормализация стоит **перед** подслоем, но residual connection идет "чистым".
    
- **Плюс:** Градиенты текут по "чистому" пути residual connection без искажений. Обучение гораздо стабильнее, можно учить без warm-up.
    
- **Минус:** Может слегка терять в финальном качестве по сравнению с Post-Norm.
    

---

#### Вопрос: Объясните разницу между soft и hard (local/global) attention?

**Ответ:**

**Soft Attention (Global):**

- Модель считает веса для **всех** патчей картинки или слов текста.
    
- Функция дифференцируема (можно учить через backprop).
    
- Используется в стандартном Трансформере.
    

**Hard Attention:**

- Модель **выбирает** одно конкретное место (patch/token), на которое смотреть.
    
- Это стохастический процесс (выборка), поэтому функция не дифференцируема.
    
- Требует обучения через Reinforcement Learning (REINFORCE).
    

**Local Attention (Windowed):**

- Компромисс. Смотрим не на всё (Global), но и не на одну точку (Hard).
    
- Смотрим на фиксированное окно вокруг текущего токена.
    
- Экономит вычисления ($O(n \cdot w)$ вместо $O(n^2)$).
    

---

#### Вопрос: Объясните Multi-Head Attention.

**Ответ:**

Идея: дать модели возможность смотреть на текст с разных "точек зрения" одновременно.

Вместо одного большого внимания размером $D=512$, мы делаем 8 "голов" по $d_k=64$.

- Голова 1 может следить за синтаксисом (существительное -> прилагательное).
    
- Голова 2 может следить за длинными зависимостями (местоимение -> имя).
    
- Голова 3 может следить за соседними словами.
    

**Процесс:**

1. Входной вектор проецируется 8 разными матрицами в 8 подпространств.
    
2. Attention считается параллельно в каждом подпространстве.
    
3. Результаты 8 голов конкатенируются.
    
4. Умножаются на финальную матрицу $W_O$, чтобы смешать информацию.
    

Это не увеличивает количество параметров (так как размерность делится на $H$), но улучшает выразительность.

---

#### Вопрос: Какие другие виды механизмов внимания вы знаете? На что направлены эти модификации?

**Ответ:**

Модификации в основном направлены на борьбу с квадратичной сложностью $O(N^2)$.

1. **Sparse Attention (Разреженное внимание):**  
    Считаем attention не для всех пар, а только для некоторых (например, каждый 5-й токен, или случайные). Пример: _BigBird, Longformer_.
    
2. **Linear Attention (Линейное внимание):**  
    Аппроксимирует softmax через kernel methods, чтобы изменить порядок умножения матриц и получить сложность $O(N)$. Пример: _Linformer, Performer_.
    
3. **Sliding Window / Local Attention:**  
    Смотрим только на соседей. Пример: _Longformer, Mistral_.
    
4. **Flash Attention:**  
    Не меняет математику, но оптимизирует операции с памятью GPU (IO-aware), делая расчет точного attention в разы быстрее.
    
5. **Grouped-Query Attention (GQA):**  
    Используется в LLaMA-2/3. Группирует головы ключей и значений, чтобы уменьшить размер KV-кэша при инференсе.
    

---

#### Вопрос: На сколько усложнится self-attention при увеличении числа голов?

**Ответ:**

**По количеству параметров:**  
**Не усложнится.**  
В стандартной реализации, если мы увеличиваем число голов $H$, мы уменьшаем размерность каждой головы $d_k$, так что $H \times d_k = d_{model}$.  
Общее количество весов в проекциях ($W^Q, W^K, W^V$) остается неизменным: $3 \times d_{model}^2$.

**По вычислениям:**  
Практически не меняется, так как общее количество операций умножения остается тем же.  
Однако, слишком большое число голов может быть неэффективным на GPU из-за накладных расходов на управление множеством мелких матриц (падение утилизации ядер).

## BERT, GPT, T5 & RoBERTa: Сравнение и особенности

---

#### Вопрос: Почему BERT во многом проигрывает RoBERTa и что вы можете взять у RoBERTa?

**Ответ:**

**RoBERTa (Robustly optimized BERT approach)** — это, по сути, "BERT, который обучили правильно". Архитектурно они почти идентичны, но RoBERTa исправляет недочеты в процедуре обучения BERT.

**Что улучшили в RoBERTa (и что стоит брать):**

1. **Dynamic Masking (Динамическое маскирование):**  
    В BERT маски генерировались один раз перед обучением (static masking). Модель видела одни и те же маски много раз.  
    В RoBERTa маска генерируется на лету для каждого батча. Модель видит новые варианты маскирования каждый раз, что улучшает обобщение.
    
2. **Больше данных и дольше обучение:**  
    BERT был "недообучен". RoBERTa училась на 160GB текста (вместо 16GB у BERT) и гораздо дольше.
    
3. **Убрали задачу NSP (Next Sentence Prediction):**  
    В BERT была задача предсказывать, идет ли предложение B за предложением A. Исследования показали, что она не помогает, а иногда вредит. RoBERTa учится только на MLM (Masked Language Modeling) с длинными непрерывными кусками текста.
    
4. **Большие батчи:**  
    Размер батча увеличен с 256 до 8000. Это стабилизирует градиенты и улучшает сходимость.
    

**Вывод:** Если нужно выбрать между BERT и RoBERTa для задачи понимания текста — **всегда берите RoBERTa**(или DeBERTa), так как это более сильная версия той же архитектуры.

---

#### Вопрос: Что такое T5 и BART модели? Чем они отличаются?

**Ответ:**

Обе модели — это **Encoder-Decoder трансформеры**, которые объединяют лучшее от BERT (encoder) и GPT (decoder).

**BART (Bidirectional and Auto-Regressive Transformers):**

- **Архитектура:** Стандартный Encoder-Decoder (как в машинном переводе).
    
- **Обучение:** Denoising Autoencoder. Мы ломаем текст (шум, удаление слов, перестановка предложений) и заставляем модель восстановить оригинал.
    
- **Применение:** Отлично подходит для генерации (суммаризация, перевод) и понимания. Похож на BERT, к которому прикрутили GPT-декодер.
    

**T5 (Text-to-Text Transfer Transformer):**

- **Философия:** "Все задачи NLP — это текст-в-текст".
    
    - Перевод: "translate English to German: ..." -> "..."
        
    - Классификация: "sentiment analysis: ..." -> "positive"
        
    - Регрессия: "sts-b: ..." -> "3.8"
        
- **Обучение:** Span corruption (заменяет куски текста на `<extra_id_0>`). Модель генерирует только пропущенные куски, а не весь текст (в отличие от BART).
    
- **Особенность:** Использует Relative Positional Embeddings.
    

**Отличие:**

- **BART** восстанавливает _весь_ текст из зашумленного. Хорош для генерации длинных текстов.
    
- **T5** генерирует только _маскированные спаны_. Универсален благодаря формату "text-to-text" и показывает отличные результаты на множестве задач.
    

---

#### Вопрос: Что такое task-agnostic модели? Приведите примеры.

**Ответ:**

**Task-agnostic модели** — это модели, архитектура которых **не меняется** в зависимости от решаемой задачи.  
Вместо того чтобы добавлять специальные слои (головы) для классификации, NER или QA, мы используем одну и ту же модель для всего.

**Примеры:**

1. **GPT-3 (Few-shot learner):**  
    Мы не дообучаем веса. Мы просто даем промпт:  
    _"Translate to French: cheese ->"_  
    Модель понимает задачу из контекста и генерирует ответ. Архитектура не меняется.
    
2. **T5 (Text-to-Text):**  
    Даже если мы файн-тюним T5, мы используем один и тот же выходной слой (генерация текста) для всего.
    
    - Классификация? Генерируем слово "positive".
        
    - Перевод? Генерируем "cheese".
        
    - QA? Генерируем ответ.
        

**Task-specific (для сравнения):**  
BERT + Classification Head (слой с 2 нейронами) отличается от BERT + NER Head (слой с N тегами).

---

#### Вопрос: Объясните transformer модели сравнивая BERT, GPT и T5.

**Ответ:**

Сравнение по трем основным типам архитектуры Трансформера:

|Характеристика|BERT (Encoder-only)|GPT (Decoder-only)|T5 (Encoder-Decoder)|
|---|---|---|---|
|**Архитектура**|Только Энкодер|Только Декодер|Энкодер + Декодер|
|**Внимание**|Bidirectional (видит всё)|Unidirectional (видит прошлое)|Bi (Enc) + Uni (Dec)|
|**Задача обучения**|MLM (восстановить скрытое слово)|CLM (предсказать следующее слово)|Span Corruption (восстановить кусок)|
|**Сильная сторона**|Понимание (NLU), классификация|Генерация (NLG), креатив|Перевод, суммаризация, универсальность|
|**Пример использования**|Поиск, классификация отзывов|Чат-боты, написание статей|Перевод, ответы на вопросы|

- **BERT** — это "читатель". Он идеально понимает контекст, но плохо пишет.
    
- **GPT** — это "писатель". Он отлично продолжает текст, но может терять глубокий контекст справа.
    
- **T5** — это "переводчик". Он читает (как BERT) и пишет (как GPT).
    

---

#### Вопрос: Какая большая проблема есть в моделях BERT, GPT и тд относительно знаний модели? Как это можно решать?

**Ответ:**

**Проблема:** Знания модели **заморожены** в весах на момент окончания обучения (Cutoff date).

- GPT-4 (обученная до 2023) не знает, кто выиграл выборы в 2024.
    
- Модели могут галлюцинировать (выдумывать факты), так как они запоминают вероятности слов, а не базу данных фактов.
    

**Как решать:**

1. **RAG (Retrieval-Augmented Generation):**  
    Не заставлять модель помнить всё. Дать ей доступ к поисковику/базе знаний.
    
    - _Запрос:_ "Кто президент?" -> _Поиск в Google_ -> _Контекст:_ "Выборы 2024..." -> _Модель:_ "Президент... на основе контекста".
        
2. **Продолжительное обучение (Continual Learning):**  
    Дообучать модель на новых данных. Сложно и дорого, риск "катастрофического забывания" старых знаний.
    
3. **Knowledge Editing (ROME, MEMIT):**  
    Методы прямого редактирования весов нейросети, чтобы обновить конкретный факт ("Эйфелева башня в Риме" -> "в Париже"). Пока экспериментально.
    

---

#### Вопрос: Как работает decoder like а-ля GPT на обучении и инференсе. В чем разница?

**Ответ:**

**На обучении (Training):**  
Используется **Teacher Forcing** и параллелизация.  
Мы знаем правильный текст целиком: "The cat sat on the mat".  
Модель получает на вход "The cat sat on the", а должна предсказать "cat sat on the mat".  
Благодаря Masked Attention, мы можем подать **весь текст сразу** и посчитать лосс для всех токенов параллельно.

- Позиция 1 видит только "The" -> предсказывает "cat".
    
- Позиция 2 видит "The cat" -> предсказывает "sat".  
    Все это происходит за **один проход**.
    

**На инференсе (Inference):**  
Работает **авторегрессионно** (пошагово). Мы не знаем будущего.

1. Вход: "The" -> Модель: "cat".
    
2. Вход: "The cat" -> Модель: "sat".
    
3. Вход: "The cat sat" -> Модель: "on".  
    Мы должны запускать модель **N раз** для генерации N слов.
    

**Разница:**

- Обучение: Параллельное, быстрое.
    
- Инференс: Последовательный, медленный (нужен KV-Cache для ускорения).
    

---

#### Вопрос: Объясните разницу между головами и слоями в трансформер моделях.

**Ответ:**

Представьте модель как офис.

**Слои (Layers):** Это **этажи** здания.

- Информация проходит последовательно с 1-го этажа на последний.
    
- Чем выше этаж (слой), тем более сложные и абстрактные вещи понимает модель.
    
    - Нижние слои: синтаксис, части речи.
        
    - Верхние слои: семантика, смысл, контекст.
        
- Увеличение слоев делает модель "глубже" и умнее.
    

**Головы (Heads):** Это **сотрудники** на одном этаже.

- На каждом слое (этаже) работает группа голов (Multi-Head Attention).
    
- Они смотрят на одну и ту же информацию, но с **разных сторон** (параллельно).
    
    - Голова 1: "Кто что сделал?" (субъект-глагол).
        
    - Голова 2: "Куда?" (предлоги).
        
    - Голова 3: "Связь с прошлым предложением".
        
- Увеличение голов не делает модель глубже, но делает её восприятие более "широким" и детализированным.
    

**Итог:**

- **Слои:** Последовательная переработка информации (Deep).
    
- **Головы:** Параллельный анализ разных аспектов (Wide).

## Positional Encoding: Позиционная информация в трансформерах

---

#### Вопрос: Почему в эмбеддингах transformer моделей с attention теряется информациях о позициях?

**Ответ:**

В RNN информация о позиции **автоматически** сохраняется, так как модель обрабатывает токены последовательно (слово 1, потом слово 2, потом слово 3).

В Трансформере все токены обрабатываются **параллельно**. Механизм Self-Attention считает взаимодействия между всеми парами токенов, но **не знает**, в каком порядке они идут.

**Пример проблемы:**

```text
Предложение 1: "cat sat on mat" Предложение 2: "sat mat on cat" После эмбеддинга: embed(cat) = [0.1, -0.5, ...] embed(sat) = [0.2, 0.3, ...] ... Self-Attention не видит разницы! Набор токенов одинаковый,  только порядок изменился, но это не отражено в эмбеддингах.
```

Без позиционной информации модель не может отличить эти два предложения, так как Attention просто смешивает все эмбеддинги.

---

#### Вопрос: Объясните подходы для позициональных эмбеддингов и их плюсы и минусы.

**Ответ:**

**1. Sinusoidal Positional Embeddings (Исходный Transformer)**

Формула:  
PE(pos,2i)=sin⁡(pos100002i/d)PE(pos,2i)=sin(100002i/dpos)  
PE(pos,2i+1)=cos⁡(pos100002i/d)PE(pos,2i+1)=cos(100002i/dpos)

где `pos` — позиция, `i` — индекс размерности.

```python
import torch
import math
from typing import Optional, Union
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding (Vaswani et al., 2017).
    Добавляет информацию о позициях через синусоидальные функции.
    
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 5000,
        base: float = 10_000.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Предвычисляем для скорости (register_buffer не сохраняется при checkpoint)
        self.register_buffer('pe', self._compute_pe(device))
    
    def _compute_pe(self, device: Optional[torch.device]) -> torch.Tensor:
        """Генерирует positional encodings."""
        pe = torch.zeros(self.max_seq_len, self.d_model, device=device, dtype=torch.float32)
        
        # Позиции: [max_seq_len, 1]
        position = torch.arange(0, self.max_seq_len, device=device).unsqueeze(1)
        
        # Частоты: exp(-log(10000) * 2i/d_model) = 10000^(-2i/d_model)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=torch.float32) 
            * -(math.log(self.base) / self.d_model)
        )
        
        # Синусоиды для чётных/нечётных индексов
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe  # [max_seq_len, d_model]
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model] — эмбеддинги
            seq_len: если None, берём из x.size(1)
        """
        if seq_len is None:
            seq_len = x.size(1)
        
        # Берём только нужную длину + batch dimension
        return self.pe[:seq_len, :].unsqueeze(0)  # [1, seq_len, d_model]


def standalone_sinusoidal_pe(
    seq_len: int,
    d_model: int,
    base: float = 10_000.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Функциональная версия без класса."""
    pe = torch.zeros(seq_len, d_model, device=device, dtype=torch.float32)
    position = torch.arange(0, seq_len, device=device).unsqueeze(1).float()
    
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device).float() 
        * -(math.log(base) / d_model)
    )
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe


def demo_positional_encoding() -> None:
    """Демонстрация работы."""
    d_model = 512
    seq_len = 100
    
    # Классовый способ (рекомендуется)
    pe_module = SinusoidalPositionalEncoding(d_model, max_seq_len=1000)
    pe1 = pe_module(torch.zeros(2, 50, d_model))  # [1, 50, 512]
    
    # Функциональный способ
    pe2 = standalone_sinusoidal_pe(seq_len, d_model)
    
    print(f"✅ Sinusoidal PE shape: {pe1.shape}")
    print(f"✅ Эквивалентны: {torch.allclose(pe1.squeeze(0), pe2[:50], atol=1e-6)}")
    
    # Визуализация первых позиций
    print("\nПервые 4 измерения для позиций 0-3:")
    print(pe1[0, :4, :4])


if __name__ == "__main__":
    demo_positional_encoding()
```

**Плюсы:**

- Детерминированные (не нужно обучение).
    
- Гладкие и периодические (модель может интерполировать на длины > обучающие).
    

**Минусы:**

- Фиксированы. Не учат о важных позициях.
    
- На очень длинных последовательностях (10k+) может быть недостаточно.
    

---

**2. Learned Positional Embeddings**

Просто добавляем обучаемый параметр размером `[max_seq_len, d_model]`.

```python
import torch
import torch.nn as nn
from typing import Optional


class LearnedPositionalEncoding(nn.Module):
    """
    Learned Positional Encoding — обучаемые векторы позиций.
    Используется в BERT, GPT-2 вместо синусоидальных.
    
    Преимущества:
    - Модель сама учится оптимальные представления позиций
    - Лучше адаптируется к конкретной задаче
    
    Недостатки:
    - Требует обучения (параметры)
    - Ограничение max_seq_len (нельзя экстраполировать)
    """
    
    def __init__(
        self,
        max_seq_len: int,
        d_model: int,
        dropout: float = 0.1,
        init_scale: float = 0.02,
    ):
        super().__init__()
        self.d_model = d_model
        
        # Обучаемый embedding для позиций [0, max_seq_len-1]
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Инициализация маленькими весами (как в GPT)
        nn.init.normal_(self.pos_embedding.weight, std=init_scale)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model] — токен эмбеддинги
            position_ids: [batch_size, seq_len] — индексы позиций (опционально)
        
        Returns:
            x + learned_pos: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = x.shape[:2]
        
        if position_ids is None:
            # Генерируем позиции 0,1,2,...,seq_len-1 для всех батчей
            position_ids = torch.arange(seq_len, device=x.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Получаем обучаемые positional embeddings
        pos_emb = self.pos_embedding(position_ids)  # [batch_size, seq_len, d_model]
        
        return self.dropout(x + pos_emb)


def compare_pos_encodings() -> None:
    """Сравнение синусоидального vs learned PE."""
    batch_size, seq_len, d_model = 2, 10, 64
    
    # Токен эмбеддинги (случайные)
    token_embeds = torch.randn(batch_size, seq_len, d_model)
    
    # Learned PE
    learned_pe = LearnedPositionalEncoding(max_seq_len=100, d_model=d_model)
    output_learned = learned_pe(token_embeds)
    
    # Sinusoidal PE (из предыдущего примера)
    sin_pe = SinusoidalPositionalEncoding(d_model)
    output_sin = token_embeds + sin_pe(token_embeds)
    
    print("🎯 Сравнение Positional Encodings")
    print(f"Token embeds:     {token_embeds.shape}")
    print(f"Learned PE:       {learned_pe.pos_embedding.weight.shape}")
    print(f"Sinusoidal PE:    {sin_pe.pe.shape}")
    print(f"Output learned:   {output_learned.shape}")
    print(f"Output sin:       {output_sin.shape}")
    print(f"Learned params:   {sum(p.numel() for p in learned_pe.parameters()):,}")
    print(f"Sinusoidal params: 0 (фиксированные)")


def demo_usage() -> None:
    """Пример использования в трансформере."""
    max_seq_len, d_model = 512, 768
    
    # Модель с learned PE (как BERT/GPT-2)
    learned_pe = LearnedPositionalEncoding(max_seq_len, d_model)
    
    # Input IDs → Token Embedding + Learned PE
    input_ids = torch.randint(0, 10_000, (2, 50))  # [batch=2, seq_len=50]
    token_embeds = torch.randn(2, 50, d_model)  # симуляция embedding layer
    
    final_embeds = learned_pe(token_embeds)
    print(f"✅ Final embeds: {final_embeds.shape}")


if __name__ == "__main__":
    compare_pos_encodings()
    demo_usage()
```

**Плюсы:**

- Модель может учиться относительной важности позиций.
    
- Нет гиперпараметров (как в sine).
    

**Минусы:**

- Не может обобщать на длины > `max_seq_len` (необходимо интерполяция или fine-tuning).
    
- Больше параметров (может быть проблемой на гигантских длинах).
    

---

**3. Relative Positional Embeddings**

Вместо абсолютных позиций, мы кодируем **расстояние между токенами**.

**Плюсы:**

- Генерализируется на любые длины.
    
- Более интуитивно (важно расстояние между словами, не их абсолютная позиция).
    

**Минусы:**

- Более сложная реализация.
    
- Требует модификации attention механизма.
    

---

**4. Rotary Positional Embeddings (RoPE)**

Комбинирует лучшее от sinusoidal и learned. Эмбеддинги "поворачиваются" (ротируются) на угол, зависящий от позиции.

**Плюсы:**

- Генерализируется на длины > обучающих.
    
- Явно кодирует relative информацию.
    
- SOTA во многих современных моделях (LLaMA, Mistral, Qwen).
    

**Минусы:**

- Более сложная математика и реализация.
    

---

#### Вопрос: Почему нельзя просто добавить эмбеддинг с индексом токена?

**Ответ:**

Если мы просто добавим позиционный эмбеддинг как отдельный вектор:

```python
embedding = token_embedding + position_embedding[index]
```

То при индексе 100 позиционный эмбеддинг может быть очень большим по норме (например, вектор из больших чисел), и это **утопит** семантическую информацию токена.

**Пример:**

```text
Token embedding (норма = 1): [0.1, -0.2, 0.15] Position embedding 100 (норма = 100): [50.0, -80.0, 40.0] Сумма: [50.1, -80.2, 40.15] Позиция доминирует! Две разные точки в пространстве эмбеддингов будут очень близко просто потому, что они на соседних позициях.
```

**Решение (Sinusoidal PE):**  
Sinusoidal PE специально спроектирован так, чтобы его норма была примерно постоянной (не растет с индексом). Значения чередуются между -1 и 1, уравновешивая разные размерности.

---

#### Вопрос: Почему мы не учим positional embeddings?

**Ответ:**

На самом деле **мы учим!** В большинстве современных моделей используются learned positional embeddings.

**Что происходит:**

1. Инициализируем параметр `PE` случайным образом: `[max_seq_len, d_model]`.
    
2. Во время обучения градиенты обновляют эти значения.
    
3. Модель учится, какие позиции важны для решения задач.
    

```python
import torch
import torch.nn as nn
from typing import Optional


class TransformerWithLearnedPE(nn.Module):
    """
    Полный трансформер с обучаемым positional encoding.
    Комбинирует token embeddings + learned PE + transformer blocks.
    """
    
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        init_scale: float = 0.02,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # 1. Token Embedding (обучаемый)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.token_embedding.weight, std=init_scale)
        
        # 2. Learned Positional Encoding (обучаемый Parameter!)
        self.pos_embedding = nn.Parameter(torch.randn(max_seq_len, d_model) * init_scale)
        
        # 3. Transformer блоки
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_blocks = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # 4. Выход на словарь (для языкового моделирования)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        nn.init.normal_(self.head.weight, std=init_scale)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        token_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            token_ids: [batch_size, seq_len] — индексы токенов
            targets: для расчёта loss (shifted token_ids)
        
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = token_ids.shape
        
        # Token embeddings
        x = self.token_embedding(token_ids)  # [batch, seq_len, d_model]
        
        # Learned Positional Encoding (+ backprop!)
        x = x + self.pos_embedding[:seq_len]  # [batch, seq_len, d_model]
        
        x = self.dropout(x)
        
        # Transformer blocks (causal attention внутри)
        x = self.transformer_blocks(x)  # [batch, seq_len, d_model]
        
        # Final LN + Head
        x = self.ln_f(x)
        logits = self.head(x)  # [batch, seq_len, vocab_size]
        
        # Loss (если нужны targets)
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,  # для паддингов
            )
            return logits, loss
        
        return logits


def demo_transformer() -> None:
    """Демонстрация работы."""
    model = TransformerWithLearnedPE(
        vocab_size=10_000,
        max_seq_len=1024,
        d_model=256,
        num_heads=8,
        num_layers=4,
    )
    
    # Подсчёт параметров
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🚀 Transformer модель создана!")
    print(f"  Параметры: {total_params:,}")
    print(f"  Token PE:  {model.token_embedding.weight.numel():,}")
    print(f"  Pos PE:    {model.pos_embedding.numel():,}")
    
    # Тест forward pass
    batch_size, seq_len = 2, 20
    token_ids = torch.randint(0, 10_000, (batch_size, seq_len))
    
    logits = model(token_ids)
    print(f"\n✅ Forward: input [{token_ids.shape}] → logits [{logits.shape}]")
    
    # С loss
    targets = torch.roll(token_ids, -1, dims=1)  # shifted для next-token prediction
    targets[:, -1] = -100  # ignore last token
    logits, loss = model(token_ids, targets)
    print(f"✅ Loss: {loss.item():.4f}")


if __name__ == "__main__":
    demo_transformer()
```

**Различия между learned и predefined:**

- **Learned:** Может переобучиться, если max_seq_len маленький. Но лучше адаптируется.
    
- **Predefined (Sinusoidal):** Более стабилен, генерализируется на длины > max_seq_len.
    

Исследования показывают, что часто лучше использовать гибрид: инициализировать learned PE sinusoidal значениями.

---

#### Вопрос: Что такое relative и absolute positional encoding?

**Ответ:**

**Absolute Positional Encoding (Абсолютное):**  
Каждому токену присваивается вектор на основе его **абсолютной позиции** в последовательности.

```text
Position 0: PE = [sin(0/10000^0), cos(0/10000^0), ...] Position 1: PE = [sin(1/10000^1), cos(1/10000^1), ...] Position 2: PE = [sin(2/10000^2), cos(2/10000^2), ...]
```

Это то, что используется в исходном Transformer (Vaswani et al., 2017).

**Relative Positional Encoding (Относительное):**  
Информация о позициях кодируется как **расстояние между токенами** в механизме Attention.

Вместо добавления одного вектора к каждому эмбеддингу, мы добавляем позиционные смещения прямо в Attention scores.

```python
def relative_attention(query, key, value, relative_pos_embeddings):
         """Вместо: scores = Q @ K^T    Мы делаем: scores = Q @ K^T + relative_positions_bias где relative_positions_bias зависит от расстояния между позициями i и j"""    
         scores = torch.matmul(query, key.transpose(-2, -1))
         # Добавляем относительную позицию    
         distances = torch.arange(query.size(1)).unsqueeze(1) - torch.arange(key.size(1)).unsqueeze(0)    relative_bias = relative_pos_embeddings[distances]         
         scores = scores + relative_bias    
         return torch.softmax(scores, dim=-1) @ value
```

**Плюсы Relative Encoding:**

- Работает на последовательностях любой длины (инекспоненциально генерализируется).
    
- Более интуитивно (расстояние между словами более важно, чем абсолютная позиция).
    

**Минусы:**

- Более сложная реализация.
    
- Требует больше памяти (матрица относительных позиций размером `[seq_len, seq_len]`).
    

---

#### Вопрос: Подробно объясните принцип работы Rotary Positional Embeddings (RoPE)

**Ответ:**

**RoPE (Rotary Position Embedding)** — это метод, где позиционная информация кодируется как **вращение (rotation)** векторов в комплексной плоскости.

**Основная идея:**  
Эмбеддинг каждого токена "поворачивается" на угол, зависящий от его позиции, внутри механизма Attention.

**Математика:**

Для 2D подпространства (для простоты):  
(xi′xj′)=(cos⁡(mθ)−sin⁡(mθ)sin⁡(mθ)cos⁡(mθ))(xixj)(xi′xj′)=(cos(mθ)sin(mθ)−sin(mθ)cos(mθ))(xixj)

где:

- $m$ — позиция токена.
    
- $\theta$ — базовый угол (обычно $\theta_k = 10000^{-2k/d}$, как в sinusoidal).
    

**Почему это генерирует relative информацию:**

Когда мы считаем скалярное произведение двух повернутых векторов в разных позициях $m$ и $n$:  
⟨Rm(q),Rn(k)⟩=∣q∣∣k∣cos⁡(mθ−nθ)=∣q∣∣k∣cos⁡((m−n)θ)⟨Rm(q),Rn(k)⟩=∣q∣∣k∣cos(mθ−nθ)=∣q∣∣k∣cos((m−n)θ)

**Результат зависит только от расстояния $(m-n)$**, а не от абсолютных позиций $m$ и $n$!

**Реализация на Python:**

```python
import torch
import math
from typing import Tuple


def rotary_embeddings(
    seq_len: int,
    dim: int,
    base: float = 10_000.0,
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Генерирует синусоиды для RoPE (Rotary Position Embedding).
    
    Args:
        seq_len: длина последовательности
        dim: размерность head_dim (d_k)
        base: базовая частота (10000)
    
    Returns:
        cos, sin: [seq_len, dim]
    """
    # Угловые частоты theta_i = base^(-2i/dim)
    theta = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
    
    # Позиции m ∈ [0, seq_len)
    m = torch.arange(seq_len, device=device).float()
    
    # Углы m * theta_i для каждой позиции
    angles = torch.outer(m, theta)  # [seq_len, dim/2]
    
    return torch.cos(angles), torch.sin(angles)  # [seq_len, dim/2]


def apply_rotary_pos_emb(
    tensor: torch.Tensor,
    cos_emb: torch.Tensor,
    sin_emb: torch.Tensor,
) -> torch.Tensor:
    """
    Применяет RoPE вращение к тензору Q или K.
    
    Args:
        tensor: [..., seq_len, dim] — query или key
        cos_emb, sin_emb: [seq_len, dim]
    
    Rotary Embedding превращает:
    x_m^(i) → x_m^(i) * cos(m*θ_i) - x_m^(i+1) * sin(m*θ_i)
    x_m^(i+1) → x_m^(i) * sin(m*θ_i) + x_m^(i+1) * cos(m*θ_i)
    """
    # Разделяем на пары (x_re, x_im)
    tensor = tensor[..., :tensor.size(-1) // 2 * 2]  # обрезаем нечётную размерность
    x1 = tensor[..., 0::2]  # real part
    x2 = tensor[..., 1::2]  # imag part
    
    # Комплексное умножение на e^(i*m*θ)
    rotated = torch.cat([
        x1 * cos_emb - x2 * sin_emb,  # Re(cosθ + i*sinθ)
        x1 * sin_emb + x2 * cos_emb,  # Im(cosθ + i*sinθ)
    ], dim=-1)
    
    # Добавляем обратно нечётную размерность (если была)
    if tensor.size(-1) % 2 == 1:
        rotated = torch.cat([rotated, tensor[..., -1:]], dim=-1)
    
    return rotated


def rope_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    seq_len: int,
    head_dim: int,
) -> torch.Tensor:
    """
    Полный attention с RoPE (Llama/GPT-NeoX стиль).
    """
    # Генерируем RoPE синусоиды для данной головы
    cos_emb, sin_emb = rotary_embeddings(seq_len, head_dim, device=query.device)
    
    # Применяем RoPE только к Q и K!
    q_rope = apply_rotary_pos_emb(query, cos_emb, sin_emb)
    k_rope = apply_rotary_pos_emb(key, cos_emb, sin_emb)
    
    # Стандартный scaled dot-product attention
    d_k = head_dim
    scores = torch.matmul(q_rope, k_rope.transpose(-2, -1)) / math.sqrt(d_k)
    attention_weights = torch.softmax(scores, dim=-1)
    
    return torch.matmul(attention_weights, value)


def demo_rope() -> None:
    """Демонстрация RoPE."""
    batch_size, num_heads, seq_len, head_dim = 2, 8, 100, 64
    
    query = torch.randn(batch_size, num_heads, seq_len, head_dim)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    print("🔄 Rotary Position Embedding (RoPE) Demo")
    print(f"Input shapes: [{query.shape}]")
    
    # Без RoPE
    scores_base = torch.matmul(query, key.transpose(-2, -1))
    
    # С RoPE
    cos_emb, sin_emb = rotary_embeddings(seq_len, head_dim)
    q_rope = apply_rotary_pos_emb(query, cos_emb, sin_emb)
    k_rope = apply_rotary_pos_emb(key, cos_emb, sin_emb)
    scores_rope = torch.matmul(q_rope, k_rope.transpose(-2, -1))
    
    print(f"RoPE cos/sin:  [{cos_emb.shape}]")
    print(f"Q/K после RoPE: [{q_rope.shape}]")
    print(f"Attention сохраняет относительные расстояния! ✓")
    
    # Проверка: RoPE сохраняет dot-product свойства
    print(f"Max diff в scores: {torch.max(torch.abs(scores_base - scores_rope)):.4f}")


if __name__ == "__main__":
    demo_rope()
```

**Преимущества RoPE:**

1. **Генерализируется на любые длины:** Если модель обучена на 2048 токенов, она может работать на 10000. Просто вычислим углы для позиции 10000.
    
2. **Явно относительная:** Attention автоматически зависит от расстояний, а не от абсолютных позиций.
    
3. **Эффективно:** Не добавляет дополнительные параметры обучения.
    
4. **Масштабируется:** Используется в LLaMA, Mistral, Qwen и других SOTA моделях.
    

**Почему это работает лучше:**  
Модель может видеть, что "расстояние 5" всегда означает одно и то же (5 слов между токенами), независимо от абсолютной позиции в текст.


## Pretraining: Обучение и использование предобученных моделей

---

#### Вопрос: Как обучается Causal Language Modelling?

**Ответ:**

**Causal Language Modelling (CLM)** — это задача предсказания следующего токена на основе предыдущих. Используется в GPT, LLaMA и других генеративных моделях.

**Процесс:**

На обучении модель видит полный текст и учится предсказывать каждый токен по его предыстории.

```text
Текст: "The cat sat on the mat" Токены: [THE, CAT, SAT, ON, THE, MAT] Шаг 1: input=[THE] → target=CAT Шаг 2: input=[THE, CAT] → target=SAT Шаг 3: input=[THE, CAT, SAT] → target=ON Шаг 4: input=[THE, CAT, SAT, ON] → target=THE Шаг 5: input=[THE, CAT, SAT, ON, THE] → target=MAT
```

**На тренинге благодаря Causal Mask все это происходит параллельно:**

```text
Вход к модели:         [THE, CAT, SAT, ON, THE, MAT] Маска внимания (треугольная): [[1, 0, 0, 0, 0, 0],  [1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]] Модель выдает 6 токенов на выходе параллельно: Pos 1: predict CAT (видит только THE) Pos 2: predict SAT (видит THE, CAT) Pos 3: predict ON (видит THE, CAT, SAT) ...
```

**Функция потерь:**

L=−∑t=1Tlog⁡P(yt∣y1:t−1)L=−∑t=1TlogP(yt∣y1:t−1)

где каждый токен $y_t$ предсказывается по его контексту $y_{1:t-1}$.

**Реализация на Python:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CausalLanguageModel(nn.Module):
    """
    Decoder-only трансформер для языкового моделирования (GPT-style).
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        init_scale: float = 0.02,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.token_embedding.weight, std=init_scale)
        
        # Learned positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * init_scale)
        
        # Transformer decoder blocks (causal attention)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_blocks = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )
        
        # LM head (shared weights с embedding для data efficiency)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # weight tying!
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] (padding mask)
            labels: [batch_size, seq_len] для loss (shifted input_ids)
        
        Returns:
            logits: [batch, seq_len, vocab_size]
            loss: scalar (если labels переданы)
        """
        batch_size, seq_len = input_ids.shape
        
        # Token + Positional embeddings
        x = self.token_embedding(input_ids)  # [batch, seq_len, d_model]
        x = x + self.pos_embedding[:, :seq_len, :]  # learned PE!
        x = self.dropout(x)
        
        # Causal mask для autoregressive attention
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device),
            diagonal=1,
        ).bool()
        
        # Transformer (применяет causal mask автоматически)
        x = self.transformer_blocks(
            x,
            memory=None,  # decoder-only
            tgt_mask=causal_mask,
            tgt_key_padding_mask=~attention_mask if attention_mask is not None else None,
        )
        
        # Final LN + LM Head
        x = self.ln_f(x)
        logits = self.lm_head(x)  # [batch, seq_len, vocab_size]
        
        # Loss computation (shifted prediction)
        loss = None
        if labels is not None:
            # shift_logits: предсказываем token[i+1] из token[0:i]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,  # padding tokens
            )
        
        return logits, loss


def train_step(
    model: CausalLanguageModel,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Один шаг обучения."""
    model.train()
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)  # shifted input_ids
    
    logits, loss = model(input_ids, labels=labels)
    
    # Backward
    model.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    torch.optim.Adam(model.parameters(), lr=1e-4).step()  # упрощённо
    
    return loss


def demo_causal_lm() -> None:
    """Демонстрация модели."""
    model = CausalLanguageModel(
        vocab_size=50_000,
        d_model=768,
        num_layers=12,
        num_heads=12,
        max_seq_len=2048,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🌟 Causal LM создана!")
    print(f"  Параметры: {total_params:,}")
    print(f"  Token Emb: {model.token_embedding.weight.numel():,}")
    print(f"  Pos Emb:   {model.pos_embedding.numel():,}")
    
    # Тест
    batch = {
        "input_ids": torch.randint(0, 50_000, (4, 32)),
        "labels": torch.randint(0, 50_000, (4, 32)),
    }
    logits, loss = model(batch["input_ids"], labels=batch["labels"])
    print(f"\n✅ Forward: input [{batch['input_ids'].shape}] → logits [{logits.shape}], loss={loss.item():.4f}")


if __name__ == "__main__":
    demo_causal_lm()
```

---

#### Вопрос: Когда мы используем предобученную модель?

**Ответ:**

Предобученные модели используются практически **всегда** в NLP (кроме очень специализированных случаев).

**Практический подход:**

|Сценарий|Решение|
|---|---|
|**Нет данных / Мало данных (< 1000)**|Используем SOTA предобученную модель (BERT, RoBERTa) + минимальный fine-tuning (1-2 слоя)|
|**Есть данные на целевом языке / домене**|Fine-tune предобученную модель на своих данных (классический подход)|
|**Очень специализированный домен**(медицина, право)|Используем domain-specific модель (SciBERT, LegalBERT) или дополнительно обучаем на domain текстах|
|**Нужна генерация текста / чат**|Используем GPT-like модель (GPT-2, GPT-3.5, LLaMA) + fine-tune с инструкциями|
|**Огромный бюджет, уникальная задача**|Обучаем с нуля (Google, OpenAI, Meta)|

**Пример Fine-tuning:**

```python
from __future__ import annotations

import torch
from torch.optim import Adam
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)  # [web:77]


def freeze_layers(model: AutoModelForSequenceClassification, unfreeze_last_n: int = 1) -> None:
    """
    Замораживает параметры модели кроме последних N слоёв энкодера.
    """
    # Замораживаем весь backbone (encoder)
    for param in model.roberta.parameters():  # [web:52]
        param.requires_grad = False

    # Оттаиваем только последние N слоёв
    for layer in model.roberta.encoder.layer[-unfreeze_last_n:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Classification head всегда trainable по умолчанию
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable параметров: {trainable_params:,}/{total_params:,} ({100*trainable_params/total_params:.1f}%)")


def create_optimizer(
    model: AutoModelForSequenceClassification,
    lr: float = 2e-5,
) -> Adam:
    """
    Создаёт оптимизатор только для trainable параметров.
    """
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    return Adam(trainable_params, lr=lr)  # [web:52]


def fine_tune_classifier(
    model: AutoModelForSequenceClassification,
    train_dataloader,
    device: torch.device,
    num_epochs: int = 3,
    lr: float = 2e-5,
) -> None:
    """
    Fine-tuning классификатора с замороженным backbone.
    """
    model.to(device)
    model.train()

    optimizer = create_optimizer(model, lr)

    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", None).to(device) if batch.get("attention_mask") is not None else None
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss  # [web:52]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загружаем предобученную модель
    model_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # [web:77]
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,  # бинарная классификация
    )  # [web:52]

    # Замораживаем backbone, оставляем trainable только последний слой + head
    freeze_layers(model)

    # Здесь предполагается, что train_dataloader уже создан.
    # Примерно так:
    # train_dataloader = DataLoader(tokenized_dataset["train"], batch_size=16, shuffle=True)
    
    # fine_tune_classifier(model, train_dataloader, device)

    pass


if __name__ == "__main__":
    main()
```

**Когда НЕ используем предобученные:**

- Обучаем модель на собственной инфраструктуре (Google, Meta, OpenAI).
    
- Очень экзотический язык или домен (может не быть предобученной).
    
- Нужна абсолютная приватность (модель на premise, не хотим использовать чужую).
    

---

#### Вопрос: Как обучить transformer с нуля? Объясните свой пайплайн и в каком случае вы будете этим заниматься.

**Ответ:**

**Когда это нужно:**

1. Обучаем model для нового языка (например, украинский, где нет хороших претренированных).
    
2. Очень специализированный домен с миллиардами токенов (финансовые данные, научные статьи).
    
3. Исследование (want to publish, need custom architecture).
    
4. Компания вроде OpenAI/Meta (огромные ресурсы).
    

**Мой пайплайн:**

**Этап 1: Подготовка данных**

```python
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List

from datasets import load_dataset  # [web:77]
from transformers import AutoTokenizer  # [web:77]


def load_corpus_dataset(
    dataset_name: str = "wikitext",
    config_name: str = "wikitext-103-v1",
) -> dict[str, any]:
    """
    Загружает датасет для обучения языковой модели.
    Рекомендации по размеру корпуса:
    - Base модель (12 слоев, ~110M параметров): 10+ млрд токенов
    - Большая модель (как GPT-3): 1+ трлн токенов
    Источники: Common Crawl, Wikipedia, GitHub, научные статьи.
    """
    dataset = load_dataset(dataset_name, config_name)  # [web:77]
    return dataset


def create_tokenizer(tokenizer_name: str = "bert-base-uncased") -> AutoTokenizer:
    """Создаёт токенайзер для предобработки корпуса."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)  # [web:77]
    
    # Добавляем pad_token если его нет (важно для батчинга)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


def tokenize_corpus(
    dataset: dict[str, any],
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    batched: bool = True,
) -> dict[str, any]:
    """Токенизирует весь датасет батчами."""
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",  # добавляем для унификации
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=batched,
        remove_columns=dataset["train"].column_names,  # убираем исходный текст
    )
    
    return tokenized_dataset


def deduplicate_dataset(
    tokenized_dataset: dict[str, any],
    hash_column: str = "input_ids",
    sample_size: int | None = None,
) -> List[dict]:
    """
    Деупликация датасета по хешу токенов.
    Важно для качества: убираем повторяющиеся документы.
    """
    seen = set()
    unique_docs = []
    
    # Берём только train сплит для деупликации
    docs = tokenized_dataset["train"]
    
    if sample_size:
        docs = docs.select(range(sample_size))  # для быстрого тестирования
    
    for doc in docs:
        # Хешируем input_ids как строку (быстро и надёжно)
        doc_hash = hashlib.md5(str(doc[hash_column]).encode()).hexdigest()
        
        if doc_hash not in seen:
            seen.add(doc_hash)
            unique_docs.append(doc)
    
    print(f"Уникальных документов: {len(unique_docs)} из {len(docs)}")
    return unique_docs


def main() -> None:
    # 1. Загрузка корпуса
    dataset = load_corpus_dataset()
    
    # 2. Токенизация
    tokenizer = create_tokenizer()
    tokenized_dataset = tokenize_corpus(dataset, tokenizer)
    
    # 3. Деупликация (на малом сэмпле для примера)
    unique_docs = deduplicate_dataset(tokenized_dataset, sample_size=10_000)
    
    # Сохраняем результат
    # tokenized_dataset.save_to_disk("deduped_tokenized_corpus")


if __name__ == "__main__":
    main()
```

**Этап 2: Выбор архитектуры**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PretrainedConfig,
)  # [web:77]


@dataclass
class CustomTransformerConfig(PretrainedConfig):
    """
    Кастомная конфигурация для трансформера (аналог GPT-2).
    
    Args:
        vocab_size: размер словаря
        hidden_size: размерность эмбеддингов (d_model)
        num_hidden_layers: количество трансформер-блоков
        num_attention_heads: количество голов attention
        intermediate_size: размер промежуточного слоя в FFN
        max_position_embeddings: максимальная длина последовательности
    """
    vocab_size: int = 50_257
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3_072
    max_position_embeddings: int = 1_024

    def __post_init__(self) -> None:
        super().__post_init__()


def create_model_from_config(config: PretrainedConfig) -> AutoModelForCausalLM:
    """
    Создаёт модель для causal LM из конфигурации (без весов).
    """
    model = AutoModelForCausalLM.from_config(config)  # [web:77]
    return model


def main() -> None:
    # Вариант 1: Берем готовую конфигурацию GPT-2
    gpt2_config = AutoConfig.from_pretrained("gpt2")  # [web:77]
    gpt2_model = create_model_from_config(gpt2_config)
    print(f"GPT-2 модель создана: {gpt2_model}")

    # Вариант 2: Наша кастомная конфигурация
    custom_config = CustomTransformerConfig(
        vocab_size=30_000,           # под наш BPE-токенайзер
        hidden_size=512,             # компактнее
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=2_048,
        max_position_embeddings=512,
    )
    custom_model = create_model_from_config(custom_config)
    print(f"Кастомная модель создана: {custom_model}")


if __name__ == "__main__":
    main()
```

**Этап 3: Обучение**

```python
from __future__ import annotations

import torch
from transformers import Trainer, TrainingArguments  # [web:77]


def build_training_args(output_dir: str = "./my_pretrained_model") -> TrainingArguments:
    """
    Создаёт набор параметров обучения для Trainer.
    """
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=32,      # можно поднять при достаточной памяти GPU [web:66]
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=4,       # эффективный batch = 32 * 4 = 128
        learning_rate=5e-4,
        weight_decay=0.01,
        warmup_steps=10_000,                 # прогрев learning rate [web:66]
        logging_steps=100,
        eval_steps=5_000,
        save_strategy="steps",
        save_steps=5_000,
        max_grad_norm=1.0,                   # gradient clipping [web:66]
        fp16=True,                           # half precision (mixed precision) [web:66]
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )


def build_trainer(model, tokenized_dataset, training_args: TrainingArguments) -> Trainer:
    """
    Оборачивает модель и датасеты в Trainer.
    """
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
    )  # [web:66][web:77]


def manual_train_loop(
    model,
    train_dataloader,
    device: torch.device,
    num_epochs: int = 3,
    lr: float = 5e-4,
    max_steps: int | None = 100_000,
) -> None:
    """
    Пример ручного цикла обучения с AdamW, cosine LR и gradient clipping.
    """
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)  # [web:66]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_steps or len(train_dataloader) * num_epochs,
    )

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)

            # Causal LM: input_ids используются и как вход, и как labels [web:66]
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # [web:66]

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if step % 100 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Здесь предполагается, что model, tokenized_dataset и train_dataloader уже созданы.
    # Пример использования Trainer:
    # training_args = build_training_args()
    # trainer = build_trainer(model, tokenized_dataset, training_args)
    # trainer.train()
    #
    # Пример использования ручного цикла:
    # manual_train_loop(model, train_dataloader, device)

    pass


if __name__ == "__main__":
    main()
```

**Этап 4: Оценка и сохранение**

```python
from __future__ import annotations

import math
import torch
from transformers import AutoModel  # [web:77]


def evaluate_perplexity(model, eval_dataloader, device: torch.device) -> float:
    """
    Оценка perplexity на контрольной выборке для языковой модели.
    Предполагается, что модель возвращает loss при передаче labels.
    """
    model.eval()
    model.to(device)

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch["input_ids"].to(device)
            outputs = model(input_ids=input_ids, labels=input_ids)  # [web:66][web:70]
            total_loss += outputs.loss.item()
            num_batches += 1

    mean_loss = total_loss / max(num_batches, 1)
    perplexity = math.exp(mean_loss)  # exp(средний loss) даёт perplexity. [web:62][web:76]

    print(f"Perplexity: {perplexity:.2f}")
    return perplexity


def save_model_and_tokenizer(model, tokenizer, save_dir: str = "./my_pretrained_model") -> None:
    """
    Сохраняет модель и токенайзер для последующей загрузки.
    """
    model.save_pretrained(save_dir)  # [web:77]
    tokenizer.save_pretrained(save_dir)  # [web:77]
    print(f"Модель и токенайзер сохранены в {save_dir}")


def load_base_model(save_dir: str = "./my_pretrained_model"):
    """
    Загружает сохранённую модель как базовый AutoModel (без головы). [web:77]
    """
    model = AutoModel.from_pretrained(save_dir)  # [web:69][web:77]
    return model


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Здесь предполагается, что model, tokenizer и eval_dataloader уже созданы и обучены ранее.
    # Примерно так:
    # model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # eval_dataloader = DataLoader(...)

    # evaluate_perplexity(model, eval_dataloader, device)
    # save_model_and_tokenizer(model, tokenizer)

    # Пример последующей загрузки:
    # loaded_model = load_base_model("./my_pretrained_model")
    pass


if __name__ == "__main__":
    main()
```
**Сколько это стоит?**

|Модель|Параметров|Минимум GPU часов|Стоимость (на AWS)|
|---|---|---|---|
|BERT-base|110M|1,000 часов (40 дней)|$20,000|
|BERT-large|340M|4,000 часов|$80,000|
|GPT-2|1.5B|10,000 часов|$200,000|
|GPT-3|175B|300+ дней на A100|$10,000,000+|

**Практические советы:**

1. Начни с маленькой модели (110M параметров) и проверь пайплайн перед тем как скейлить.
    
2. Используй mixed precision (fp16) для экономии памяти и ускорения.
    
3. Распределенное обучение на multiple GPUs (Data Parallel, Distributed Data Parallel).
    
4. Используй framework'и (Hugging Face Trainer, DeepSpeed), не пиши с нуля.
    

---

#### Вопрос: Какие модели кроме BERT и GPT по различным задачам предобучения вы знаете?

**Ответ:**

**Классификация / Понимание (NLU):**

|Модель|Архитектура|Фокус|Когда использовать|
|---|---|---|---|
|**RoBERTa**|Encoder-only|Улучшенный BERT (динамическое маскирование)|Классификация, поиск. Лучше BERT.|
|**DeBERTa**|Encoder-only|Относительный позиционный bias|Когда нужна максимальная точность (GLUE benchmark).|
|**ALBERT**|Encoder-only|Легкая версия BERT (parameter sharing)|Когда ограничены ресурсы (мобильные).|
|**ELECTRA**|Encoder-only|Дискриминативное предобучение (обнаружение замен)|Меньше данных нужно, более эффективно.|
|**XLNet**|Decoder (permutation LM)|Autoregressive + bidirectional context|На некоторых бенчмарках лучше BERT.|

---

**Генерация (NLG):**

|Модель|Архитектура|Фокус|Когда использовать|
|---|---|---|---|
|**GPT-2**|Decoder-only|Основа для генерации|Базовая генерация, fine-tune для чатов.|
|**GPT-3**|Decoder-only|Few-shot learning (in-context)|Когда мало labeled данных, можно использовать примеры.|
|**LLaMA**|Decoder-only|Эффективная генерация|Open source альтернатива GPT-3, можно fine-tune.|
|**Mistral**|Decoder-only|Еще более легкая, быстрая|Когда нужна скорость (инференс на CPU).|

---

**Seq2Seq (перевод, суммаризация):**

|Модель|Архитектура|Фокус|Когда использовать|
|---|---|---|---|
|**T5**|Encoder-Decoder|"Text-to-Text" парадигма|Универсальна: перевод, QA, суммаризация одной моделью.|
|**BART**|Encoder-Decoder|Denoising autoencoder|Похожа на T5, хорошо для генерации.|
|**mBART**|Encoder-Decoder|Multilingual BART|Перевод между 50+ языками.|

---

**Специализированные:**

|Модель|Область|Фокус|
|---|---|---|
|**SciBERT**|Научные статьи|Предобучена на arXiv + ACL|
|**LegalBERT**|Юридические документы|Предобучена на судебных решениях|
|**BioBERT**|Биомедицина|Для извлечения информации из статей|
|**CodeBERT**|Программирование|Код + комментарии|
|**DistilBERT**|Все|Дистиллированный BERT (40% параметров, 60% скорости)|

---

**Multilingual:**

|Модель|Языки|Применение|
|---|---|---|
|**mBERT**|104 языка|Кроссязычный трансфер (обучи на английском, юзай на русском)|
|**XLM-RoBERTa**|100+ языков|Лучше mBERT|
|**mT5**|101 язык|T5 multilingual версия|

---

**Рекомендация по выбору:**

```python
from __future__ import annotations

from transformers import pipeline  # [web:9]


def build_pipelines() -> dict[str, any]:
    """
    Создает и возвращает набор готовых NLP-пайплайнов под разные задачи.
    """
    # Классификация тональности
    sentiment_classifier = pipeline(
        task="sentiment-analysis",
        model="roberta-base",  # RoBERTa обычно даёт более точные результаты, чем BERT, на sentiment-задачах. [web:52]
    )

    # Генерация текста
    text_generator = pipeline(
        task="text-generation",
        model="gpt2",  # GPT‑2 часто используется как базовая модель для генерации текста. [web:52]
    )

    # Перевод EN → DE
    translator_en_de = pipeline(
        task="translation_en_to_de",
        model="t5-base",  # T5 хорошо работает как универсальная seq2seq модель, включая перевод. [web:52]
    )

    # Вопрос‑ответ по контексту
    question_answering = pipeline(
        task="question-answering",
        model="bert-base-uncased",  # BERT изначально проектировался, в том числе, под QA-задачи. [web:9]
    )

    # Универсальный text2text (перефразирование, суммаризация, перевод и т.п.)
    t5_text2text = pipeline(
        task="text2text-generation",
        model="t5-base",  # T5 использует формат text‑to‑text для широкого спектра задач. [web:52]
    )

    return {
        "sentiment": sentiment_classifier,
        "generator": text_generator,
        "translator_en_de": translator_en_de,
        "qa": question_answering,
        "t5_text2text": t5_text2text,
    }


def main() -> None:
    pipes = build_pipelines()

    text = "I absolutely love using transformers for NLP!"
    print(pipes["sentiment"](text))

    prompt = "Once upon a time in machine learning,"
    print(pipes["generator"](prompt, max_new_tokens=30))

    print(pipes["translator_en_de"]("Transformers are widely used in NLP."))

    qa_input = {
        "question": "What library is used?",
        "context": "The transformers library by Hugging Face is widely used in NLP.",
    }
    print(pipes["qa"](qa_input))

    print(pipes["t5_text2text"]("summarize: Transformers have changed NLP forever."))


if __name__ == "__main__":
    main()
```
## Tokenizers: Полный обзор токенизации в NLP

---

#### Вопрос: Какие виды токенайзеров вы знаете? Сравните их.

**Ответ:**

Токенизер разбивает текст на токены (слова или подслова) и преобразует их в числовые ID.

**Три основных вида:**

**1. Byte-Pair Encoding (BPE)**

**Как работает:**

- Начинаем со словаря, состоящего из **отдельных символов** (A, B, C, ...).
    
- Итеративно находим **самую частую пару** соседних символов и объединяем их.
    
- Повторяем, пока не достигнем желаемого размера словаря.
    

**Пример:**

```text
Шаг 0 (инициализация): d, e, e, p, l, e, a, r, n, i, n, g Частоты: e=3, n=2, других=1 Шаг 1: Объединяем "e" + "e" → "ee" (частота 2) Новый словарь: d, ee, p, l, a, r, n, i, g Шаг 2: "e" + "e" больше нет, ищем новую пару Частоты: "ng"=2, "ar"=1, ... Объединяем "ng" → "ng" Шаг 3: Объединяем "in" → "in" ...
```

**Финальная токенизация:** "deep" → ["d", "ee", "p"] →

**Плюсы:**

- Простая и интуитивная.
    
- Хорошо обрабатывает редкие слова (разбивает на подслова).
    
- Используется в GPT-2, GPT-3.
    

**Минусы:**

- Требует предварительной разбивки на слова (язык-зависимо).
    
- Может быть неоптимально (выбирает частоту, не качество).
    

---

**2. WordPiece**

**Как работает:**

- Похожа на BPE, но выбирает пары **не по частоте**, а по **likelihood**.
    
- При объединении пары, модель смотрит: насколько это улучшит вероятность корпуса?
    

**Математика:**

Вероятность пары = $\frac{P(\text{pair})}{P(\text{token1}) \times P(\text{token2})}$

Выбираем пару с **максимальной** этой вероятностью.

**Пример:**

```text
Пары:    "e" + "e"    "i" + "n"    "d" + "e" Вероятности:  2/100      15/1000      5/500 P(ee) / (P(e)*P(e)) = 0.02 / (0.01 * 0.01) = 200 P(in) / (P(i)*P(n)) = 0.015 / (0.05 * 0.1) = 3 P(de) / (P(d)*P(e)) = 0.01 / (0.02 * 0.01) = 50 Выбираем пару "ee" (максимум 200)
```

**Плюсы:**

- Более интеллектуальная выборка пар (оптимизирует likelihood).
    
- Используется в BERT, RoBERTa, ALBERT.
    
- Более компактный словарь.
    

**Минусы:**

- Требует предварительной разбивки на слова.
    
- Сложнее для языков без пробелов (китайский, японский).
    

---

**3. SentencePiece**

**Как работает:**

- Обрабатывает текст как **поток символов** (включая пробелы).
    
- Пробел кодируется как специальный символ `▁` (underscore).
    
- Может использовать BPE или Unigram алгоритм внутри.
    

**Пример:**

```text
Входной текст: "hello world" После обработки: "▁hello▁world" (пробелы как ▁) Токенизация: ["▁h", "e", "l", "l", "o", "▁w", "o", "r", "l", "d"] или с подсловами: ["▁hello", "▁world"]
```

**Плюсы:**

- Работает с **любыми** языками (без предварительной разбивки).
    
- Можно восстановить оригинальный текст из токенов.
    
- Используется в T5, mBERT, mT5.
    

**Минусы:**

- Медленнее на инференсе (специальная обработка пробелов).
    

---

**Сравнительная таблица:**

|Характеристика|BPE|WordPiece|SentencePiece|
|---|---|---|---|
|**Критерий выбора пары**|Частота|Likelihood (вероятность)|BPE или Unigram|
|**Предварительная разбивка**|Нужна|Нужна|НЕ нужна|
|**Поддержка языков без пробелов**|Плохо|Плохо|Отлично|
|**Восстановление текста**|Сложно|Сложно|Легко (▁ = пробел)|
|**Используется в**|GPT-2, GPT-3, Mistral|BERT, RoBERTa, DeBERTa|T5, mBERT, LLaMA|

---

#### Вопрос: Можете ли вы расширять токенайзер? Если да, то в каком случае вы будете этим заниматься? Когда вы будете переобучать токенайзер? Что необходимо сделать при добавлении новых токенов?

**Ответ:**

**Да, токенайзер можно расширять.**

**Когда нужно добавлять новые токены:**

1. **Domain-специфичные термины:** Медицинские слова, химические формулы, которые не в оригинальном словаре.
    
2. **Новые языки / диалекты:** Добавляем символы нового языка.
    
3. **Special tokens:** `[CLS]`, `[MASK]`, `</s>`, пользовательские токены.
    

**Процесс добавления токенов:**

```python
from __future__ import annotations

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_base_model(model_name: str = "bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # [web:2][web:9]
    model = AutoModelForSequenceClassification.from_pretrained(model_name)  # [web:52]
    return tokenizer, model


def extend_tokenizer_vocab(tokenizer, model, new_tokens: list[str]) -> int:
    """
    Добавляет новые токены в токенайзер и
    расширяет embedding-слой модели.
    """
    print(f"Размер словаря до расширения: {len(tokenizer)}")

    num_added = tokenizer.add_tokens(new_tokens)  # [web:44]
    if num_added == 0:
        print("Новые токены уже присутствуют в словаре.")
        return 0

    print(f"Добавлено {num_added} токенов")
    # Расширяем embedding-матрицу под новый словарь
    model.resize_token_embeddings(len(tokenizer))  # [web:44][web:42]

    print(f"Размер словаря после расширения: {len(tokenizer)}")
    return num_added


def fine_tune(
    model,
    train_dataloader,
    device: torch.device,
    lr: float = 2e-5,
    num_epochs: int = 3,
) -> None:
    """
    Простая петля дообучения на доменных данных.
    """
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # [web:52]

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, labels=labels)  # [web:9]
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch + 1}/{num_epochs} — loss: {loss.item():.4f}")


def save_model_and_tokenizer(model, tokenizer, save_dir: str = "./medical_bert") -> None:
    tokenizer.save_pretrained(save_dir)  # [web:44]
    model.save_pretrained(save_dir)  # [web:52]
    print(f"Модель и токенайзер сохранены в {save_dir}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer, model = load_base_model()

    # Новые токены для медицинского домена
    new_tokens = ["[DISEASE]", "[DRUG]", "[PATIENT]", "COVID-19"]

    extend_tokenizer_vocab(tokenizer, model, new_tokens)

    # Здесь предполагается, что train_dataloader уже определён снаружи
    # и отдаёт батчи формата {"input_ids": ..., "labels": ...}
    # fine_tune(model, train_dataloader, device)

    save_model_and_tokenizer(model, tokenizer)


if __name__ == "__main__":
    main()
```
**Когда переобучать токенайзер полностью:**

Переобучение токенайзера с нуля нужно, когда:

1. Новый язык (русский, китайский, арабский).
    
2. Очень специализированный домен (генетика, юриспруденция).
    
3. Существующий токенайзер неэффективен (слишком много OOV слов).
    

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


def train_medical_tokenizer(
    corpus_paths: list[str],
    vocab_size: int = 30_000,
    min_frequency: int = 2,
    save_path: str = "medical_tokenizer.json",
) -> Tokenizer:
    """
    Обучает BPE-токенайзер на заданном корпусе и сохраняет его в файл.
    """
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))  # [web:25]

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[MASK]", "[PAD]"],  # [web:25][web:32]
    )  # [web:34]

    # Предварительная токенизация по пробелам
    tokenizer.pre_tokenizer = Whitespace()  # [web:25][web:36]

    # Обучение на своем корпусе
    tokenizer.train(files=corpus_paths, trainer=trainer)  # [web:25][web:28]

    # Сохранение в JSON
    tokenizer.save(save_path)  # [web:30]

    return tokenizer


if __name__ == "__main__":
    train_medical_tokenizer(["my_corpus.txt"])
```

**Что происходит при добавлении токенов:**

1. **Токен добавляется в словарь** (новый ID).
    
2. **Embedding инициализируется случайно** (нужно обучить).
    
3. **Модель должна быть переразмерена** (`resize_token_embeddings`).
    
4. **Fine-tune на новых данных**, чтобы эмбеддинг был полезным.
    

---

#### Вопрос: Чем обычные токены отличаются от специальных токенов?

**Ответ:**

**Обычные токены (Regular tokens):**

- Представляют **реальные слова/подслова** из текста.
    
- Пример: "hello", "world", "##ing", "▁le" (в SentencePiece).
    

**Специальные токены (Special tokens):**

- Имеют **служебную функцию** в модели.
    
- Не появляются в исходном тексте (добавляются автоматически).
    
- Пример: `[CLS]`, `[SEP]`, `[MASK]`, `[PAD]`, `[UNK]`.
    

**Таблица специальных токенов BERT:**

|Токен|Назначение|
|---|---|
|`[CLS]`|В начале предложения. Выход используется для классификации.|
|`[SEP]`|Разделитель между двумя предложениями (для задач вроде similarity).|
|`[MASK]`|Маскирует слово при обучении. Модель учится его предсказывать.|
|`[PAD]`|Заполняет короткие предложения до фиксированной длины.|
|`[UNK]`|Unknown token — если слово не в словаре, заменяем на `[UNK]`.|

**Как они используются:**

```python
from transformers import AutoTokenizer


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # [web:14]

    text = "Hello world"

    # Обычная токенизация (добавляются спецтокены автоматически)
    tokens_with_special = tokenizer.encode(text, add_special_tokens=True)
    print(tokenizer.convert_ids_to_tokens(tokens_with_special))  # [CLS], hello, world, [SEP] [web:7]

    # Токенизация без специальных токенов
    tokens_without_special = tokenizer.encode(text, add_special_tokens=False)  # [web:1]
    print(tokenizer.convert_ids_to_tokens(tokens_without_special))  # hello, world [web:7]

    # Маскирование для MLM (Masked Language Modeling)
    masked_tokens = tokens_with_special.copy()
    masked_tokens[1] = tokenizer.mask_token_id  # маскируем "hello" [web:12]
    print(tokenizer.convert_ids_to_tokens(masked_tokens))  # [CLS], [MASK], world, [SEP] [web:7]


if __name__ == "__main__":
    main()
```
---

#### Вопрос: Почему в трансформерах не используется лемматизация? И зачем нам нужны токены?

**Ответ:**

**Почему нет лемматизации:**

**Лемматизация = приведение слова к нормальной форме.**

```text
"running", "runs", "ran" → "run" "better", "best" → "good"
```

В трансформерах лемматизация **не используется** по нескольким причинам:

1. **Потеря информации:**  
    Лемма "run" теряет информацию о том, что это было "running" (форма Present Continuous). Трансформер лучше работает, если сохраняет эту информацию через **контекст**.
    
2. **Токенизер уже разбивает на морфемы:**  
    BPE/WordPiece разбивают "running" на ["run", "##ing"]. Это уже достаточно для модели. Она видит корень "run" и суффикс "ing", может выучить их отдельные эмбеддинги.
    
3. **Контекст достаточен:**  
    Трансформер использует Self-Attention. Если "running" появилось, модель знает контекст вокруг него и может понять значение. Не нужно явно приводить к лемме.
    
4. **Языки без четких лемм:**  
    Для китайского, японского нет очевидной лемматизации. Единый подход (токенизация) работает везде.
    

**Пример:**

```text
Предложение: "The runner is running in the running shoes" С лемматизацией: "The run be run in the run shoes"  ← Потеря информации! Без лемматизации (с BPE): ["The", "run", "##ner", "is", "run", "##ning", "in", "the", "run", "##ning", "shoes"] Модель видит: корень "run" + разные суффиксы (##ner, ##ning) и понимает связь.
```

---

**Зачем нам нужны токены:**

1. **Числовое представление:** Нейросети работают с числами, а не со строками.
    
2. **Эффективность:** Вместо 1000 символов можно использовать 100 токенов.
    
3. **Эмбеддинги:** Каждый токен имеет вектор (embedding), который модель обучает.
    
4. **Обработка редких слов:** Если слова нет, разбиваем на подслова.
    
5. **Стандартизация:** Разные версии одного слова (case, punctuation) → один токен.
    

```python
# Example of why tokens are more efficient
text = "Transformers are awesome!"

# Character level processing
chars = list(text)
print(f"Number of characters: {len(chars)}")  # 27 characters

# Token level processing
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize(text)
print(f"Number of tokens: {len(tokens)}")  # 5 tokens
print(tokens)  # ['transformers', 'are', 'awesome', '!']

# The model processes 5 tokens instead of 27 characters!
# This saves memory and speeds up training.
```
---

#### Вопрос: Как обучается токенизатор? Объясните на примерах WordPiece и BPE.

**Ответ:**

**BPE (Byte-Pair Encoding) — Пошагово:**

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# 1. Initialize the BPE model
bpe_model = models.BPE(unk_token="[UNK]")
tokenizer = Tokenizer(bpe_model)

# 2. Pre-tokenization (split into words)
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# 3. Create a trainer
trainer = trainers.BpeTrainer(
    vocab_size=30000,          # Target vocabulary size
    min_frequency=2,           # Ignore subwords appearing less than 2 times
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

# 4. Train on the corpus
tokenizer.train(["corpus.txt"], trainer=trainer)

# Example of BPE operation:
# Corpus: "hello hello world"
# Step 1: Initialization (characters)
# Vocab: {h, e, l, l, o, w, o, r, d}
# Step 2: Count pairs and their frequencies
# "h" + "e" = 2 times
# "e" + "l" = 4 times
# "l" + "l" = 2 times
# "l" + "o" = 4 times
# "o" + " " = 2 times
# etc.
# Step 3: Merge the most frequent pair
# "e" + "l" appears 4 times → merge into "el"
# Vocab: {..., el}
# Text transforms: "h el l o" instead of "h e l l o"
# Step 4: Repeat (find next most frequent pair, etc.)
# Continue until reaching vocab_size = 30000
```
---

**WordPiece (Google's approach) — Пошагово:**

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# 1. Initialize the WordPiece model
wordpiece_model = models.WordPiece(unk_token="[UNK]")
tokenizer = Tokenizer(wordpiece_model)

# 2. Pre-tokenization
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# 3. Create a trainer for WordPiece
trainer = trainers.WordPieceTrainer(
    vocab_size=30000,
    min_frequency=2,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

# 4. Train on the corpus
tokenizer.train(["corpus.txt"], trainer=trainer)

# Example of WordPiece operation:
# Corpus: "playing played plays play"
# Step 1: Initialization (all characters)
# Vocab: {p, l, a, y, i, n, g, d, e}
# Step 2: Calculate all pairs and their probabilities
# Pair: "p" + "l"
# P(pl) = freq(pl) / (freq(p) * freq(l)) = ?
# Pair: "l" + "a"
# P(la) = freq(la) / (freq(l) * freq(a)) = ?
# Pair: "a" + "y"
# P(ay) = freq(ay) / (freq(a) * freq(y)) = ?
# Pair: "i" + "n"
# P(in) = freq(in) / (freq(i) * freq(n)) = ?
# Pair: "n" + "g"
# P(ng) = freq(ng) / (freq(n) * freq(g)) = ?
# Step 3: Choose the pair with the HIGHEST probability
# Suppose P(ing) = 0.95 (highest)
# Combine "i" + "n" + "g" → "ing"
# Vocab: {..., ing}
# Now "playing" → ["play", "ing"]
# Step 4: Repeat process for remaining pairs
# Find the next pair with the highest probability (not frequency!)
```
---

**Сравнение BPE vs WordPiece:**

|Аспект|BPE|WordPiece|
|---|---|---|
|**Критерий выбора**|Максимальная частота пары|Максимальная вероятность (likelihood)|
|**Эффективность**|Может быть неоптимально|Лучше для качества модели|
|**Скорость обучения**|Быстрее|Медленнее (нужно считать вероятности)|
|**Используется в**|GPT-2, OpenAI|BERT, Google (исторически)|

---

#### Вопрос: На какой позиции стоит CLS вектор? Почему?

**Ответ:**

**CLS токен стоит в НАЧАЛЕ (позиция 0) предложения.**

**Почему именно в начале:**

1. **Стандартизация:** Это стандартная позиция. Легко всегда использовать `output[0]` (первый элемент) для получения представления всего предложения.
    
2. **Распространение градиентов:** При обратном проходе градиенты текут от классификационного слоя → CLS эмбеддинг. Если CLS в начале, градиенты доходят до него напрямую, без прохождения через другие токены.
    
3. **Агрегация информации:** Self-Attention позволяет CLS "смотреть" на все остальные токены. Благодаря механизму внимания, CLS собирает информацию от всего предложения.
    

```python
from transformers import AutoTokenizer, AutoModel
import torch

# Initialize the BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Tokenize the text and prepare inputs
text = "Hello world"
inputs = tokenizer(text, return_tensors="pt")

# inputs['input_ids']: [[101, 7592, 2088, 102]]
# where 101 = [CLS], 7592 = "hello", 2088 = "world", 102 = [SEP]

# Get model outputs
outputs = model(**inputs)

# last_hidden_state: [batch_size, seq_len, hidden_size] -> [1, 4, 768]
cls_representation = outputs.last_hidden_state[:, 0, :]  # Take position 0 ([CLS])
print(cls_representation.shape)  # [1, 768] - representation of the entire sentence

# For classification, this representation is fed into a classifier head:
# logits = classifier_head(cls_representation)
```

---

#### Вопрос: Какой токенизатор используется в BERT, а какой в GPT?

**Ответ:**

**BERT → WordPiece**

- Разработан Google.
    
- Выбирает пары по likelihood (максимум P(pair) / (P(token1) * P(token2))).
    
- Требует предварительной разбивки на слова.
    
- Размер словаря: ~30k токенов.
    

```python
from transformers import AutoTokenizer

# Initialize the BERT tokenizer
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the text
text = "Hello world"
tokens = bert_tokenizer.tokenize(text)
print(tokens)  # ['hello', 'world']

# Encode the text to get token IDs
ids = bert_tokenizer.encode(text)
print(ids)  # [101, 7592, 2088, 102] (with [CLS] and [SEP])
```
---

**GPT → BPE (Byte-Pair Encoding)**

- Разработан OpenAI.
    
- Выбирает пары по частоте (самая частая пара объединяется).
    
- Не требует предварительной разбивки (работает на уровне байтов, символов).
    
- Размер словаря: ~50k токенов (GPT-2), ~100k (GPT-3).
    

```python
from transformers import AutoTokenizer

# Initialize the GPT-2 tokenizer
gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Tokenize the text
text = "Hello world"
tokens = gpt_tokenizer.tokenize(text)
print(tokens)  # ['Hello', ' world'] (note the space before "world")

# Encode the text to get token IDs
ids = gpt_tokenizer.encode(text)
print(ids)  # [15496, 995] (without [CLS] and [SEP])
```

**Сравнение:**

|Аспект|BERT (WordPiece)|GPT (BPE)|
|---|---|---|
|**Алгоритм**|Likelihood-based|Frequency-based|
|**Размер словаря**|30k|50k-100k|
|**Спецтокены**|[CLS], [SEP], [MASK]|Нет (или минимум)|
|**Пробелы**|Разбивает слова|Пробел как часть токена|
|**Преимущество**|Компактнее|Больше контроля над подсловами|

---

#### Вопрос: Объясните как современные токенизаторы обрабатывают out-of-vocabulary words?

**Ответ:**

**Out-of-Vocabulary (OOV)** — это слова, которых нет в словаре токенайзера.

**Способы обработки:**

**1. Разбиение на подслова (Subword Tokenization) — основной метод**

Если слова нет целиком, разбиваем его на подслова, которые **есть** в словаре.

```python
from transformers import AutoTokenizer

# Initialize the BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# OOV word (made-up)
oov_word = "xlkjzxcv"
# Tokenizer breaks it into known subwords
tokens = tokenizer.tokenize(oov_word)
print(tokens)  # ['[UNK]'] or ['x', 'lk', 'j', 'z', 'x', 'c', 'v']

# Normal word
normal_word = "beautiful"
tokens = tokenizer.tokenize(normal_word)
print(tokens)  # ['beautiful'] or ['beautiful']

# Word that breaks into subwords
morphed_word = "unbelievable"
tokens = tokenizer.tokenize(morphed_word)
print(tokens)  # ['un', '##believable'] (## indicates continuation)
```
---

**2. [UNK] токен (Unknown token) — fallback**

Если слово не может быть разбито на известные подслова, оно заменяется на специальный токен `[UNK]`.

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# Слово, которое полностью неизвестно (содержит символы вне словаря)
text = "The ##$%^&()@ was strange"
tokens = tokenizer.tokenize(text)
print(tokens) # ['the', '[UNK]', 'was', 'strange']
```

---

**3. Character-level tokenization — дополнительный механизм**

SentencePiece и FastText могут работать на уровне **символов** в качестве fallback.

```python
from sentencepiece import SentencePieceProcessor

# Initialize the SentencePiece processor
sp = SentencePieceProcessor()
sp.load("sentencepiece.model")

# Even with a completely new word, SentencePiece breaks it into characters
tokens = sp.encode("unknownword")
print(tokens)  # [1, 2, 3, 4, 5, ...] (each character has an ID)

# This ensures that any text can be tokenized!
```

---

**4. Byte-level fallback — в BPE/GPT**

GPT использует byte-level BPE, что позволяет кодировать **любой** символ (даже эмодзи, спецсимволы).

```python
from transformers import AutoTokenizer

# Initialize the GPT-2 tokenizer
gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Even unusual characters are handled
weird_text = "Hello 🎉 world"
tokens = gpt_tokenizer.encode(weird_text)
print(tokens)  # Each character (including emoji) is encoded

decoded = gpt_tokenizer.decode(tokens)
print(decoded)  # "Hello 🎉 world" (accurately restored)
```

---

**Сравнение методов:**

|Метод|Как работает|Результат для OOV|
|---|---|---|
|**WordPiece**|Разбивает на подслова (##prefix)|"unknownword" → ["un", "##known", "##word"]|
|**BPE**|Byte-level encoding|"unknownword" → байты → токены|
|**SentencePiece**|Character-level + n-grams|"unknownword" → [symbols]|
|**[UNK] fallback**|Заменяет неизвестное|"###strange###" → "[UNK]"|

---

#### Вопрос: На что влияет tokenizer vocab size? Как вы будете его выбирать в случае нового обучения?

**Ответ:**

**Vocab Size (размер словаря)** — количество уникальных токенов, которые токенайзер может распознать.

**На что влияет:**

**1. Размер embedding матрицы (память)**

Embedding матрица: `[vocab_size, embedding_dim]`

- BERT (vocab=30k, dim=768): 30k × 768 = 23M параметров
    
- GPT-2 (vocab=50k, dim=768): 50k × 768 = 38M параметров
    
- GPT-3 (vocab=50k, dim=12288): 50k × 12288 = 614M параметров!
    

**Чем больше vocab_size, тем больше памяти нужно.**

---

**2. Длина предложения (tokens)**

Большой vocab_size → каждое слово может быть **целиком** в словаре → короче предложение в токенах.  
Маленький vocab_size → слова разбиваются на подслова → длинные предложения.

```Пример: "Hello world beautiful" Большой vocab (100k): [hello, world, beautiful] = 3 токена Маленький vocab (5k): [h, el, lo, w, o, r, l, d, b, eau, t, i, f, u, l] = 15 токенов!```

Длинные предложения → больше вычислений в Self-Attention ($O(n^2)$).

---

**3. Качество представления слов**

**Маленький vocab (< 10k):** OOV слова часто, модель разбивает их на много подслов, теряется информация.

**Большой vocab (> 100k):** Редкие слова целиком в словаре, но embedding для каждого подслова получает мало данных при обучении.

**Оптимальный vocab:** ~30k-50k (BERT/GPT-2).

---

**Как выбирать vocab_size при обучении с нуля:**

```python
# Эмпирическая рекомендация:
# Правило: vocab_size ≈ corpus_size_in_tokens / 50
corpus_size = 10_000_000  # 10M токенов
recommended_vocab = corpus_size / 50
print(f"Рекомендуемый vocab_size: {recommended_vocab}")  # 200k

# Но на практике:
# - Маленький корпус (< 100M токенов): vocab = 10k-30k
# - Средний корпус (100M-1B токенов): vocab = 30k-50k
# - Большой корпус (> 1B токенов): vocab = 50k-100k
```

**Практический пайплайн выбора:**

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import os
import glob
from collections import Counter

# 1. Оцениваем размер корпуса
corpus_size_bytes = sum(os.path.getsize(f) for f in glob.glob("corpus/*.txt"))
print(f"Размер корпуса: {corpus_size_bytes / 1e9:.1f}GB")

# 2. Считаем примерное количество слов / токенов
avg_word_length = 5  # средняя длина слова
corpus_size_tokens = corpus_size_bytes / avg_word_length
print(f"Примерно токенов: {corpus_size_tokens / 1e6:.1f}M")

# 3. Определяем vocab_size
if corpus_size_tokens < 100e6:
    vocab_size = 20000
elif corpus_size_tokens < 1e9:
    vocab_size = 30000
elif corpus_size_tokens < 10e9:
    vocab_size = 50000
else:
    vocab_size = 100000
print(f"Выбран vocab_size: {vocab_size}")

# 4. Обучаем токенайзер
bpe_model = models.BPE(unk_token="[UNK]")
tokenizer = Tokenizer(bpe_model)
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.BpeTrainer(
    vocab_size=vocab_size,
    min_frequency=2,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)
tokenizer.train(glob.glob("corpus/*.txt"), trainer=trainer)

# 5. Проверяем OOV rate
oov_count = 0
total_count = 0
test_texts = [...]  # Replace with actual test texts
for text in test_texts:
    tokens = tokenizer.encode(text).tokens
    for token in tokens:
        total_count += 1
        if token == "[UNK]":
            oov_count += 1
oov_rate = oov_count / total_count
print(f"OOV rate: {oov_rate:.2%}")  # Должно быть < 1%

# Если OOV rate > 1%, увеличиваем vocab_size
# Если OOV rate < 0.1% и используем много памяти, уменьшаем vocab_size
```


**Финальные рекомендации:**

|Сценарий|Vocab Size|Причина|
|---|---|---|
|**Маленький датасет**|10k-20k|Мало данных, риск переобучения на редких словах|
|**Стандартный (100M токенов)**|30k-50k|Оптимальный баланс качества и памяти|
|**Большой датасет (1B+ токенов)**|50k-100k|Достаточно данных для обучения редких слов|
|**Multilingual**|100k-200k|Много языков, больше уникальных символов|
|**Ограничена память (мобила)**|5k-10k|Маленькие embedding, но ухудшается качество|


## Training: Оптимизация и Данные

### Дисбаланс классов (Class Imbalance)

**Определение:** Ситуация, когда в данных один класс (мажоритарный) доминирует над другим (минорным). Например, в fraud detection 99.9% транзакций легальны, и лишь 0.1% — мошеннические.

**Как увидеть:**

1. **Гистограмма:** `df['target'].value_counts().plot(kind='bar')`.
    
2. **Dummy Classifier:** Если модель, предсказывающая всегда "0", дает 99% accuracy — у вас дисбаланс.
    
3. **Метрики:** Высокая Accuracy, но Recall/Precision для минорного класса $\approx 0$.
    

**Методы решения:**

|Метод|Описание|Trade-offs|
|---|---|---|
|**Resampling**|**Undersampling:** Выкидываем лишние мажорные примеры.  <br>**Oversampling:** Дублируем минорные примеры.|❌ Undersampling теряет данные.  <br>❌ Oversampling ведет к переобучению.|
|**Class Weights**|Даем больший штраф (Loss) за ошибку на редком классе. В PyTorch: `CrossEntropyLoss(weight=tensor([0.1, 0.9]))`.|✅ Просто и эффективно.  <br>⚠️ Может сделать обучение нестабильным.|
|**SMOTE**|Synthetic Minority Over-sampling Technique. Генерирует _новые_ точки между соседями минорного класса.|✅ Лучше простого дублирования.  <br>❌ Плохо работает на High-dim (текст/картинки).|
|**Focal Loss**|Модификация CrossEntropy, снижающая вес _легких_ примеров, фокусируясь на сложных.|✅ Стандарт де-факто в Object Detection (RetinaNet).|

---

### Оптимизаторы: SGD, RMSProp, Adam, AdamW

**Эволюция идей:**

1. **SGD (Stochastic Gradient Descent):** Просто идем в сторону антиградиента.
    
    - _Проблема:_ Застревает в локальных минимумах и седловых точках.
        
2. **SGD + Momentum:** Добавляем инерцию. Если градиент меняет знак (зигзаг), инерция сглаживает путь.
    
3. **RMSProp:** Адаптивный шаг. Делит градиент на среднее квадратичное прошлых градиентов.
    
    - _Суть:_ Если по какому-то весу градиенты огромные — замедляемся, если маленькие — ускоряемся.
        
4. **Adam (Adaptive Moment Estimation):** Momentum + RMSProp.
    
    - Считает среднее градиентов ($m_t$, инерция).
        
    - Считает средний квадрат градиентов ($v_t$, масштабирование).
        

**Adam vs AdamW:**  
Разница в **Weight Decay (L2 регуляризации)**.

- **Adam:** Добавляет штраф $\lambda w$ прямо в градиент. Из-за адаптивности Adam (деления на $v_t$), этот штраф искажается и работает некорректно для весов с большой/малой дисперсией.  
    wt+1=wt−η⋅(AdamStep+λwt)wt+1=wt−η⋅(AdamStep+λwt)
    
- **AdamW (Decoupled Weight Decay):** Выносит уменьшение весов из формулы градиента. Мы просто уменьшаем веса на каждом шаге, _независимо_ от градиента.  
    wt+1=wt−η⋅AdamStep−ηλwtwt+1=wt−η⋅AdamStep−ηλwt  
    _Результат:_ AdamW работает значительно лучше для Трансформеров.[mbrenndoerfer+1](https://mbrenndoerfer.com/writing/adamw-optimizer-decoupled-weight-decay)​
    

---

### Gradient Accumulation

**Проблема:** В GPU влезает батч размера 16, а для стабильного обучения нужно 256.  
**Решение:** Не обновляем веса (`optimizer.step()`) на каждом шаге. Копим градиенты 16 шагов (16 * 16 = 256), и только потом делаем шаг.

**Потребление ресурсов:**

- **Память:** Не меняется (равна 1 микро-батчу). ✅
    
- **Скорость:** Чуть медленнее, так как нужно делать N forward/backward проходов для одного обновления. Но снижаются накладные расходы на коммуникацию (в распределенном обучении).
    

**Код:**

```python
accumulation_steps = 4
model.zero_grad()
for i, (inputs, labels) in enumerate(dataloader):     
    outputs = model(inputs)    
    loss = criterion(outputs, labels)    
    loss = loss / accumulation_steps  # Важно: нормализация!    
    loss.backward()    
    if (i + 1) % accumulation_steps == 0:        
        optimizer.step()        
        model.zero_grad()
```

---

### Resource Optimization & Distributed Training

**Как оптимизировать потребление (одна GPU):**

1. **Mixed Precision (AMP):** `fp16` вместо `fp32`. Ускорение x2-3, память x2.
    
2. **Gradient Checkpointing:** Не храним промежуточные активации, а пересчитываем их в backward pass. Экономит 70% памяти ценой 20% времени.[uplatz](https://uplatz.com/blog/gradient-accumulation-a-comprehensive-technical-guide-to-training-large-scale-models-on-memory-constrained-hardware/)​
    
3. **Gradient Clipping:** Обрезка градиентов (`torch.nn.utils.clip_grad_norm_`), чтобы не взрывались в RNN/Transformer.
    

**Распределенное обучение (Multi-GPU):**

|Метод|Что делает|Когда использовать|
|---|---|---|
|**DDP** (Distributed Data Parallel)|Копия модели на каждой GPU. Данные бьются на части. Синхронизация градиентов.|Стандарт, если модель влезает в одну GPU.|
|**Model Parallelism**|Режем _модель_ по слоям (Pipeline) или тензорам (Tensor Parallel).|Если модель НЕ влезает в одну GPU (LLM).|
|**ZeRO / FSDP** (Fully Sharded Data Parallel)|Шардируем (разбиваем) не только данные, но и **состояния оптимизатора, градиенты и параметры модели**.|Обучение огромных моделей (миллиарды параметров) на кластерах.|

**ZeRO Stages:**

- **Stage 1:** Шардинг Optimizer States (память / 4).
    
- **Stage 2:** + Шардинг Gradients (память / 8).
    
- **Stage 3:** + Шардинг Parameters (линейное масштабирование памяти).
    

---

### Работа с текстом (NLP Specifics)

### Текстовые аугментации

В отличие от картинок, текст хрупок. Поменяешь слово — изменится смысл.

1. **Easy Data Augmentation (EDA):** Удаление слов, перестановка, вставка случайных слов.
    
2. **Back-translation:** Перевод RU → EN → RU. Получаем перефразированный текст.
    
3. **MixUp (на эмбеддингах):** Линейная интерполяция между векторами двух предложений.
    
4. **Synonym Replacement:** Замена слов на синонимы (через WordNet или BERT-masking).
    

### Padding vs Packing

- **Padding:** Добивание нулями до max_len. Если в батче одно предложение 1000 слов, а остальные 10 — мы считаем 99% мусора.
    
- **Решение (Packing):** Склеиваем короткие примеры в одну длинную последовательность через разделитель `[SEP]`. Используем **attention mask**, чтобы примеры не "видели" друг друга. Эффективность вычислений растет в разы.
    

### Warm-up

Линейное увеличение Learning Rate от 0 в начале обучения.  
**Зачем:** В начале веса рандомные, градиенты огромные. Если сразу дать большой LR, модель "улетит" (divergence). Warm-up дает весам стабилизироваться перед основной фазой обучения.

### Teacher Forcing (Seq2Seq)

При обучении декодера (RNN/GPT) на вход подаем **не то, что он сгенерировал** на прошлом шаге, а **правильный токен (ground truth)**.

- ✅ Быстрая сходимость.
    
- ❌ **Exposure Bias:** На тесте "учителя" нет, модель начинает ошибаться, ошибки накапливаются.
    

---

### Архитектура: Skip Connections & Adapters

**Skip Connection (Residual):** $x \to [Layer] \to F(x) + x$.

- Позволяет градиенту течь сквозь сеть беспрепятственно ("gradient highway"). Решает проблему затухания градиента в глубоких сетях (ResNet, Transformer).
    

**Adapters (PEFT - Parameter-Efficient Fine-Tuning):**  
Вставляем маленькие обучаемые слои _внутрь_ замороженного Трансформера.

- **Зачем:** Fine-tuning LLM под задачу. Обучаем всего 1-3% параметров.
    
- **Плюс:** Одна базовая модель + много легких адаптеров под разные задачи.
    

---

### Metric Learning

Задача: построить пространство, где похожие объекты близко, а разные — далеко.  
**Loss-функции:**

1. **Contrastive Loss (Siamese):** Работает с парами (0/1). Тянет "похожие" ($D \to 0$), толкает "разные" ($D > margin$).
    
2. **Triplet Loss:** Работает с тройками (Anchor, Positive, Negative).  
    L=max⁡(d(A,P)−d(A,N)+margin,0)L=max(d(A,P)−d(A,N)+margin,0)  
    Требует сложного майнинга "трудных" троек (hard negative mining).
    
3. **ArcFace:** SOTA для распознавания лиц. Добавляет margin в угловое пространство (на сфере).
    

---

## Inference

### Dropout на инференсе

- **Обычно:** ❌ **Нет**. Отключаем (`model.eval()`), веса масштабируются. Нам нужен детерминированный результат.
    
- **Исключение:** ✅ **Monte Carlo Dropout**. Включаем dropout на инференсе и делаем N прогонов. Дисперсия предсказаний показывает **неуверенность (uncertainty)** модели.
    

### Температура и Sampling

Pi=exp⁡(zi/T)∑exp⁡(zj/T)Pi=∑exp(zj/T)exp(zi/T)

- **$T < 1$:** "Острый" softmax. Модель уверена, консервативна.
    
- **$T > 1$:** "Плоский" softmax. Больше разнообразия, но больше бреда.
    

**Виды Sampling:**

1. **Greedy:** Всегда берем топ-1. Скучно, повторы.
    
2. **Top-K:** Обрезаем хвост, берем K лучших, сэмплируем.
    
3. **Top-P (Nucleus):** Берем топ токенов, чья сумма вероятностей $\ge P$ (например, 0.9). Адаптивный размер списка кандидатов.
    

### Beam Search

Вместо одной ветки держим $B$ лучших гипотез на каждом шаге.  
**Сложность:** $O(T \cdot B \cdot V)$ (или $O(T \cdot B \cdot \log B)$ с оптимизациями), где $B$ — ширина луча.

- Гарантирует более вероятную последовательность в целом, чем жадный поиск, но медленнее.
    

### Sentence Embeddings

Как получить вектор предложения из BERT?

1. **CLS token:** ❌ Плохо без файн-тюнинга (BERT не обучен на это).
    
2. **Mean Pooling:** ⚠️ Средне. Усреднение всех токенов.
    
3. **SBERT (Siamese BERT):** ✅ **SOTA**. BERT дообучается на парах предложений (SNLI dataset) с сиамской архитектурой, чтобы косинусная близость отражала смысл.​
##  Large Language Models 

### LoRA (Low-Rank Adaptation)

**Как работает:** Вместо обучения всей матрицы весов $W \in \mathbb{R}^{d \times k}$, мы замораживаем ее и обучаем две маленькие матрицы $A$ и $B$, аппроксимирующие обновление весов $\Delta W$.  
Wnew=Wfrozen+A⋅B⏟ΔWWnew=Wfrozen+ΔWA⋅B  
Где $A \in \mathbb{R}^{d \times r}, B \in \mathbb{R}^{r \times k}$, а ранг $r \ll \min(d, k)$.

- **Параметры:**
    
    - **Rank ($r$):** Обычно 8, 16, 32. Меньше = меньше памяти, но меньше "емкость" для новых знаний. Для простых задач (классификация) хватает $r=8$. Для сложных (reasoning, новый язык) нужно $r=64+$.[unsloth](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)​
        
    - **Alpha ($\alpha$):** Скейлинг обновлений. Обычно ставят $\alpha = 2r$ или $\alpha = r$.
        

**Что делать, если не влезает в память даже с LoRA?**

1. **QLoRA (Quantized LoRA):**
    
    - Базовая модель хранится в **4-bit (NF4)**.
        
    - Градиенты пробрасываются через 4-битные веса (dequantization на лету).
        
    - _Экономия:_ 7B модель занимает ~5GB VRAM (вместо 14GB).
        
2. **Gradient Checkpointing:** Жертвуем скоростью ради памяти (не храним активации).
    
3. **Paged Optimizers:** Выгрузка состояний оптимизатора (Adam states) в CPU RAM при пиках нагрузки (библиотека `bitsandbytes`).
    
4. **Offloading:** Выгрузка части слоев модели на CPU (ужасно медленно, но работает).
    

---

### Parameter-Efficient Fine-Tuning (PEFT)

**Польза:**

1. **Память:** Обучаем <1% параметров. Можно тюнить Llama-70B на потребительских GPU.
    
2. **Storage:** Веса адаптера весят 50-200 МБ. Можно хранить одну базовую модель и 100 мелких адаптеров под разные задачи (hot-swapping).
    
3. **Catastrophic Forgetting:** Замороженная база не забывает претрейн-знания.
    

**Сравнение методов тюнинга промптов:**

|Метод|Что обучаем?|Где находится?|Механика|
|---|---|---|---|
|**Prompt Tuning**|Виртуальные токены|Только входной слой|Добавляем обучаемые вектора $P$ к эмбеддингам входа. Остальная сеть заморожена.|
|**P-Tuning**|Виртуальные токены + LSTM|Входной слой (хитро)|Использует небольшую нейросеть (Encoder) для генерации виртуальных токенов, чтобы они были более связными.|
|**Prefix Tuning**|"Виртуальные активации"|**Во всех слоях**|Добавляем обучаемые префиксы к ключам и значениям ($K, V$) в _каждом_слое Attention. Самый мощный метод из трех [aclanthology](https://aclanthology.org/2021.acl-long.353.pdf)​.|

---

### Scaling Laws (Законы масштабирования)

**Kaplan et al. (2020):** "Просто добавь параметры". Считалось, что важнее растить размер модели.  
**Chinchilla (Hoffmann et al., 2022):** "Баланс важнее".

- Для оптимального обучения нужно масштабировать параметры ($N$) и данные ($D$) **пропорционально**.
    
- **Золотое правило:** $\approx 20$ токенов на 1 параметр.
    
- _Пример:_ Llama-7B (7 млрд параметров) по Chinchilla должна учиться на 140 млрд токенов. По факту ее учили на 1Т+ (Overtraining), и это дало супер-качество при малом размере (inference-optimal).
    

---

### Этапы обучения LLM

1. **Pre-training:** (Self-Supervised). Учим предсказывать следующий токен на терабайтах текста.
    
    - _Результат:_ База знаний, умеет говорить, но не слушается инструкций.
        
    - _Можно пропустить:_ Если берете готовую open-source базу (Llama, Mistral).
        
2. **SFT (Supervised Fine-Tuning):** (Instruction Tuning). Учим на парах "Инструкция — Идеальный ответ".
    
    - _Результат:_ Модель-ассистент, следует формату.
        
3. **Alignment (RLHF / DPO):** Учим соответствовать человеческим предпочтениям (безопасность, полезность).
    
    - _RLHF:_ Тренируем Reward Model на сравнениях, потом PPO.
        
    - _DPO (Direct Preference Optimization):_ Оптимизируем лосс прямо на парах "Лучше/Хуже" без Reward Model.
        
    - _Можно пропустить:_ Для узких задач (суммаризация, классификация) часто хватает SFT.
        

---

### RAG vs Few-Shot KNN

**RAG (Retrieval-Augmented Generation):**

- **Механизм:** Вопрос $\to$ Поиск в базе (Vector DB) $\to$ Контекст + Вопрос $\to$ LLM.
    
- **Плюс:** Модель "видит" факты, которых не было в трейне.
    
- **Отличие:** RAG подает найденное _в контекст_ промпта.
    

**Few-Shot kNN (kNN-LM):**

- **Механизм:** Интерполяция предсказаний LLM с "памятью".
    
- На шаге $t$ ищем похожие контексты в базе, смотрим, какое слово шло дальше. Смешиваем вероятность от LLM и вероятность от kNN.
    
- **Отличие:** Это изменение _процедуры генерации_ (logits), а не просто промпта. RAG — это промпт-инжиниринг, kNN-LM — архитектурная добавка.
    

---

### KV Cache & Attention optimization

При генерации мы предсказываем токены по одному. Чтобы не пересчитывать Attention для _прошлых_ токенов каждый раз, мы кэшируем ключи ($K$) и значения ($V$).

**Проблема:** Кэш жрет память. Для Llama-70B с длинным контекстом KV-кэш может весить 100GB+.

|Тип|Механика|Экономия памяти|
|---|---|---|
|**MHA (Multi-Head)**|У каждой головы свои $Q, K, V$.|0 (База)|
|**MQA (Multi-Query)**|Все головы делят **один** $K$ и $V$.|$N_{heads}$ раз (очень сильно) [mbrenndoerfer](https://mbrenndoerfer.com/writing/multi-query-attention-memory-efficient-inference)​|
|**GQA (Grouped-Query)**|Компромисс. Группируем головы (напр. по 8), каждая группа делит свои $K, V$.|В 8-16 раз (используется в Llama-2/3) youtube​|

---

### MixTral (MoE - Mixture of Experts)

**Технология:** Вместо одного жирного FFN слоя, у нас есть 8 "экспертов" (маленьких FFN).  
**Router:** Нейросеть-швейцар, которая для _каждого токена_ решает, каким 2 экспертам его отдать.

- **Sparse activation:** В памяти модель огромная (47B), но для одного токена работает только 13B параметров.
    
- **Плюсы:** Быстрый инференс (как у маленькой модели), знания как у большой.
    
- **Минусы:** Жрет VRAM (нужно грузить всех экспертов). Сложно файнтюнить (риск дисбаланса экспертов).[arxiv+1](https://arxiv.org/abs/2401.04088)​
    

---

## LLM Quantization (3)

### . Квантизация в Resource-Constrained Env

**Когда:**

- Edge-устройства (телефоны, Raspberry Pi).
    
- Запуск огромных моделей (70B) на одной GPU.
    

**Выбор параметров:**

- **4-bit (GPTQ/AWQ):** Золотой стандарт для GPU инференса. Потеря качества <1%, память x4 меньше.
    
- **8-bit (LLM.int8()):** Устарело, почти нет выигрыша в скорости, только память.
    
- **GGUF/Llama.cpp (2-6 bit):** Для CPU/Apple Silicon. Лучший баланс — **Q4_K_M** (4 бита, умная группировка).
    
- **Правило:** Лучше взять модель побольше (13B) и сильно сжать (4-bit), чем маленькую (7B) в fp16.
    

### . QAT vs PTQ

- **PTQ (Post-Training Quantization):**
    
    - Берем готовую модель $\to$ калибруем на 100 примерах (ищем min/max активаций) $\to$ округляем веса.
        
    - _Быстро, но теряет точность на низких битах (<4)._
        
- **QAT (Quantization-Aware Training):**
    
    - Дообучаем модель, симулируя ошибки округления прямо в Forward Pass (fake quantization). Модель учится адаптироваться к "шуму" квантизации.
        
    - _Долго, но дает SOTA качество для экстремального сжатия (2-3 бита)._
        

### EDGE Inference

**Разница:** Вычисления происходят на устройстве пользователя, а не на сервере.  
**Плюсы:**

1. **Privacy:** Данные не покидают устройство.
    
2. **Latency:** Нет сетевых задержек.
    
3. **Cost:** 0 расходов на облачные GPU для компании.  
    **Минусы:** Ограниченная батарея, память и тепловыделение. Сложность поддержки зоопарка железа (Android/iOS/NPU).[zyphra](https://www.zyphra.com/post/edge-llms-benefits-challenges-and-solutions)​
    

---

## Analyze questins (Дополнительные)

**1. Тренды NLP:**

- **Reasoning Models:** (DeepSeek-R1, OpenAI o1). Модели, которые "думают" (CoT) перед ответом.
    
- **Small Language Models (SLM):** Llama-3-8B, Gemma-2-9B. Качество GPT-3.5 на телефоне.
    
- **Context Window:** 1M+ токенов (Gemini), RAG становится не нужен для документов среднего размера.
    

**2. Как отучить генерировать "плохое"?**

1. **System Prompt:** "Ты полезный ассистент, не груби". (Слабо).
    
2. **Negative Constraints:** Штраф логитов для запрещенных слов (Logit Bias).
    
3. **SFT:** Файн-тюнинг на "хороших" примерах отказа ("Я не могу ответить на этот вопрос...").
    
4. **RLHF/DPO:** Основной метод. Наказываем модель за токсичность через Preference Optimization.
    

**3. Классификация k классов по 10 примеров:**

1. **Few-Shot Prompting:** LLM + примеры в контекст.
    
2. **SetFit:** (Sentence Transformer Fine-tuning). Генерируем пары (похож/не похож), учим эмбеддинг, потом LogReg голову. Работает идеально на малых данных.
    
3. **Synthetic Data:** Просим GPT-4 сгенерировать 100 примеров на каждый класс $\to$ обучаем BERT/DistilBERT.
    

**4. 90% уверенности, но ошибка?**  
Да. Это **Calibaration Error**. Современные LLM (после RLHF) часто переуверены (Overconfident). RLHF толкает вероятности к краям (0 или 1), убивая калибровку.

**5. FP16 vs BF16 vs Mix Precision:**

- **FP16:** Маленький диапазон ($6 \cdot 10^{-5} ... 6 \cdot 10^4$). Нужен **Loss Scaling**, чтобы градиенты не обнулились (underflow).
    
- **BF16 (Brain Float):** Тот же диапазон, что у FP32 (много бит на экспоненту), но меньше точность. **Loss Scaling не нужен**. Стандарт для A100/H100.
    
- **Mixed Precision:** Храним веса в FP32 (Master weights), считаем в FP16/BF16.
