#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Экспериментальное сравнение форматов разреженных матриц на датасете MovieLens 32M
Автор: Стрелков Владимир
Курсовая работа: Разреженные матрицы в статистике и машинном обучении
"""

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
import time
import os
import urllib.request
import zipfile
import sys

# ============================================================
# 1. ЗАГРУЗКА ДАННЫХ MOVIELENS 32M (автоматическая)
# ============================================================

def download_movielens():
    """Скачивает и распаковывает MovieLens 32M, если его нет"""
    url = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"
    zip_path = "ml-32m.zip"
    extract_path = "ml-32m"
    
    if os.path.exists(extract_path):
        print(f"Данные уже есть в {extract_path}")
        return extract_path
    
    print(f"Скачивание {url}... (около 950 МБ, может занять несколько минут)")
    urllib.request.urlretrieve(url, zip_path)
    
    print("Распаковка...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")
    
    os.remove(zip_path)
    print("Готово!")
    return extract_path

def load_ratings(data_path):
    """Загружает рейтинги из CSV"""
    ratings_file = os.path.join(data_path, "ml-32m", "ratings.csv")
    if not os.path.exists(ratings_file):
        ratings_file = os.path.join(data_path, "ratings.csv")
    
    print(f"Загрузка {ratings_file}...")
    df = pd.read_csv(ratings_file)
    print(f"Загружено {len(df)} рейтингов")
    print(f"Пользователей: {df['userId'].nunique()}")
    print(f"Фильмов: {df['movieId'].nunique()}")
    return df

# ============================================================
# 2. ПОСТРОЕНИЕ РАЗРЕЖЕННОЙ МАТРИЦЫ В РАЗНЫХ ФОРМАТАХ
# ============================================================

def build_sparse_matrices(df):
    """Строит матрицу рейтингов в форматах COO, CSR, CSC"""
    users = df['userId'].values - 1  # 0-based индексация
    items = df['movieId'].values - 1
    ratings = df['rating'].values
    
    num_users = df['userId'].max()
    num_items = df['movieId'].max()
    
    print(f"\nРазмер матрицы: {num_users} × {num_items}")
    print(f"Ненулевых элементов: {len(df)}")
    print(f"Плотность: {len(df) / (num_users * num_items) * 100:.4f}%")
    
    # Построение в COO (самый простой для создания)
    print("\nПостроение COO...")
    start = time.perf_counter()
    mat_coo = coo_matrix((ratings, (users, items)), shape=(num_users, num_items))
    time_coo_build = time.perf_counter() - start
    
    # Конвертация в CSR
    print("Конвертация COO → CSR...")
    start = time.perf_counter()
    mat_csr = mat_coo.tocsr()
    time_csr_convert = time.perf_counter() - start
    
    # Конвертация в CSC
    print("Конвертация COO → CSC...")
    start = time.perf_counter()
    mat_csc = mat_coo.tocsc()
    time_csc_convert = time.perf_counter() - start
    
    return mat_coo, mat_csr, mat_csc, (time_coo_build, time_csr_convert, time_csc_convert)

# ============================================================
# 3. ИЗМЕРЕНИЕ ПАМЯТИ
# ============================================================

def get_memory_mb(mat):
    """Возвращает объём памяти в МБ для разреженной матрицы"""
    mem = mat.data.nbytes
    if hasattr(mat, 'indices'):
        mem += mat.indices.nbytes
    if hasattr(mat, 'indptr'):
        mem += mat.indptr.nbytes
    if hasattr(mat, 'row'):
        mem += mat.row.nbytes
    if hasattr(mat, 'col'):
        mem += mat.col.nbytes
    return mem / (1024 ** 2)

# ============================================================
# 4. УМНОЖЕНИЕ МАТРИЦЫ НА ВЕКТОР (SpMV)
# ============================================================

def benchmark_spmv(mat, name, n_trials=10):
    """Измеряет время умножения матрицы на вектор"""
    vec = np.random.rand(mat.shape[1])
    times = []
    for _ in range(n_trials):
        start = time.perf_counter()
        result = mat @ vec
        times.append(time.perf_counter() - start)
    avg_time = np.mean(times) * 1000  # в миллисекундах
    std_time = np.std(times) * 1000
    print(f"  {name}: {avg_time:.2f} ± {std_time:.2f} мс")
    return avg_time

# ============================================================
# 5. ПРАКТИЧЕСКАЯ ЗАДАЧА: ПОИСК ПОХОЖИХ ПОЛЬЗОВАТЕЛЕЙ
# ============================================================

def find_similar_users(mat_csr, user_id, k=10):
    """
    Находит k пользователей, наиболее похожих на заданного
    на основе косинусного сходства их рейтингов
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    user_idx = user_id - 1
    user_vector = mat_csr[user_idx, :]
    
    # Вычисляем сходство со всеми пользователями
    start = time.perf_counter()
    similarities = cosine_similarity(user_vector, mat_csr).flatten()
    elapsed = time.perf_counter() - start
    
    # Исключаем самого пользователя и берём топ-k
    similarities[user_idx] = -1
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    top_k_scores = similarities[top_k_indices]
    
    # Переводим индексы обратно в userId (1-based)
    top_k_users = [idx + 1 for idx in top_k_indices]
    
    return top_k_users, top_k_scores, elapsed

# ============================================================
# 6. ОСНОВНОЙ ЭКСПЕРИМЕНТ
# ============================================================

def main():
    print("=" * 60)
    print("ЭКСПЕРИМЕНТАЛЬНОЕ СРАВНЕНИЕ ФОРМАТОВ РАЗРЕЖЕННЫХ МАТРИЦ")
    print("Датасет: MovieLens 32M")
    print("=" * 60)
    
    # 1. Загрузка данных
    data_dir = download_movielens()
    df = load_ratings(data_dir)
    
    # 2. Построение матриц
    mat_coo, mat_csr, mat_csc, build_times = build_sparse_matrices(df)
    
    # 3. Измерение памяти
    print("\n" + "=" * 40)
    print("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА 1: ПАМЯТЬ")
    print("=" * 40)
    print(f"COO: {get_memory_mb(mat_coo):.2f} МБ")
    print(f"CSR: {get_memory_mb(mat_csr):.2f} МБ")
    print(f"CSC: {get_memory_mb(mat_csc):.2f} МБ")
    
    # 4. Время загрузки/построения
    print("\n" + "=" * 40)
    print("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА 2: ВРЕМЯ ПОСТРОЕНИЯ")
    print("=" * 40)
    print(f"COO (прямое построение): {build_times[0]:.2f} сек")
    print(f"CSR (конвертация из COO): {build_times[1]:.2f} сек")
    print(f"CSC (конвертация из COO): {build_times[2]:.2f} сек")
    
    # 5. Производительность SpMV
    print("\n" + "=" * 40)
    print("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА 3: УМНОЖЕНИЕ МАТРИЦЫ НА ВЕКТОР (SpMV)")
    print("=" * 40)
    t_coo = benchmark_spmv(mat_coo, "COO")
    t_csr = benchmark_spmv(mat_csr, "CSR")
    t_csc = benchmark_spmv(mat_csc, "CSC")
    
    # 6. Практическая задача: поиск похожих пользователей
    print("\n" + "=" * 40)
    print("ПРАКТИЧЕСКАЯ ЗАДАЧА: ПОИСК ПОХОЖИХ ПОЛЬЗОВАТЕЛЕЙ")
    print("=" * 40)
    test_user_id = 1
    top_users, scores, time_taken = find_similar_users(mat_csr, test_user_id, k=5)
    print(f"Для пользователя {test_user_id} наиболее похожие пользователи:")
    for i, (uid, score) in enumerate(zip(top_users, scores), 1):
        print(f"  {i}. user_id={uid}, косинусное сходство={score:.4f}")
    print(f"Время выполнения запроса: {time_taken*1000:.2f} мс")
    
    # 7. Итоговый вывод
    print("\n" + "=" * 40)
    print("ОБЩИЕ ВЫВОДЫ")
    print("=" * 40)
    print(f"1. Экономия памяти CSR vs плотный аналог:")
    dense_memory = mat_csr.shape[0] * mat_csr.shape[1] * 8 / (1024**3)
    csr_memory = get_memory_mb(mat_csr) / 1024
    print(f"   Плотная матрица заняла бы ≈ {dense_memory:.1f} ГБ")
    print(f"   CSR занимает {csr_memory:.2f} ГБ")
    print(f"   Экономия в {dense_memory / csr_memory:.0f} раз")
    
    print(f"\n2. Ускорение SpMV:")
    print(f"   CSR быстрее COO в {t_coo / t_csr:.1f} раз")
    print(f"   CSR быстрее CSC в {t_csc / t_csr:.1f} раз")
    
    print("\n3. Рекомендация:")
    print("   - Для построения матриц из сырых данных удобен COO")
    print("   - Для вычислений (SpMV, ML-алгоритмы) предпочтителен CSR")
    print("   - CSC эффективен при работе по столбцам")

if __name__ == "__main__":
    main()
