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
# 1. ЗАГРУЗКА ДАННЫХ MOVIELENS 32M
# ============================================================

def download_movielens():
    """Скачивает и распаковывает MovieLens 32M, если его нет"""
    url = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"
    zip_path = "ml-32m.zip"
    extract_path = "ml-32m"
    
    # Проверяем, не распакован ли уже
    if os.path.exists(extract_path) and os.path.exists(os.path.join(extract_path, "ratings.csv")):
        print(f"Датасет уже распакован в {extract_path}")
        return extract_path
    
    # Проверяем, не скачан ли zip
    if os.path.exists(zip_path):
        print(f"Файл {zip_path} уже скачан, распаковываю...")
    else:
        print(f"Скачивание {url}... (около 950 МБ, может занять 10-20 минут)")
        print("Если скачивание прервётся, запустите код ещё раз - продолжится с того же места")
        try:
            urllib.request.urlretrieve(url, zip_path)
            print("Скачивание завершено!")
        except Exception as e:
            print(f"Ошибка скачивания: {e}")
            print("Попробуйте скачать датасет вручную с сайта https://grouplens.org/datasets/movielens/32m/")
            print("и распаковать архив в папку ml-32m")
            sys.exit(1)
    
    print("Распаковка...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        print("Распаковка завершена!")
    except Exception as e:
        print(f"Ошибка распаковки: {e}")
        sys.exit(1)
    
    return "ml-32m"

def load_ratings(data_path):
    """Загружает рейтинги из CSV"""
    ratings_file = os.path.join(data_path, "ratings.csv")
    
    if not os.path.exists(ratings_file):
        print(f"Файл {ratings_file} не найден!")
        sys.exit(1)
    
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
    users = df['userId'].values - 1
    items = df['movieId'].values - 1
    ratings = df['rating'].values
    
    num_users = df['userId'].max()
    num_items = df['movieId'].max()
    
    print(f"\nРазмер матрицы: {num_users} × {num_items}")
    print(f"Ненулевых элементов: {len(df)}")
    print(f"Плотность: {len(df) / (num_users * num_items) * 100:.4f}%")
    
    print("\nПостроение COO...")
    start = time.perf_counter()
    mat_coo = coo_matrix((ratings, (users, items)), shape=(num_users, num_items))
    time_coo_build = time.perf_counter() - start
    print(f"  Готово за {time_coo_build:.2f} сек")
    
    print("Конвертация COO → CSR...")
    start = time.perf_counter()
    mat_csr = mat_coo.tocsr()
    time_csr_convert = time.perf_counter() - start
    print(f"  Готово за {time_csr_convert:.2f} сек")
    
    print("Конвертация COO → CSC...")
    start = time.perf_counter()
    mat_csc = mat_coo.tocsc()
    time_csc_convert = time.perf_counter() - start
    print(f"  Готово за {time_csc_convert:.2f} сек")
    
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
    avg_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000
    print(f"  {name}: {avg_time:.2f} ± {std_time:.2f} мс")
    return avg_time

# ============================================================
# 5. ОСНОВНОЙ ЭКСПЕРИМЕНТ
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
    mem_coo = get_memory_mb(mat_coo)
    mem_csr = get_memory_mb(mat_csr)
    mem_csc = get_memory_mb(mat_csc)
    print(f"COO: {mem_coo:.2f} МБ")
    print(f"CSR: {mem_csr:.2f} МБ")
    print(f"CSC: {mem_csc:.2f} МБ")
    
    # 4. Время построения
    print("\n" + "=" * 40)
    print("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА 2: ВРЕМЯ ПОСТРОЕНИЯ")
    print("=" * 40)
    print(f"COO (прямое построение): {build_times[0]:.2f} сек")
    print(f"CSR (конвертация из COO): {build_times[1]:.2f} сек")
    print(f"CSC (конвертация из COO): {build_times[2]:.2f} сек")
    
    # 5. Производительность SpMV
    print("\n" + "=" * 40)
    print("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА 3: УМНОЖЕНИЕ НА ВЕКТОР (SpMV)")
    print("=" * 40)
    t_coo = benchmark_spmv(mat_coo, "COO")
    t_csr = benchmark_spmv(mat_csr, "CSR")
    t_csc = benchmark_spmv(mat_csc, "CSC")
    
    # 6. Итоговый вывод
    print("\n" + "=" * 40)
    print("ВЫВОДЫ")
    print("=" * 40)
    
    # Сравнение с плотной матрицей
    dense_memory_gb = mat_csr.shape[0] * mat_csr.shape[1] * 8 / (1024**3)
    csr_memory_gb = mem_csr / 1024
    print(f"Плотная матрица заняла бы: {dense_memory_gb:.1f} ГБ")
    print(f"CSR занимает: {csr_memory_gb:.2f} ГБ")
    print(f"Экономия памяти: {dense_memory_gb / csr_memory_gb:.0f}x")
    
    print(f"\nCSR быстрее COO в {t_coo / t_csr:.1f} раз")
    print(f"CSR быстрее CSC в {t_csc / t_csr:.1f} раз")
    print(f"CSR экономит память: {mem_coo / mem_csr:.1f}x по сравнению с COO")
    
    print("\n" + "=" * 40)
    print("ИТОГОВАЯ РЕКОМЕНДАЦИЯ")
    print("=" * 40)
    print("Для задач машинного обучения и статистики")
    print("рекомендуется использовать формат CSR.")
    
    input("\nНажмите Enter, чтобы закрыть окно...")

if __name__ == "__main__":
    main()
