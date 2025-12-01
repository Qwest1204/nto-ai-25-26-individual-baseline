from . import config, constants
from .data_processing import load_and_merge_data
from .features import create_features


def prepare_data() -> None:
    print("Запуск подготовки данных")

    # грузим и мержим
    merged_df, book_genres_df, _, descriptions_df = load_and_merge_data()

    # делаем все фичи кроме агрегаций
    df = create_features(merged_df, book_genres_df, descriptions_df, include_aggregates=False)

    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME

    print(f"Сохраняем в {out_path}")
    df.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")

    train_cnt = len(df[df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN])
    test_cnt = len(df[df[constants.COL_SOURCE] == constants.VAL_SOURCE_TEST])

    print("Готово!")
    print(f"Трейн: {train_cnt:,} | Тест: {test_cnt:,} | Фичей: {df.shape[1]}")
    print(f"Файл: {out_path}")


if __name__ == "__main__":
    prepare_data()
