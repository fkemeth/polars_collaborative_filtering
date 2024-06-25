import polars as pl
import multiprocessing as mp

from itertools import cycle

from sklearn.model_selection import KFold, train_test_split

from .collaborative_filtering import CollaborativeFilter


class Evaluator:
    def evaluate(ratings):
        cf = CollaborativeFilter(
        correlation_method = "pearson",
        similarity_threshold = 0.7,
        minimal_number_of_ratings = 6,
        minimum_number_of_books_rated_in_common = 10,
        neighborhood_method = "threshold",)
        user_ids = ratings.select("user_id").unique().sort("user_id").collect()
        kf = KFold(n_splits=len(user_ids), random_state=42)
        for i in range(kf.get_n_splits()):
            train_index, test_index = next(kf.split(user_ids))
            input_ratings, heldout_ratings = train_test_split(test_ratings, test_size=1)
            predictions = cf.predict(ratings.filter(pl.col("user_id").is_in(train_index)), 
                                     ratings.filter(pl.col("user_id").is_in(test_index))



    def __init__(self, ratings: pl.DataFrame, params: dict) -> None:
        self.params = params

        self.ratings = ratings
        self.user_ids = ratings.user_id.unique()
        self.num_users = len(self.user_ids)

        self.kf = KFold(n_splits=self.num_users, shuffle=True, random_state=42)

        self.metrics = {}
        self.metrics["coverage"] = []
        self.metrics["mae"] = []

    @staticmethod
    def log_results(metrics: dict, coverage: float, mae: float) -> None:
        metrics["coverage"].append(coverage)
        metrics["mae"].append(mae)

    def evaluate(self, train_index, test_index) -> None:
        train_user_ids, test_user_ids = (
            self.user_ids[train_index],
            self.user_ids[test_index],
        )
        train_ratings, test_ratings = (
            self.ratings[self.ratings.user_id.isin(train_user_ids)],
            self.ratings[self.ratings.user_id.isin(test_user_ids)],
        )

        input_ratings, heldout_ratings = train_test_split(test_ratings, stratify=test_ratings.user_id, test_size=1)

        cf = CollaborativeFilter(
            train_ratings,
            user_col="user_id",
            item_col="item_id",
            neighborhood_method=self.params["neighborhood_method"],
            correlation_method=self.params["correlation_method"],
            minimal_similarity=self.params["minimal_similarity"],
            number_of_neighbors=self.params["number_of_neighbors"],
            minimum_number_of_items_rated_in_common=self.params[
                "minimum_number_of_items_rated_in_common"
            ],
            minimal_number_of_ratings=self.params["minimal_number_of_ratings"],
            deviation_from_mean=self.params["deviation_from_mean"],
        )

        similarities = cf.get_similarities(input_ratings)
        predicted_scores = cf.get_scores(similarities, input_ratings)

        predictions = heldout_ratings.merge(
            predicted_scores.rename("scores"), on="item_id", how="left"
        )
        coverage = 1 - predictions.scores.isna().sum() / len(predictions)
        mae = (predictions.rating - predictions.scores).abs().mean()
        Evaluator.log_results(self.metrics, coverage, mae)
        return coverage, mae

    def run(self, number_of_runs: int = 1) -> None:
        for i, (train_index, test_index) in enumerate(cycle(list(self.kf.split(self.user_ids)))):
            self.evaluate(train_index=train_index, test_index=test_index)

            if i == number_of_runs - 1:
                return True

    def run_parallel(self, number_of_runs: int = 1) -> None:
        metrics = {"coverage": [], "mae": []}
        pool = mp.Pool(mp.cpu_count() - 4)
        for i, (train_index, test_index) in enumerate(cycle(list(self.kf.split(self.user_ids)))):
            pool.apply_async(
                self.evaluate,
                args=(train_index, test_index),
                callback=lambda x: Evaluator.log_results(metrics, *x),
            )

            if i == number_of_runs - 1:
                break
        pool.close()
        pool.join()
        self.metrics = metrics
