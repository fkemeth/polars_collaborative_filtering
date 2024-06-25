import polars as pl


class CollaborativeFilter:
    def __init__(
        self,
        correlation_method: str = "pearson",
        similarity_threshold: float | int = 0.7,
        minimal_number_of_ratings: int = 5,
        minimum_number_of_books_rated_in_common: int = 10,
        neighborhood_method: str = "threshold",
    ) -> None:
        self.correlation_method = correlation_method
        self.similarity_threshold = similarity_threshold
        self.minimal_number_of_ratings = minimal_number_of_ratings
        self.minimum_number_of_books_rated_in_common = (
            minimum_number_of_books_rated_in_common
        )
        self.neighborhood_method = neighborhood_method

    @staticmethod
    def filter_on_minimum_number_of_books_rated_in_common(
        ratings: pl.DataFrame, minimum_number_of_books_rated_in_common: int
    ) -> pl.DataFrame:
        # Calculate the number of articles rated in common for each user
        articles_rated_in_common = ratings.group_by("user_id").agg(
            (pl.col("rating_user").is_not_null())
            .sum()
            .alias("articles_rated_in_common")
        )

        # Filter out users who have rated less than the minimum number of books in common
        ratings = ratings.join(
            articles_rated_in_common, how="left", on="user_id"
        ).filter(
            pl.col("articles_rated_in_common")
            >= minimum_number_of_books_rated_in_common
        )
        return ratings.drop("articles_rated_in_common")

    @staticmethod
    def filter_on_minimum_number_of_ratings(
        ratings: pl.DataFrame, minimal_number_of_ratings
    ) -> pl.DataFrame:
        # Calculate the number of ratings per article
        ratings_per_article = ratings.group_by("item_id").agg(
            (pl.col("rating").is_not_null()).sum().alias("ratings_per_article")
        )

        # Filter out articles that have less than the minimum number of ratings
        ratings = ratings.join(ratings_per_article, on="item_id", how="left")
        ratings = ratings.filter(
            pl.col("ratings_per_article") >= minimal_number_of_ratings
        )
        return ratings.drop("ratings_per_article")

    @staticmethod
    def calculate_similarity(
        ratings: pl.DataFrame, correlation_method: str
    ) -> pl.DataFrame:
        # Calculate the similarity between users based on their ratings
        similarities = ratings.group_by("user_id", maintain_order=True).agg(
            pl.corr("rating", "rating_user", method=correlation_method).alias("corr")
        )

        # Filter out users whose similarity is below the minimum similarity threshold
        ratings = ratings.join(similarities, on="user_id", how="left")
        return ratings

    @staticmethod
    def select_neighborhood(
        ratings: pl.DataFrame, threshold: float | int, neighborhood_method: str
    ) -> pl.DataFrame:
        if neighborhood_method == "threshold":
            ratings = ratings.filter(pl.col("corr") > threshold)
        elif neighborhood_method == "number":
            ratings = ratings.sort(by="corr", reverse=True).limit(threshold)
        else:
            raise ValueError(
                "Only 'threshold' and 'number' neighborhood methods are supported"
            )
        ratings = ratings.filter(pl.col("corr").is_not_nan())
        return ratings

    def predict(
        self,
        ratings: pl.DataFrame | pl.LazyFrame,
        user_ratings: pl.DataFrame | pl.LazyFrame,
        num_predictions: int = 10,
    ):
        # Join the ratings dataset with the user's own ratings
        ratings = ratings.join(
            user_ratings.select("item_id", "rating"),
            how="left",
            on="item_id",
            suffix="_user",
        )

        if self.minimum_number_of_books_rated_in_common:
            ratings = (
                CollaborativeFilter.filter_on_minimum_number_of_books_rated_in_common(
                    ratings, self.minimum_number_of_books_rated_in_common
                )
            )

        # Calculate the similarity between users
        ratings = CollaborativeFilter.calculate_similarity(
            ratings, self.correlation_method
        )
        ratings = CollaborativeFilter.select_neighborhood(
            ratings, self.similarity_threshold, self.neighborhood_method
        )

        if self.minimal_number_of_ratings:
            ratings = CollaborativeFilter.filter_on_minimum_number_of_ratings(
                ratings, self.minimal_number_of_ratings
            )

        # Define the prediction function
        def predict_func():
            return ((pl.col("rating") * pl.col("corr")).sum()) / (pl.col("corr").sum())

        # Calculate the predicted ratings for each book
        predictions = (
            ratings.group_by("item_id", maintain_order=True)
            .agg(predict_func().alias("prediction"))
            .select("item_id", "prediction")
        )

        # Return the top recommendations
        if num_predictions:
            predictions = predictions.sort(by="prediction", descending=True).limit(
                num_predictions
            )
        return predictions
