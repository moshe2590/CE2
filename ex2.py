import abc
from datetime import date, datetime
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm  # TODO: To_remove
import time  # TODO: To_remove

NO_TIMESTAMP = 0
NUM_NEIGHBOURS = 3


class Recommender(abc.ABC):
    def __init__(self, ratings: pd.DataFrame):
        self.initialize_predictor(ratings)

    @abc.abstractmethod
    def initialize_predictor(self, ratings: pd.DataFrame):
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        raise NotImplementedError()

    def rmse(self, true_ratings) -> float:
        """
        :param true_ratings: DataFrame of the real ratings
        :return: RMSE score
        """
        pass


class BaselineRecommender(Recommender):
    def initialize_predictor(self, ratings: pd.DataFrame):
        self.b_u, self.b_i = {}, {}

        for user in range(int(max(ratings["user"])) + 1):
            self.b_u[user] = np.mean(ratings["rating"][ratings["user"] == user])
        for item in range(int(max(ratings["item"])) + 1):
            self.b_i[item] = np.mean(ratings["rating"][ratings["item"] == item])

        self.r_avg = np.mean(ratings["rating"])

    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        r_hat = self.b_u[user] + self.b_i[item] - self.r_avg
        if r_hat < 0.5:
            return 0.5
        if r_hat > 5:
            return 5
        return r_hat

    def rmse(self, true_ratings) -> float:
        """
        :param true_ratings: DataFrame of the real ratings
        :return: RMSE score
        """
        r = true_ratings["rating"].to_numpy()
        r_hat = [self.predict(row["user"], row["item"], NO_TIMESTAMP) for index, row in true_ratings.iterrows()]
        return np.sqrt(mean_squared_error(r, r_hat))


class NeighborhoodRecommender(Recommender):
    def initialize_predictor(self, ratings: pd.DataFrame):
        start_time = time.time()
        self.b_u, self.b_i = {}, {}
        self.R_dict = {}
        self.norm = {}
        self.r_avg = np.mean(ratings["rating"])
        self.movies_of_user = {}
        self.users_of_movie = {}

        # movies of each user
        for index, row in ratings.iterrows():
            self.R_dict[(row["user"], row["item"])] = row["rating"]
            if row["user"] in self.movies_of_user:
                self.movies_of_user[row["user"]].append(row["item"])
            else:
                self.movies_of_user[row["user"]] = [row["item"]]
            if row["item"] in self.users_of_movie:
                self.users_of_movie[row["item"]].append(int(row["user"]))
            else:
                self.users_of_movie[row["item"]] = [int(row["user"])]

        # compute b_i,b_u
        for user in range(int(max(ratings["user"])) + 1):
            self.b_u[user] = np.mean(ratings["rating"][ratings["user"] == user])
        for item in range(int(max(ratings["item"])) + 1):
            self.b_i[item] = np.mean(ratings["rating"][ratings["item"] == item])

        self.num_users, self.num_items = len(self.b_u), len(self.b_i)

        for obj in self.R_dict.keys():
            self.R_dict[obj] -= self.r_avg

        # compute correlations
        self.corr = np.zeros((self.num_users, self.num_users))
        for user1 in range(self.num_users):
            for user2 in range(user1 + 1, self.num_users):
                up, down1, down2 = 0, 0, 0
                common_movies = list(set(self.movies_of_user[user1]).intersection(self.movies_of_user[user2]))
                for movie in common_movies:
                    up += self.R_dict[(user1, movie)] * self.R_dict[(user2, movie)]
                    down1 += self.R_dict[(user1, movie)] ** 2
                    down2 += self.R_dict[(user2, movie)] ** 2
                corr = up / np.sqrt(down1 * down2)
                if down1 != 0 and down2 != 0:
                    self.corr[user1][user2] = corr
                    self.corr[user2][user1] = corr
        print(f"Time of initialize_predictor is : {time.time() - start_time} ")

    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        # start_time = time.time()

        top3 = []
        up, down = 0, 0
        optional_neighbours = self.users_of_movie[item]
        correlations = self.corr[user]
        # option 1
        """
        indices = correlations.argsort()[::-1][1:]
        counter = 0
        for ind in indices:
            if ind in optional_neighbours:
                top3.append(ind)
                counter += 1
            if counter == 3:
                break
        """
        # option 2

        indices = correlations[optional_neighbours].argsort()[-4:-1]
        top3 = [optional_neighbours[indices[i]] for i in range(len(indices))]

        for neigh in top3:
            up += correlations[neigh] * (self.R_dict[(neigh, item)])
            down += np.abs(correlations[neigh])
        corr_part = up / down if down != 0 else 0
        r_hat = self.b_u[user] + self.b_i[item] - self.r_avg + corr_part
        # print(f"Time of predict is : {time.time() - start_time} ")
        # print(f"r_hat : {r_hat} ")

        if r_hat < 0.5:
            return 0.5
        if r_hat > 5:
            return 5
        return r_hat

    def user_similarity(self, user1: int, user2: int) -> float:
        """
                :param user1: User identifier
                :param user2: User identifier
                :return: The correlation of the two users (between -1 and 1)
                """
        return self.corr[user1][user2]

    def rmse(self, true_ratings) -> float:
        """
        :param true_ratings: DataFrame of the real ratings
        :return: RMSE score
        """
        r = true_ratings["rating"].to_numpy()
        r_hat = [self.predict(int(row["user"]), int(row["item"]), NO_TIMESTAMP) for index, row in
                 true_ratings.iterrows()]
        return np.sqrt(np.sum((r - r_hat) ** 2) / len(r))


def convert_ts(ts):
    b_d, b_n, b_w = 0, 0, 0
    obj = datetime.fromtimestamp(ts)
    weekday = date.weekday(obj)
    if 4 <= weekday <= 5:
        b_w = 1
    if 6 <= obj.hour <= 17:
        b_d, b_n = 1, 0
    else:
        b_d, b_n = 0, 1
    return b_d, b_n, b_w


class LSRecommender(Recommender):
    def initialize_predictor(self, ratings: pd.DataFrame):
        self.num_users, self.num_movies = int(max(ratings["user"])) + 1, int(max(ratings["item"])) + 1
        self.r_avg = np.mean(ratings["rating"])
        self.X = np.zeros((len(ratings), self.num_users + self.num_movies + 3))
        for index, row in ratings.iterrows():
            user, item, ts = int(row["user"]), int(row["item"]), row["timestamp"]
            self.X[index][user] = 1
            self.X[index][self.num_users + item] = 1
            b_d, b_n, b_w = convert_ts(ts)
            self.X[index][self.num_users + self.num_movies] = b_d  # b_d
            self.X[index][self.num_users + self.num_movies + 1] = b_n  # b_n
            self.X[index][self.num_users + self.num_movies + 2] = b_w  # b_w
        self.y = np.array(ratings["rating"] - self.r_avg)


    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        return self.r_avg + self.beta[user] + self.beta[self.num_users + item] + \
               self.beta[self.num_users + self.num_movies] + self.beta[self.num_users + self.num_movies + 1] + \
               self.beta[self.num_users + self.num_movies + 2]

    def solve_ls(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates and solves the least squares regression
        :return: Tuple of X, b, y such that b is the solution to min ||Xb-y||
        """
        self.beta = np.linalg.lstsq(self.X, self.y, rcond=None)[0]
        return (self.X, self.beta, self.y)

    def rmse(self, true_ratings) -> float:
        """
        :param true_ratings: DataFrame of the real ratings
        :return: RMSE score
        """
        r = true_ratings["rating"].to_numpy()
        r_hat = [self.predict(int(row["user"]), int(row["item"]), NO_TIMESTAMP) for index, row in
                 true_ratings.iterrows()]
        return np.sqrt(np.sum((r - r_hat) ** 2) / len(r))


class CompetitionRecommender(Recommender):
    def initialize_predictor(self, ratings: pd.DataFrame):
        self.b_u, self.b_i = {}, {}
        self.R_dict = {}
        self.norm = {}
        self.r_avg = np.mean(ratings["rating"])
        self.movies_of_user = {}
        self.users_of_movie = {}

        # movies of each user
        for index, row in ratings.iterrows():
            self.R_dict[(row["user"], row["item"])] = row["rating"]
            if row["user"] in self.movies_of_user:
                self.movies_of_user[row["user"]].append(row["item"])
            else:
                self.movies_of_user[row["user"]] = [row["item"]]
            if row["item"] in self.users_of_movie:
                self.users_of_movie[row["item"]].append(int(row["user"]))
            else:
                self.users_of_movie[row["item"]] = [int(row["user"])]

        # compute b_i,b_u
        for user in range(int(max(ratings["user"])) + 1):
            self.b_u[user] = np.mean(ratings["rating"][ratings["user"] == user])
        for item in range(int(max(ratings["item"])) + 1):
            self.b_i[item] = np.mean(ratings["rating"][ratings["item"] == item])

        self.num_users, self.num_items = len(self.b_u), len(self.b_i)

        for obj in self.R_dict.keys():
            self.R_dict[obj] -= self.r_avg

        # compute correlations
        self.corr = np.zeros((self.num_users, self.num_users))
        for user1 in range(self.num_users):
            for user2 in range(user1 + 1, self.num_users):
                up, down1, down2 = 0, 0, 0
                common_movies = list(set(self.movies_of_user[user1]).intersection(self.movies_of_user[user2]))
                for movie in common_movies:
                    up += self.R_dict[(user1, movie)] * self.R_dict[(user2, movie)]
                    down1 += self.R_dict[(user1, movie)] ** 2
                    down2 += self.R_dict[(user2, movie)] ** 2
                corr = up / np.sqrt(down1 * down2)
                if down1 != 0 and down2 != 0:
                    self.corr[user1][user2] = corr
                    self.corr[user2][user1] = corr

        self.num_users, self.num_movies = int(max(ratings["user"])) + 1, int(max(ratings["item"])) + 1
        self.r_avg = np.mean(ratings["rating"])
        self.X = np.zeros((len(ratings), self.num_users + self.num_movies + 3))
        for index, row in ratings.iterrows():
            user, item, ts = int(row["user"]), int(row["item"]), row["timestamp"]


            # find top 3
            optional_neighbours = self.users_of_movie[item]
            correlations = self.corr[user]
            indices = correlations[optional_neighbours].argsort()[-4:-1]
            top3 = [optional_neighbours[indices[i]] for i in range(len(indices))]
            for neigh in top3:
                self.X[index][neigh] = 1

            self.X[index][user] = 1
            self.X[index][self.num_users + item] = 1
            b_d, b_n, b_w = convert_ts(ts)
            self.X[index][self.num_users + self.num_movies] = b_d  # b_d
            self.X[index][self.num_users + self.num_movies + 1] = b_n  # b_n
            self.X[index][self.num_users + self.num_movies + 2] = b_w  # b_w
        self.y = np.array(ratings["rating"] - self.r_avg)
        self.beta = np.linalg.lstsq(self.X, self.y, rcond=None)[0]

    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        return self.r_avg + self.beta[user] + self.beta[self.num_users + item] + \
               self.beta[self.num_users + self.num_movies] + self.beta[self.num_users + self.num_movies + 1] + \
               self.beta[self.num_users + self.num_movies + 2]



    def rmse(self, true_ratings) -> float:
        """
        :param true_ratings: DataFrame of the real ratings
        :return: RMSE score
        """
        r = true_ratings["rating"].to_numpy()
        r_hat = [self.predict(int(row["user"]), int(row["item"]), NO_TIMESTAMP) for index, row in
                 true_ratings.iterrows()]
        return np.sqrt(np.sum((r - r_hat) ** 2) / len(r))


class NeighborhoodRecommender2(Recommender):
    def initialize_predictor(self, ratings: pd.DataFrame):
        self.r_avg = np.mean(ratings["rating"])

        self.b_u, self.b_i = {}, {}
        self.R = pd.pivot_table(ratings, values='rating', index=['user'],
                                columns=['item'], fill_value=0.0)
        ratings["rating"] -= self.r_avg
        self.R_centered = pd.pivot_table(ratings, values='rating', index=['user'],
                                         columns=['item'], fill_value=0.0)

        # compute b_u
        for user in self.R.index.values:
            row = list(self.R.loc[user][:])
            row = [i for i in row if i != 0]
            self.b_u[user] = np.mean(row)

        # compute b_i
        for item in self.R.columns.values:
            col = list(self.R.loc[:][item])
            col = [i for i in col if i != 0]
            self.b_i[item] = np.mean(col)
            # self.b_i[item] = list(self.R.loc[:][item]).mean()

        # cosine similarity + top3 neighbours
        self.corr = cosine_similarity(self.R_centered)
        self.top3 = {}
        for user in self.R.index.values:
            user = int(user)
            indices = self.corr[user].argsort()[-4:-1][::-1]

            self.top3[user] = ((indices[0], self.corr[user][indices[0]]), (indices[1], self.corr[user][indices[1]]),
                               (indices[2], self.corr[user][indices[2]]))
        # R_dict
        self.R_dict = {}
        for index, row in ratings.iterrows():
            self.R_dict[(row["user"], row["item"])] = row["rating"]

        print("b_u : ")
        print(self.b_u)
        print("b_i : ")
        print(self.b_i)
        print("top3 : ")
        print(self.top3)
        print("corr_matrix")
        print(self.corr)

    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        up, down = 0, 0
        for neigh, corr_score in self.top3[user]:
            if (neigh, item) in self.R_dict:
                up += corr_score * (self.R_dict[(neigh, item)] - self.r_avg)
                down += np.abs(corr_score)
            else:
                up += corr_score * (0)
                # down += np.abs(corr_score)
        corr_part = up / down if down else 0
        r_hat = self.b_u[user] + self.b_i[item] - self.r_avg + corr_part
        # print(self.b_u[user], self.b_i[item], self.r_avg, corr_part)
        if r_hat < 0.5:
            return 0.5
        if r_hat > 5:
            return 5
        return r_hat

    def user_similarity(self, user1: int, user2: int) -> float:
        """
                :param user1: User identifier
                :param user2: User identifier
                :return: The correlation of the two users (between -1 and 1)
                """
        return self.corr[user1][user2]

    def rmse(self, true_ratings) -> float:
        """
        :param true_ratings: DataFrame of the real ratings
        :return: RMSE score
        """
        r = true_ratings["rating"].to_numpy()
        r_hat = [self.predict(int(row["user"]), row["item"], NO_TIMESTAMP) for index, row in true_ratings.iterrows()]
        for i in range(len(r)):
            print(f"r = {r[i]}, r_hat = {r_hat[i]}")
        return np.sqrt(mean_squared_error(r, r_hat))
