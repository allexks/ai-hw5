"""62136"""

import pandas as pd

class NaiveBayesClassifier:
    def __init__(self, model: pd.DataFrame):
        self.model = model

    def predict(self, features_vector):
        """https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Constructing_a_classifier_from_the_probability_model"""
        prob_k = {}
        for k in ("democrat", "republican"):
            Ck_row_inds = [
                ind
                for ind in range(len(self.model))
                if self.model["class"].values[ind] == k
            ]
            pCk = len(Ck_row_inds) / len(self.model)
            features_p_product = 1.0
            for i, f_i in enumerate(features_vector):
                if f_i == "?":
                    features_p_product *= 1/2
                else:
                    relevant_features = [self.model[str(i)].values[ind] for ind in Ck_row_inds]
                    features_p_product *= len(list(filter(lambda f: f == f_i, relevant_features))) / len(relevant_features)
            prob_k[k] = pCk * features_p_product
        result = "democrat" if prob_k["democrat"] > prob_k["republican"] else "republican"
        return result

if __name__ == "__main__":
    df = pd.read_csv("data/house-votes-84.data")
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle
    df.columns = ["class"] + list(map(str, range(16)))

    # Perform 10-fold cross validation
    accuracies = []
    for start_index in range(0, 430, 43):
        end_index = start_index + 42
        print(f"Testing with {start_index}:{end_index}...")

        learning_df = df[:start_index].append(df[end_index+1:])
        classifier = NaiveBayesClassifier(learning_df)

        correct = 0
        tested_count = 0
        for test_vector_index in range(start_index, end_index+1):
            test_vector = df.loc[test_vector_index]
            prediction = classifier.predict(test_vector[1:])
            if test_vector[0] == prediction:
                correct += 1
            tested_count += 1
        accuracies.append(correct / tested_count)
        print(f"Accuracy: {accuracies[-1]:.2%}\n")

    avg = sum(accuracies) / len(accuracies)
    print(f"Overall Accuracy by 10-fold validation: {avg:.2%}")
