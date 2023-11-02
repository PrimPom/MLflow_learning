
import warnings
from urllib.parse import urlparse

import keras_tuner

import mlflow.sklearn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from model.model import Model


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    X_t = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=int)
    y_t = np.array([0, 0, 0, 1], dtype=int)

    X_v = np.array([[1, 0], [1, 1]], dtype=int)
    y_v = np.array([0, 1], dtype=int)

    with mlflow.start_run():
        model = Model.create_basic_model(keras_tuner.HyperParameters())

        hp = keras_tuner.HyperParameters()
        print(hp.Int("units", min_value=32, max_value=512, step=32))

        # Start the search, RandomSearch, BayesianOptimization, HYperband
        tuner = keras_tuner.RandomSearch(
            hypermodel=Model.create_basic_model,
            objective="val_mae",
            max_trials=5,
            executions_per_trial=1,
            overwrite=True,
            directory="../outputs",
            project_name="firstTuning"
        )
        #Afficher l espace de recherche
        tuner.search_space_summary()
        #lancer la recherche
        tuner.search(X_t, y_t, epochs=10, validation_data=(X_v, y_v))
        # afficher le sommaire de la recherche
        tuner.results_summary()
        # recuperer les deux meilleurs hp
        print("\n Best 2 lrs")
        best_2_lr = tuner.get_best_hyperparameters(5)
        for lr in best_2_lr:
            print(lr.get("lr"))
        print(str(best_2_lr[0].get('lr')))

        model = Model.create_basic_model(best_2_lr[0])
        # model.fit(X_t, y_t, validation_data=(X_v, y_v), epochs=10, verbose=1)

        """
        model.fit(X_t, y_t, epochs=10, verbose=1)
        pred = model.predict(np.array([[1, 1]], dtype=int))[0]
        pred_true = [1]
        (rmse, mae, r2) = eval_metrics(pred_true, pred)
        print(mae)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(model, "model", registered_model_name="Basic")
        else:
            mlflow.sklearn.log_model(model, "model")
            
        """
