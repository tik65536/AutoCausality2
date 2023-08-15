import copy
import warnings
from copy import deepcopy
from typing import List, Optional, Union
from collections import defaultdict

import traceback
import pandas as pd
import numpy as np
from sklearn.linear_model import _base
from flaml import tune

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from dowhy import CausalModel
from dowhy.causal_identifier import IdentifiedEstimand
from joblib import Parallel, delayed

from auto_causality.params import SimpleParamService
from auto_causality.scoring import Scorer
from auto_causality.r_score import RScoreWrapper
from auto_causality.utils import clean_config, treatment_is_multivalue
from auto_causality.models.monkey_patches import AutoML
from auto_causality.models.Bayesian import Gaussian
from auto_causality.models.Bayesian import Multinomial
from auto_causality.models.SVM import SVM
from auto_causality.models.GaussianProcess import GPC
from auto_causality.models.DTS import DTC
from auto_causality.models.DTS import ExTC
from auto_causality.models.MLP import MLP
from auto_causality.models.LogisticRegression import LR
from auto_causality.data_utils import CausalityDataset
from auto_causality.models.passthrough import feature_filter


# Patched from sklearn.linear_model._base to adjust rtol and atol values
def _check_precomputed_gram_matrix(
    X, precompute, X_offset, X_scale, rtol=1e-4, atol=1e-2
):
    n_features = X.shape[1]
    f1 = n_features // 2
    f2 = min(f1 + 1, n_features - 1)

    v1 = (X[:, f1] - X_offset[f1]) * X_scale[f1]
    v2 = (X[:, f2] - X_offset[f2]) * X_scale[f2]

    expected = np.dot(v1, v2)
    actual = precompute[f1, f2]

    if not np.isclose(expected, actual, rtol=rtol, atol=atol):
        raise ValueError(
            "Gram matrix passed in via 'precompute' parameter "
            "did not pass validation when a single element was "
            "checked - please check that it was computed "
            f"properly. For element ({f1},{f2}) we computed "
            f"{expected} but the user-supplied value was "
            f"{actual}."
        )


_base._check_precomputed_gram_matrix = _check_precomputed_gram_matrix


class AutoCausality:
    """Performs AutoML to find best econML estimator.
    Optimises hyperparams of component models of each estimator
    and hyperparams of the estimators themselves. Uses the ERUPT
    metric for estimator selection.

    Example:
    ```python

    estimator_list = [".LinearDML","LinearDRLearner","metalearners"]
    auto_causality = AutoCausality(time_budget=10, estimator_list=estimator_list)

    auto_causality.fit(train_df, test_df, treatment, outcome,
    features_W, features_X)

    print(f"Best estimator: {auto_causality.best_estimator}")

    ```
    """

    def __init__(
        self,
        data_df=None,
        metric="energy_distance",
        metrics_to_report=None,
        time_budget=None,
        verbose=3,
        use_ray=False,
        estimator_list="auto",
        train_size=0.8,
        test_size=None,
        num_samples=-1,
        propensity_model="dummy",
        components_task="regression",
        components_verbose=0,
        components_pred_time_limit=10 / 1e6,
        components_njobs=-1,
        components_time_budget=None,
        try_init_configs=True,
        resources_per_trial=None,
        include_experimental_estimators=False,
        store_all_estimators: Optional[bool] = False,
    ):
        """constructor.

        Args:
            data_df (pandas.DataFrame): dataset to perform causal inference on
            metric (str): metric to optimise.
                Defaults to "erupt" for CATE, "energy_distance" for IV
            metrics_to_report (list). additional metrics to compute and report.
                Defaults to ["qini","auc","ate","erupt", "norm_erupt"] for CATE
                or ["energy_distance"] for IV
            time_budget (float): a number of the time budget in seconds. -1 if no limit.
            num_samples (int): max number of iterations.
            verbose (int):  controls verbosity, higher means more messages. range (0,3). Defaults to 0.
            use_ray (bool): use Ray backend (nrequires ray to be installed).
            estimator_list (list): a list of strings for estimator names,
             or "auto" for a recommended subset, "all" for all, or a list of substrings of estimator names
               e.g. ```['dml', 'CausalForest']```
            train_size (float): Fraction of data used for training set. Defaults to 0.5.
            test_fraction (float): Optional size of test dataset. Defaults to None.
            propensity_model (Union[str, Any]): 'dummy' for dummy classifier, 'auto' for AutoML, or an
                sklearn-style classifier
            components_task (str): task for component models. Defaults to "regression".
            components_verbose (int): verbosity of component model HPO. range (0,3). Defaults to 0.
            components_pred_time_limit (float): prediction time limit for component models
            components_njobs (int): number of concurrent jobs for component model optimisation.
                Defaults to -1 (all available cores).
            components_time_budget (float): time budget for HPO of component models in seconds.
                Defaults to overall time budget / 2.
            try_init_configs (bool): try list of good performing estimators before continuing with HPO.
                Defaults to False.
            blacklisted_estimators (list): [optional] list of estimators not to include in fitting
            store_all_estimators (Optional[bool]). store estimator objects for interim trials. Defaults to False
        """
        assert (
            time_budget is not None or components_time_budget is not None
        ), "Either time_budget or components_time_budget must be specified"

        self._settings = {}
        self._settings["tuner"] = {}
        self._settings["tuner"]["time_budget_s"] = time_budget
        self._settings["tuner"]["num_samples"] = num_samples
        self._settings["tuner"]["verbose"] = verbose
        self._settings["tuner"][
            "use_ray"
        ] = use_ray  # requires ray to be installed via pip install flaml[ray]
        self._settings["tuner"]["resources_per_trial"] = (
            resources_per_trial if resources_per_trial is not None else {"cpu": 0.5}
        )
        self._settings["try_init_configs"] = try_init_configs
        self._settings[
            "include_experimental_estimators"
        ] = include_experimental_estimators

        # params for FLAML on component models:
        self._settings["component_models"] = {}
        self._settings["component_models"]["task"] = components_task
        self._settings["component_models"]["verbose"] = components_verbose
        self._settings["component_models"][
            "pred_time_limit"
        ] = components_pred_time_limit
        self._settings["component_models"]["n_jobs"] = components_njobs
        self._settings["component_models"]["time_budget"] = components_time_budget
        self._settings["component_models"]["eval_method"] = "holdout"

        if 0 < train_size < 1:
            component_test_size = 1 - train_size
        else:
            # TODO: convert train_size to fraction based on data size, in fit()
            component_test_size = 0.2
        self._settings["component_models"]["split_ratio"] = component_test_size
        self._settings["train_size"] = train_size
        self._settings["test_size"] = test_size
        self._settings["store_all"] = store_all_estimators
        self._settings["metric"] = metric
        self._settings["metrics_to_report"] = metrics_to_report
        self._settings["propensity_model"] = propensity_model

        self.results = None
        self._best_estimators = defaultdict(lambda: (float("-inf"), None))

        self.original_estimator_list = estimator_list
        self.data_df = data_df or pd.DataFrame()
        self.causal_model = None
        self.identified_estimand = None
        self.problem = None
        # properties that are used to resume fits (warm start)
        self.resume_scores = []
        self.resume_cfg = []

    def get_params(self, deep=False):
        return self._settings.copy()

    def get_estimators(self, deep=False):
        return self.estimator_list.copy()

    def init_propensity_model(self, propensity_model: str):
        # user can choose between flaml and dummy for propensity model.
        if propensity_model == "dummy":
            self.propensity_model = DummyClassifier(strategy="prior")
        elif propensity_model == "auto":
            self.propensity_model = AutoML(
                **{**self._settings["component_models"], "task": "classification"}
            )
        elif propensity_model == "xgb":
            self.propensity_model = AutoML(
                **{**self._settings["component_models"], "task": "classification","estimator_list":['xgboost']}
            )
        elif propensity_model == "bayesian":
            self.propensity_model = AutoML(
                **{**self._settings["component_models"], "task": "classification","estimator_list":['Gaussian']}
            )
            self.propensity_model.add_learner(learner_name='Gaussian', learner_class=Gaussian)
        elif propensity_model == "svm":
            self.propensity_model = AutoML(
                **{**self._settings["component_models"], "task": "classification","estimator_list":['SVM']}
            )
            self.propensity_model.add_learner(learner_name='SVM', learner_class=SVM)
        elif propensity_model == "mlp":
            self.propensity_model = AutoML(
                **{**self._settings["component_models"], "task": "classification","estimator_list":['MLP']}
            )
            self.propensity_model.add_learner(learner_name='MLP', learner_class=MLP)
        elif propensity_model == "gpc":
            self.propensity_model = AutoML(
                **{**self._settings["component_models"], "task": "classification","estimator_list":['GPC']}
            )
            self.propensity_model.add_learner(learner_name='GPC', learner_class=GPC)
            #self.propensity_model.add_learner(learner_name='GPC', learner_class=GPC)
        elif propensity_model == "dt":
            self.propensity_model = AutoML(
                **{**self._settings["component_models"], "task": "classification","estimator_list":['DTC','ExTC']}
            )
            self.propensity_model.add_learner(learner_name='DTC', learner_class=DTC)
            self.propensity_model.add_learner(learner_name='ExTC', learner_class=ExTC)
        elif propensity_model == "lr":
            self.propensity_model = AutoML(
                **{**self._settings["component_models"], "task": "classification","estimator_list":['LR']}
            )
            self.propensity_model.add_learner(learner_name='LR', learner_class=LR)
        elif propensity_model == "super":
            self.propensity_model = AutoML(
                **{**self._settings["component_models"], "task": "classification","estimator_list":['Gaussian','Multinomial','SVM','MLP','DTC','ExTC','LR']}
            )
            self.propensity_model.add_learner(learner_name='Gaussian', learner_class=Gaussian)
            self.propensity_model.add_learner(learner_name='Multinomial', learner_class=Multinomial)
            self.propensity_model.add_learner(learner_name='SVM', learner_class=SVM)
            self.propensity_model.add_learner(learner_name='MLP', learner_class=MLP)
            self.propensity_model.add_learner(learner_name='DTC', learner_class=DTC)
            self.propensity_model.add_learner(learner_name='ExTC', learner_class=ExTC)
            self.propensity_model.add_learner(learner_name='LR', learner_class=LR)
        elif hasattr(propensity_model, "fit") and hasattr(
            propensity_model, "predict_proba"
        ):
            self.propensity_model = propensity_model
        else:
            raise ValueError(
                'propensity_model valid values are "dummy", "auto", or a classifier object'
            )

    def fit(
        self,
        data: Union[pd.DataFrame, CausalityDataset],
        vdata: Union[pd.DataFrame, CausalityDataset] = None,
        treatment: str = None,
        outcome: str = None,
        common_causes: List[str] = None,
        effect_modifiers: List[str] = None,
        instruments: List[str] = None,
        propensity_modifiers: Optional[List[str]] = None,
        estimator_list: Optional[Union[str, List[str]]] = None,
        resume: Optional[bool] = False,
        time_budget: Optional[int] = None,
        stratify=None
    ):
        """Performs AutoML on list of causal inference estimators
        - If estimator has a search space specified in its parameters, HPO is performed on the whole model.
        - Otherwise, only its component models are optimised

        Args:
            data_df (pandas.DataFrame): dataset for causal inference
            treatment (str): name of treatment variable
            outcome (str): name of outcome variable
            common_causes (List[str]): list of names of common causes
            effect_modifiers (List[str]): list of names of effect modifiers
            propensity_modifiers (List[str]): list of names of propensity modifiers
            instruments (List[str]): list of names of instrumental variables
            estimator_list (Optional[Union[str, List[str]]]): subset of estimators to consider
            resume (Optional[bool]): set to True to continue previous fit
            time_budget (Optional[int]): change new time budget allocated to fit, useful for warm starts.

        """
        if not isinstance(data, CausalityDataset):
            assert isinstance(data, pd.DataFrame)
            data = CausalityDataset(
                data,
                treatment,
                outcome,
                common_causes=common_causes,
                effect_modifiers=effect_modifiers,
                instruments=instruments,
                propensity_modifiers=propensity_modifiers,
            )

        self.data = data
        treatment_values = data.treatment_values

        assert (
            len(treatment_values) > 1
        ), "Treatment must take at least 2 values, eg 0 and 1!"

        self._control_value = treatment_values[0]
        self._treatment_values = list(treatment_values[1:])

        # To be used for component model training/selection
        stratifylist= self.data.data[stratify] if (stratify is not None) else None
        if(vdata is None):
            self.train_df, self.test_df = train_test_split(
                self.data.data, train_size=self._settings["train_size"], shuffle=True,stratify=stratifylist
            )
        else:
            self.train_df = self.data.data
            self.test_df = vdata 

        # smuggle propensity modifiers into common causes, filter later in component models
        self.causal_model = CausalModel(
            data=self.train_df,
            treatment=data.treatment,
            outcome=data.outcomes[0],
            common_causes=data.common_causes + data.propensity_modifiers,
            effect_modifiers=data.effect_modifiers,
            instruments=data.instruments,
        )

        self.init_propensity_model(self._settings["propensity_model"])
        # if we are only supplying certain features to the propensity function,
        # make them invisible to the outcome component model
        # This is a workaround for the DoWhy/EconML data model which doesn't
        # support that out of the box
        propensity_only_cols = [
            p
            for p in data.propensity_modifiers
            if p not in data.common_causes + data.effect_modifiers
        ]

        if len(propensity_only_cols):
            outcome_model_class = feature_filter(
                AutoML, data.effect_modifiers + data.common_causes, first_cols=True
            )
        else:
            outcome_model_class = AutoML
        # outcome_model_class = AutoML
        self.outcome_model = outcome_model_class(**self._settings["component_models"])

        self.identified_estimand: IdentifiedEstimand = (
            self.causal_model.identify_effect(proceed_when_unidentifiable=True)
        )

        if bool(self.identified_estimand.estimands["iv"]) and bool(instruments):
            self.problem = "iv"
        elif bool(self.identified_estimand.estimands["backdoor"]):
            self.problem = "backdoor"
        else:
            raise ValueError(
                "Couldn't identify the kind of problem from "
                + str(self.identified_estimand.estimands)
            )
        print(f'AutoCausality :: Causal Identify Problem: {self.problem}',flush=True)

        # This must be stateful because we need to train the treatment propensity function
        self.scorer = Scorer(
            self.causal_model,
            self.propensity_model,
            self.problem,
            treatment_is_multivalue(self._treatment_values),
        )

        self.metric = self.scorer.resolve_metric(self._settings["metric"])
        self.metrics_to_report = self.scorer.resolve_reported_metrics(
            self._settings["metrics_to_report"], self.metric
        )

        if (self.metric == "energy_distance" or self.metric=="effectMSE"):
            self._best_estimators = defaultdict(lambda: (float("inf"), None))

        # TODO: allow specifying an exclusion list, too
        used_estimator_list = (
            self.original_estimator_list if estimator_list is None else estimator_list
        )

        assert (
            isinstance(used_estimator_list, str) or len(used_estimator_list) > 0
        ), "estimator_list must either be a str or an iterable of str"

        # config with method-specific params
        self.cfg = SimpleParamService(
            self.propensity_model,
            self.outcome_model,
            n_jobs=self._settings["component_models"]["n_jobs"],
            include_experimental=self._settings["include_experimental_estimators"],
            multivalue=treatment_is_multivalue(self._treatment_values),
        )

        self.estimator_list = self.cfg.estimator_names_from_patterns(
            self.problem,
            used_estimator_list,
            len(self.data),
        )

        if not self.estimator_list:
            #raise ValueError(
            #    f"No valid estimators in {str(used_estimator_list)}, "
            #    f"available estimators: {str(self.cfg.estimator_names)}"
            #)
            print(
                f"No valid estimators in {str(used_estimator_list)}, "
                f"available estimators: {str(self.cfg.estimator_names)}"
            ,flush=True)
            print(f'Problem: {self.problem} , Estimator: {str(used_estimator_list)}',flush=True)
            return -1

        if time_budget:
            self._settings["tuner"]["time_budget_s"] = time_budget

        if self._settings["component_models"]["time_budget"] is None:
            self._settings["component_models"]["time_budget"] = self._settings["tuner"][
                "time_budget_s"
            ] / (2.5 * len(self.estimator_list))

        if (
            self._settings["tuner"]["time_budget_s"] is None
            and self._settings["tuner"]["num_samples"] == -1
        ):
            self._settings["tuner"]["time_budget_s"] = (
                2.5
                * len(self.estimator_list)
                * self._settings["component_models"]["time_budget"]
            )

        cmtb = self._settings["component_models"]["time_budget"]

        if cmtb < 300:
            warnings.warn(
                f"Component model time budget is {cmtb}. "
                f"Recommended value is at least 300 for smallish datasets, 1800 for datasets with> 100K rows"
            )

        if self._settings["test_size"] is not None:
            self.test_df = self.test_df.sample(self._settings["test_size"])

        self.r_scorer = (
            None
            if "r_scorer" not in self.metrics_to_report
            else RScoreWrapper(
                self.outcome_model,
                self.propensity_model,
                self.train_df,
                self.test_df,
                outcome,
                treatment,
                common_causes,
                effect_modifiers,
            )
        )

        search_space = self.cfg.search_space(self.estimator_list)
        init_cfg = (
            self.cfg.default_configs(self.estimator_list)
            if self._settings["try_init_configs"]
            else []
        )

        if resume and self.results:
            # pull out configs and resume_scores from previous trials:
            for _, result in self.results.results.items():
                self.resume_scores.append(result[self.metric])
                self.resume_cfg.append(result["config"])
            # append init_cfgs that have not yet been evaluated
            for cfg in init_cfg:
                self.resume_cfg.append(cfg) if cfg not in self.resume_cfg else None

        self.results = tune.run(
            self._tune_with_config,
            search_space,
            metric=self.metric,
            points_to_evaluate=init_cfg
            if len(self.resume_cfg) == 0
            else self.resume_cfg,
            evaluated_rewards=[]
            if len(self.resume_scores) == 0
            else self.resume_scores,
            mode="min" if (self.metric == "energy_distance" or self.metric=="effectMSE") else "max",
            low_cost_partial_config={},
            **self._settings["tuner"],
        )

        if self.results.get_best_trial() is None:
            raise Exception(
                "Optimization failed! Did you set large enough time_budget and components_budget?"
            )

        self.update_summary_scores()

    def update_summary_scores(self):
        self.scores = Scorer.best_score_by_estimator(self.results.results, self.metric)
        # now inject the separately saved model objects
        for est_name in self.scores:
            # Todo: Check approximate scores for OrthoIV (possibly other IV estimators)
            # assert (
            #     self._best_estimators[est_name][0] == self.scores[est_name][self.metric]
            # ), "Can't match best model to score"
            self.scores[est_name]["estimator"] = self._best_estimators[est_name][1]

    def _tune_with_config(self, config: dict) -> dict:
        """Performs Hyperparameter Optimisation for a
        causal inference estimator

        Args:
            config (dict): dictionary with search space for
            all tunable parameters

        Returns:
            dict: values of metrics after optimisation
        """
        # estimate effect with current config

        # if using FLAML < 1.0.7 need to set n_jobs = 2 here
        # to spawn a separate process to prevent cross-talk between tuner and automl on component models:

        estimates = Parallel(n_jobs=2, backend="threading")(
            delayed(self._estimate_effect)(config["estimator"]) for i in range(1)
        )[0]

        # pop and cache separately the fitted model object, so we only store the best ones per estimator
        if "exception" not in estimates:
            est_name = estimates["estimator_name"]
            if (
                self._best_estimators[est_name][0] > estimates[self.metric]
                if (self.metric == "energy_distance" or self.metric == "effectMSE")
                else self._best_estimators[est_name][0] < estimates[self.metric]
            ):
                if self._settings["store_all"]:
                    self._best_estimators[est_name] = (
                        estimates[self.metric],
                        estimates["estimator"],
                    )
                else:
                    self._best_estimators[est_name] = (
                        estimates[self.metric],
                        estimates.pop("estimator"),
                    )

        return estimates

    def _estimate_effect(self, config):
        """estimates effect with chosen estimator"""

        # add params that are tuned by flaml:
        config = clean_config(copy.copy(config))
        self.estimator_name = config.pop("estimator_name")
        # params_to_tune = {
        #     k: v for k, v in config.items() if (not k == "estimator_name")
        # }
        cfg = self.cfg.method_params(self.estimator_name)

        try:  #
            # if True:  #
            estimate = self.causal_model.estimate_effect(
                self.identified_estimand,
                method_name=self.estimator_name,
                control_value=self._control_value,
                treatment_value=self._treatment_values,
                target_units="ate",  # condition used for CATE
                confidence_intervals=False,
                method_params={
                    "init_params": {**deepcopy(config), **cfg.init_params},
                    "fit_params": {},
                },
            )
            scores = {
                "estimator_name": self.estimator_name,
                "train": self._compute_metrics(
                    estimate,
                    self.train_df,
                ),
                "validation": self._compute_metrics(
                    estimate,
                    self.test_df,
                ),
            }

            return {
                self.metric: scores["validation"][self.metric],
                "estimator": estimate,
                "estimator_name": scores.pop("estimator_name"),
                "scores": scores,
                "config": config,
            }
        except Exception as e:
            print("Evaluation failed!\n", config, traceback.format_exc())
            return {
                self.metric: -np.inf,
                "estimator_name": self.estimator_name,
                "exception": e,
                "traceback": traceback.format_exc(),
            }

    def _compute_metrics(self, estimator, df: pd.DataFrame) -> dict:
        return self.scorer.make_scores(
            estimator, df, self.metrics_to_report, r_scorer=None
        )

    def score_dataset(self, df: pd.DataFrame, dataset_name: str):
        """
        After fitting, generate scores for an additional dataset, add them to the scores dict
        @param df:
        @param dataset_name:
        @return:
        """
        for scr in self.scores.values():
            scr["scores"][dataset_name] = self._compute_metrics(scr["estimator"], df)

    @property
    def best_estimator(self) -> str:
        """A string indicating the best estimator found"""
        return self.results.best_result["estimator_name"]

    @property
    def model(self):
        """Return the *trained* best estimator"""
        return self.results.best_result["estimator"].estimator

    def best_model_for_estimator(self, estimator_name):
        """Return the best model found for a particular estimator.
        estimator: self.tune_results[estimator].best_config

        Args:
            estimator_name: a str of the estimator's name.

        Returns:
            An object storing the best model for estimator_name.
        """
        # Note that this returns the trained Econml estimator, whose attributes include
        # fitted  models for E[T | X, W], for E[Y | X, W], CATE model, etc.
        return self.scores[estimator_name]["estimator"]

    @property
    def best_config(self):
        """A dictionary containing the best configuration"""
        return self.results.best_config

    @property
    def best_config_per_estimator(self):
        """A dictionary of all estimators' best configuration."""
        return {e: s["config"] for e, s in self.scores.values()}

    @property
    def best_score_per_estimator(self):
        """A dictionary of all estimators' best score."""
        return {}

    @property
    def best_score(self):
        """A float of the best score found."""
        return self.results.best_result[self.metric]

    def effect(self, df, *args, **kwargs):
        return self.model.effect(df, *args, **kwargs)
