import pickle
import warnings
import time
warnings.filterwarnings('ignore') # suppress sklearn deprecation warnings for now..

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from auto_causality import AutoCausality
from auto_causality.datasets import generate_synthetic_data
from auto_causality.data_utils import CausalityDataset
from loading import load_realcause_dataset
import pandas as pd
import numpy as np
import argparse
metrics = ["effectMSE"]
estimator_list = "all"
out_dir = "./Super_NewData/"
filename_out = "synthetic_observational_cate"
#datapath="./DataSet/"
parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, default="lalonde_psid",
                    help='lalonde_psid or lalonde_cps or twins' )

parser.add_argument('-numOfSample', type=int, default=4,
                    help='Numer of Sub Dataset to use')

parser.add_argument('-budget', type=int, default=300,
                    help='Time budget fro Components Model')
args = parser.parse_args()
datasetname=args.dataset
numOfSample=args.numOfSample
components_time_budget = args.budget
run=np.arange(80)
test=np.arange(80,100)
np.random.shuffle(run)


for nouse in range(1):
    run=run[:numOfSample]
    print(f"Dataset ID:{run} Start",flush=True)
    train_df = load_realcause_dataset(datasetname, int(run[0]))
    for di in run[1:]:
       train_df=train_df.append(load_realcause_dataset(datasetname,int(di)))
    print(f'Data Set Shape : {train_df.shape}',flush=True)
    features_X=list(train_df.columns[:-5])
    train_df.rename(columns = {'t':'treatment','y':'outcome','ite':'true_effect'}, inplace = True)
    print(f"features_X: {features_X}",flush=True)
    starttime=time.time()

    for metric in metrics:
        ac = AutoCausality(
            metric=metric,
            verbose=1,
            components_verbose=1,
            components_time_budget=components_time_budget,
            estimator_list=estimator_list,
            store_all_estimators=True,
            propensity_model="super",
        )

        ac.fit(
            train_df,
            treatment="treatment",
            outcome=["outcome"],
            #common_causes=features_W,
            effect_modifiers=features_X,
        )
        # compute relevant scores (skip newdummy)
        # get scores on train,val,test for each trial,
        # sort trials by validation set performance
        # assign trials to estimators
        estimator_scores = {est: [] for est in ac.scores.keys() if "NewDummy" not in est}
        print(f"Dataset {datasetname} Make Score",flush=True)
        for trial in ac.results.trials:
            # estimator name:
            estimator_name = trial.last_result["estimator_name"]
            print(f"Dataset {datasetname} Make Score {estimator_name} ",flush=True)
            if  trial.last_result["estimator"]:
                estimator = trial.last_result["estimator"]
                scores = {}
                t = int(np.random.choice(test,1))
                tmp =  load_realcause_dataset('lalonde_psid', t)
                tmp.rename(columns = {'t':'treatment','y':'outcome','ite':'true_effect'}, inplace = True)
                tmp.drop(['y0','y1'],axis=1,inplace=True)
                datasets = {"test":tmp}
                for ds_name, df in datasets.items():
                    print(f"Dataset {datasetname} Make Score {estimator_name}, TestSet ID: {t} ",flush=True)
                    scores[ds_name] = {}
                    # make scores
                    if not isinstance(df, CausalityDataset):
                        assert isinstance(df, pd.DataFrame)
                        data = CausalityDataset(
                            df,
                            treatment="treatment",
                            outcomes=["outcome"],
                            #common_causes=features_W,
                            effect_modifiers=features_X,
                        )

                    est_scores = ac.scorer.make_scores(
                        estimator,
                        data.data,
                        #problem=ac.problem,
                        metrics_to_report=ac.metrics_to_report,
                    )

                    # add cate:
                    scores[ds_name]["CATE_estimate"] = estimator.estimator.effect(df)
                    # add ground truth for convenience
                    scores[ds_name]["CATE_groundtruth"] = df["true_effect"]
                    scores[ds_name][metric] = est_scores[metric]
                    try:
                        scores[ds_name]['#_Propensity_model']=est_scores['#_Propensity_model']
                        scores[ds_name]['#_Propensity_Para']=est_scores['#_Propensity_model_param']
                        scores[ds_name]['values']=est_scores['values']
                    except KeyError:
                        pass
                estimator_scores[estimator_name].append(scores)

        # sort trials by validation performance
        for k in estimator_scores.keys():
            estimator_scores[k] = sorted(
                estimator_scores[k],
            key=lambda x: x["test"][metric],
            reverse=False if (metric == "energy_distance" or metric =="effectMSE") else True,
        )
    results = {
        "best_estimator": ac.best_estimator,
        "best_config": ac.best_config,
        "best_score": ac.best_score,
        "optimised_metric": metric,
        "scores_per_estimator": estimator_scores,
    }
    print(f"Dataset {datasetname} End {time.time()-starttime} , best {ac.best_estimator}",flush=True)



    with open(f"{out_dir}{filename_out}_{metric}_run_{run}_test_{t}.pkl", "wb") as f:
        pickle.dump(results, f)
