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
metrics = ["effectMSE"]
#metrics = ["energy_distance"]
components_time_budget = 300
estimator_list = "all"
out_dir = "./Super_NewData/"
filename_out = "synthetic_observational_cate"
#datapath="./DataSet/"

run=np.arange(100)
#np.random.shuffle(run)
test=run[72:]
run=run[:70]

for i_run in [45]:
    i_run = int(i_run)
    print(f"Dataset {i_run} Start")
    #with open(f"{datapath}dataset_run_{i_run+1}.data", "rb") as f:
    #    data=pickle.load(f)
    #train_df=data['train_df']
    #test_df=data['test_df']
    i_run = int(i_run)
    train_df = load_realcause_dataset('lalonde_psid', i_run)
    for di in range(1,4):
    	train_df=train_df.append(load_realcause_dataset('lalonde_psid',i_run+di))
    print(f'Data Set Shape : {train_df.shape}',flush=True)
    features_X=list(train_df.columns[:-5])
    train_df.rename(columns = {'t':'treatment','y':'outcome','ite':'true_effect'}, inplace = True)
    itemedian=train_df['true_effect'].median()
    d=train_df['true_effect'].values
    d=np.where(d>itemedian,1,0)
    train_df['over']=d
    #omin,omax=np.min(train_df['re74']),np.max(train_df['re74'])
    #tmin,tmax=np.min(train_df['re75']),np.max(train_df['re75'])
    #train_df['re74']=(train_df['re74']-omin)/omax
    #train_df['re75']=(train_df['re75']-tmin)/tmax
    #xmin,xmax=np.min(train_df['outcome']),np.max(train_df['outcome'])
    #tmean,tvar=train_df['true_effect'].mean(),train_df['true_effect'].std()
    #train_df['outcome']=(train_df['outcome']-xmin)/xmax
    #train_df['true_effect']=(train_df['true_effect']-tmean)/tvar
    #features_W=data['features_W']
    print(f"features_X: {features_X}")
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
            stratify="over"
        )
        # compute relevant scores (skip newdummy)
        # get scores on train,val,test for each trial,
        # sort trials by validation set performance
        # assign trials to estimators
        estimator_scores = {est: [] for est in ac.scores.keys() if "NewDummy" not in est}
        print(f"Dataset {i_run} Make Score")
        for trial in ac.results.trials:
            # estimator name:
            estimator_name = trial.last_result["estimator_name"]
            print(f"Dataset {i_run} Make Score {estimator_name} ")
            if  trial.last_result["estimator"]:
                estimator = trial.last_result["estimator"]
                scores = {}
                t = int(np.random.choice(test,1))
                tmp =  load_realcause_dataset('lalonde_psid', t)
                tmp.rename(columns = {'t':'treatment','y':'outcome','ite':'true_effect'}, inplace = True)
                tmp.drop(['y0','y1'],axis=1,inplace=True)
                #omin,omax=np.min(tmp['outcome']),np.max(tmp['outcome'])
                #tmin,tmax=tmp['true_effect'].mean(),tmp['true_effect'].std()
                #tmp['outcome']=(tmp['outcome']-omin)/omax
                #tmp['true_effect']=(tmp['true_effect']-tmin)/tmax
                #omin,omax=np.min(tmp['re74']),np.max(tmp['re74'])
                #tmin,tmax=np.min(tmp['re75']),np.max(tmp['re75'])
                #tmp['re74']=(tmp['re74']-omin)/omax
                #tmp['re75']=(tmp['re75']-tmin)/tmax
                datasets = {"test":tmp}
                for ds_name, df in datasets.items():
                    print(f"Dataset {i_run} Make Score {estimator_name}, TestSet {ds_name} ")
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
                break


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
    print(f"Dataset {i_run} End {time.time()-starttime} , best {ac.best_estimator}")



    with open(f"{out_dir}{filename_out}_{metric}_run_{i_run}_test_{t}.pkl", "wb") as f:
        pickle.dump(results, f)
