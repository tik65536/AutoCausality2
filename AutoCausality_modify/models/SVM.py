from flaml.model import SKLearnEstimator
from flaml import tune
import logging
from flaml.data import (
    group_counts,
)
from sklearn.svm import NuSVC
from sklearn.svm import SVC
import time
logger = logging.getLogger("flaml.automl")

class SVM(SKLearnEstimator):
    def __init__(self,task='classification',**config):
        super().__init__(task,**config)
        self.estimator_class=SVC

    @classmethod
    def search_space(cls,data_size,task):
        space={
            'C':{'domain':tune.uniform(lower=0.1,upper=100),'init_value': 1 },
            'kernel':{'domain':tune.choice(['linear','poly','rbf','sigmoid']),'init_value':'rbf'},
            'degree':{'domain':tune.choice([1,2,3,4,5,6,7,8,9,10]),'init_value':3},
            #'gamma':{'domain':tune.choice(['auto','scale',float(tune.uniform(lower=0.001,upper=100).sample())]),'init_value': 'scale'}
            'gamma':{'domain':tune.choice(['auto','scale']),'init_value': 'auto'}
        }
        return space


    def _fit(self, X_train, y_train, **kwargs):
            current_time = time.time()
            if "groups" in kwargs:
                kwargs = kwargs.copy()
                groups = kwargs.pop("groups")
                if self._task == "rank":
                    kwargs["group"] = group_counts(groups)
                    # groups_val = kwargs.get('groups_val')
                    # if groups_val is not None:
                    #     kwargs['eval_group'] = [group_counts(groups_val)]
                    #     kwargs['eval_set'] = [
                    #         (kwargs['X_val'], kwargs['y_val'])]
                    #     kwargs['verbose'] = False
                    #     del kwargs['groups_val'], kwargs['X_val'], kwargs['y_val']
            X_train = self._preprocess(X_train)
            params=self.params
            del params['n_jobs']
            params['probability']=True
            params['max_iter']=1000
            model = self.estimator_class(**params)
            if logger.level == logging.DEBUG:
                # xgboost 1.6 doesn't display all the params in the model str
                logger.debug(f"flaml.model - {model} fit started with params {self.params}")
            model.fit(X_train, y_train, **kwargs)
            if logger.level == logging.DEBUG:
                logger.debug(f"flaml.model - {model} fit finished")
            train_time = time.time() - current_time
            self._model = model
            return train_time

