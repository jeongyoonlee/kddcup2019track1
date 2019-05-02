import logging
import os
import sys


team_name = 'CaptainEnsembl'

logger = logging.getLogger(name=team_name)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(name)s: %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)
logger.addHandler(handler)

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class Config(object):

    def __init__(self,
                 data_dir=os.path.join(root_dir, 'input'),
                 build_dir=os.path.join(root_dir, 'build'),
                 phase=1,
                 seed=2019,
                 n_fold=5,
                 n_class=12):

        #----------------------------------------------------------------------
        # Input data folder and files
        #----------------------------------------------------------------------
        self.phase = phase
        self.data_dir = os.path.join(data_dir, 'data_set_phase{}'.format(self.phase))
        self.profile_file = os.path.join(self.data_dir, 'profiles.csv')
        self.train_click_file = os.path.join(self.data_dir, 'train_clicks.csv')
        self.train_plan_file = os.path.join(self.data_dir, 'train_plans.csv')
        self.train_query_file = os.path.join(self.data_dir, 'train_queries.csv')
        self.test_plan_file = os.path.join(self.data_dir, 'test_plans.csv')
        self.test_query_file = os.path.join(self.data_dir, 'test_queries.csv')

        #----------------------------------------------------------------------
        # Output folders
        #----------------------------------------------------------------------
        self.build_dir = build_dir
        self.feature_dir = os.path.join(self.build_dir, 'feature')
        self.model_dir = os.path.join(self.build_dir, 'model')
        self.metric_dir = os.path.join(self.build_dir, 'metric')
        self.val_dir = os.path.join(self.build_dir, 'val')
        self.tst_dir = os.path.join(self.build_dir, 'tst')
        self.sub_dir = os.path.join(self.build_dir, 'sub')
        for d in [self.build_dir, self.feature_dir, self.model_dir, self.metric_dir,
                  self.val_dir, self.tst_dir, self.sub_dir]:
            if not os.path.exists(d):
                os.makedirs(d)
                logger.info('{} created'.format(d))

        #----------------------------------------------------------------------
        # Output files
        #----------------------------------------------------------------------
        self.metric_file = os.path.join(self.metric_dir, 'metrics.csv')

        # Common feature files
        self.plan_file = os.path.join(self.feature_dir, 'plans.csv')
        self.pid_feature_file = os.path.join(self.feature_dir, 'pid_features.csv')
        self.od_feature_file = os.path.join(self.feature_dir, 'od_features.csv')
        self.od_cluster_feature_file = os.path.join(self.feature_dir, 'od_cluster_features.csv')

        #----------------------------------------------------------------------
        # Member variables
        #----------------------------------------------------------------------
        self.seed = seed
        self.n_fold = n_fold
        self.n_class = n_class
        self.feature_name = None
        self.algo_name = None
        self.model_name = None


    def set_feature_name(self, feature_name):
        self.feature_name = feature_name
        self.train_feature_file = os.path.join(self.feature_dir, '{}.trn.csv'.format(self.feature_name))
        self.test_feature_file = os.path.join(self.feature_dir, '{}.tst.csv'.format(self.feature_name))

    def set_algo_name(self, algo_name):
        assert self.feature_name

        self.algo_name = algo_name
        self.model_name = '{}_{}'.format(self.algo_name, self.feature_name)
        self.feature_imp_file = os.path.join(self.model_dir, '{}.imp.csv'.format(self.model_name))
        self.model_file = os.path.join(self.model_dir, '{}.mdl'.format(self.model_name))
        self.predict_val_file = os.path.join(self.val_dir, '{}.val.csv'.format(self.model_name))
        self.predict_tst_file = os.path.join(self.tst_dir, '{}.tst.csv'.format(self.model_name))
        self.submission_file = os.path.join(self.sub_dir, '{}.sub.csv'.format(self.model_name))


config = Config()