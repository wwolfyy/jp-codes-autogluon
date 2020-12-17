# %% import modules
# import mxnet as mx
from autogluon.tabular import TabularPrediction as task
from datetime import datetime
import pandas as pd

# %% define data

root_folder = "/home/lstm/Google Drive/MATLAB data files/Project__autoML/datasets for autoML/data_weekly_archive/"
data_folder = "20200213/"
data_file = "GCP_trainvalid_KOSPIb1f0bNsCFCCOFOC20200213.csv"
data_ref = 'KOSPIb1f0bNsCFCCOFOC20200213'
target_col = "target"
most_recent_folder = "20112032/"
most_recent_file = "GCP_trainvalid_KOSPIb1f0bNsCFCCOFOC2020112032.csv"

cols_2_drop_4_training = ["timestamp", "split_tag", "weight_vector"]

data_trainvalid = task.Dataset(file_path= root_folder+ data_folder + data_file)
data_trainvalid["DoW"] = data_trainvalid["DoW"].astype('category')

train_data = data_trainvalid.loc[data_trainvalid.split_tag=='TRAIN',:]
print(train_data.head())
print(train_data.tail())

valid_data = data_trainvalid.loc[data_trainvalid.split_tag=='VALIDATE',:] # do not provide if bagging/stacking
print(valid_data.head())
print(valid_data.tail())
latest_valid_date = valid_data["timestamp"][valid_data["timestamp"]==valid_data["timestamp"].max()]

## REDO TEST DATA (to be pre-processed in matlab first)
test_data = task.Dataset(file_path= root_folder+ most_recent_folder + most_recent_file)
test_data["DoW"] = test_data["DoW"].astype('category')
test_data = test_data.loc[test_data["timestamp"] > latest_valid_date.iloc[0],:]
print(test_data.head())
print(test_data.tail())

y_test = pd.DataFrame(test_data[target_col]).reset_index(drop=True)  # values to predict
x_test = test_data.drop(labels=[target_col],axis=1)  # drop target col from dataset

# %% define training paramaters

# now = datetime.now().strftime("%Y%m%d_%H%M%S") # current date and time
output_dir = "~/Desktop/AutogluonModels/"+data_ref # where to save trained predictors
time_limit = 3600 * 1
metric = "root_mean_squared_error" # this is default if not specified
        # ‘root_mean_squared_error’, ‘mean_squared_error’, ‘mean_absolute_error’, ‘median_absolute_error’, ‘r2’
preset = "best_quality" # same as auto_stack=True
        # "medium_quality_faster_train"
        # 'good_quality_faster_inference_only_refit'
        # 'optimize_for_deployment'
hyperparameter_tune = True # set to False if ensemble/stack
# hyperparameters =
# stack_ensemble_levels =
# auto_stack =
# num_bagging_folds =
# num_bagging_sets =
# num_trials =  # max number of trials for each parameter combination
# search_strategy = # "skopt", etc...
# ngpus_per_trial = # automatically determined if unspecified
# tuning_data = # validation data (don't use if bagging/stacking)
# holdout_frac =

# %% train model

predictor = task.fit(train_data=train_data.drop(labels=cols_2_drop_4_training, axis=1),
                     tuning_data=valid_data.drop(labels=cols_2_drop_4_training, axis=1),
                     label=target_col,
                     #hyperparameter_tune=hyperparameter_tune,
                     auto_stack=True,
                     time_limits=time_limit,
                     output_directory=output_dir,
                     eval_metric=metric,
                     keep_only_best=True,
                     save_space=True,
                     ngpus_per_trial=1,
                     presets=preset)

# %% output model info

results = predictor.fit_summary()
performance = predictor.evaluate(test_data.drop(labels=cols_2_drop_4_training, axis=1))
predictor.leaderboard(test_data, silent=True)
predictor.get_model_best() # get name of best model
predictor.get_model_names() # get list of model names
# specific_model = predictor._trainer.load_model("model_name")
# model_info = specific_model.get_info() # get info on specific model
predictor_information = predictor.info() # get info on predictor

# %% predict with test data

pred = pd.Series(predictor.predict(test_data.drop(labels=cols_2_drop_4_training, axis=1)))
#predictor.predict(test_data, model='NeuralNetClassifier')

# %% plot

axis_array = pd.Series(range(len(y_test)))
divider_idx = len(axis_array)
plot_engine = 'matplotlib' # 'plotly'

import sys
sys.path.append("/home/lstm/Github/jp-codes-python/autoML_py36")
import jp_utils
jp_utils.stem_plot_compare(  # FIXME
    axis_array=axis_array,
    divider_idx=divider_idx,
    array_a=y_test,
    name_a='target',
    array_b=pred,
    name_b='pred',
    title_prefix=data_ref,
    window_size=5,
    fig_height=7,
    sharex=True,  # bool
    alert=False,  # alert if no test set
    plot_engine=plot_engine  # matplotlib or plotly
)

# %% scratches

importance_scores = predictor.feature_importance(test_data) # get feature importance score

# if using csv as input
predictor = task.fit(train_data=task.Dataset(file_path=TRAIN_DATA.csv), label=COLUMN_NAME)
predictions = predictor.predict(task.Dataset(file_path=TEST_DATA.csv))

# to save disk space
predictor.save_space() # delete auxiliary files produced during fit(): loses metrics etc.
predictor.delete_models(models_to_keep='best', dry_run=False)


