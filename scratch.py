# %% import modules
# import mxnet as mx
from autogluon.tabular import TabularPrediction as task

# %% define training paramaters

target_col = "age"
output_dir = "models_scratch" # where to save trained predictors
time_limit = 3600
metric = "root_mean_squared_error" # this is default if not specified
        # mean_absolute_error
preset = "best_quality" # same as auto_stack=True
        # "medium_quality_faster_train"
        # 'good_quality_faster_inference_only_refit'
        # 'optimize_for_deployment'
hyperparameter_tune = False # set to True if not ensembling
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



train_data = task.Dataset(file_path='https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
subsample_size = 500  # subsample subset of data for faster demo, try setting this to much larger values
train_data = train_data.sample(n=subsample_size, random_state=0)
print(train_data.head())
train_data[target_col].describe()

# valid_data : do not provide if bagging/stacking

test_data = task.Dataset(file_path='https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
test_data = test_data.sample(n=100, random_state=0)
y_test = test_data[target_col]  # values to predict
x_test = test_data.drop(labels=[target_col],axis=1)  # drop target col from dataset
print(x_test.head())

predictor = task.fit(train_data=train_data,
                     label=target_col,
                     time_limits=time_limit,
                     output_directory=output_dir
                     eval_metric=metric,
                     presets=preset)

results = predictor.fit_summary()
performance = predictor.evaluate(test_data)
predictor.leaderboard(test_data, silent=True)
predictor.get_model_best() # get name of best model
predictor.get_model_names() # get list of model names
specific_model = predictor._trainer.load_model("model_name")
model_info = specific_model.get_info() # get info on specific model
predictor_information = predictor.info() # get info on predictor

predictor.predict(test_data)
#predictor.predict(test_data, model='NeuralNetClassifier')

importance_scores = predictor.feature_importance(test_data) # get feature importance score

# if using csv as input
predictor = task.fit(train_data=task.Dataset(file_path=TRAIN_DATA.csv), label=COLUMN_NAME)
predictions = predictor.predict(task.Dataset(file_path=TEST_DATA.csv))

# to save disk space
predictor.save_space() # delete auxiliary files produced during fit(): loses metrics etc.
predictor.delete_models(models_to_keep='best', dry_run=False)


