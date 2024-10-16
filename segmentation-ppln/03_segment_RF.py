#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Supported by BiCubes project (HFRI grant 3943)

"""
This script trains and evaluates a Random Forest classifier on satellite imagery data.
It handles feature extraction, model training, and validation, and outputs a confusion matrix.

Dependencies:
- numpy
- osgeo (gdal)
- pandas
- sklearn
- natsort
- joblib
- optparse
- xlsxwriter
- multiprocessing
"""

import numpy as np
from osgeo import gdal
from glob import glob
import pandas as pd
import sklearn.metrics as metr
import sklearn.ensemble as ensemble
import sklearn.svm as svm
import sklearn.model_selection as model_selection
import natsort
import joblib
import os
import optparse
import sys
import time
import xlsxwriter
import multiprocessing

class OptionParser(optparse.OptionParser):
  """
  Custom option parser to handle required arguments.
  """
  def check_required(self, opt):
      """
      Check if a required option is provided.
      """
      option = self.get_option(opt)
      # Assumes the option's 'default' is set to None!
      if getattr(self.values, option.dest) is None:
          self.error("{} option is not supplied. {} is required for this script!".format(option, option))

def convert(seconds):
  """
  Convert seconds to a formatted string of hours, minutes, and seconds.

  Parameters:
  - seconds: int, time in seconds.

  Returns:
  - str, formatted time string.
  """
  seconds = seconds % (24 * 3600)
  hour = seconds // 3600
  seconds %= 3600
  minutes = seconds // 60
  seconds %= 60

  return "%d:%02d:%02d" % (hour, minutes, seconds)

def confusion_matrix(gt, pred, fullpath, labels, labels_present_only_in_one, unique_classes_train):
  """
  Compute and save a confusion matrix with additional statistics.

  Parameters:
  - gt: numpy array, ground truth labels.
  - pred: numpy array, predicted labels.
  - fullpath: str, path to save the confusion matrix.
  - labels: list of str, class labels.
  - labels_present_only_in_one: numpy array, labels present only in one of train or test datasets.
  - unique_classes_train: numpy array, unique classes in the training data.

  Returns:
  - df: pandas DataFrame, the confusion matrix with statistics.
  """
  # Convert to int
  y_gt = gt.astype('int')
  y_pred = pred.astype('int')

  # Compute metrics
  cm = metr.confusion_matrix(y_gt, y_pred)
  kappa = metr.cohen_kappa_score(y_gt, y_pred)
  OA = metr.accuracy_score(y_gt, y_pred)
  UA = metr.precision_score(y_gt, y_pred, average=None, zero_division=0)
  PA = metr.recall_score(y_gt, y_pred, average=None, zero_division=0)
      
  # Handle cases where a label is present in training but not in test or predictions
  te = set(y_gt)
  pr = set(y_pred)
  tr = set(unique_classes_train)
  te_pr = te.union(pr)
  train_minus_pred_or_test = list(tr.difference(te_pr))
  if len(train_minus_pred_or_test) > 0:
      # Create a new CM, inserting zeros for indices in train_minus_pred_or_test 
      classes = len(labels)
      cm_new = np.zeros((classes, classes))
      r = 0
      for i in range(classes):
          c = 0
          if i in train_minus_pred_or_test:
              r -= 1
          for j in range(classes):
              if j in train_minus_pred_or_test:
                  c -= 1
              else:
                  if i not in train_minus_pred_or_test:
                      cm_new[i, j] = cm[i+r, j+c]
      cm = cm_new
      # Create new PA, UA, inserting NaN for indices in train_minus_pred_or_test
      new_UA = np.zeros((classes,))
      new_PA = np.zeros((classes,))
      for i in range(classes):
          if i in train_minus_pred_or_test:
              new_UA[i] = np.nan
              new_PA[i] = np.nan
              r -= 1
          else:
              new_UA[i] = UA[i+r]
              new_PA[i] = PA[i+r]
      UA = new_UA
      PA = new_PA        
      
  # Handle UA and PA values that are not computable
  if labels_present_only_in_one.size > 0:
      UA[labels_present_only_in_one] = np.nan
      PA[labels_present_only_in_one] = np.nan

  # Confusion matrix with UA, PA
  rows, cols = cm.shape
  cm_with_stats = np.zeros((rows+2, cols+2))
  cm_with_stats[0:-2, 0:-2] = cm
  cm_with_stats[-1, 0:-2] = np.round(100*UA, 2)
  cm_with_stats[0:-2, -1] = np.round(100*PA, 2)
  cm_with_stats[-2, 0:-2] = np.sum(cm, axis=0)
  cm_with_stats[0:-2, -2] = np.sum(cm, axis=1)

  # Convert to list
  cm_list = cm_with_stats.tolist()
  
  # First row
  first_row = []
  first_row.extend(labels)
  first_row.append('sum')
  first_row.append('PA')

  # First col
  first_col = []
  first_col.extend(labels)
  first_col.append('sum')
  first_col.append('UA')

  # Fill rest of the text
  idx = 0
  for sublist in cm_list:
      if idx == rows:
          cm_list[idx] = sublist
          sublist[-2] = 'kappa:'
          sublist[-1] = round(100*kappa, 2)
      elif idx == rows+1:
          sublist[-2] = 'OA:'
          sublist[-1] = round(100*OA, 2)
          cm_list[idx] = sublist
      idx += 1
  
  # Convert to data frame
  df = pd.DataFrame(np.array(cm_list))
  df.columns = first_row
  df.index = first_col

  # Write to xls
  writer = pd.ExcelWriter(fullpath, engine='xlsxwriter', options={'strings_to_numbers': True})
  df.to_excel(writer, 'Sheet 1')
  writer.save()

  return df

def tile_callback(option, opt, value, parser):
  """
  Callback function to handle comma-separated list options.
  """
  setattr(parser.values, option.dest, value.split(','))

def glob_and_check(path, pattern):
  """
  Find files matching a pattern in a directory and ensure only one file matches.

  Parameters:
  - path: str, the directory path.
  - pattern: str, the glob pattern to match files.

  Returns:
  - output: str, the matched file path.
  """
  os.chdir(path)
  files = glob(pattern)
  num_files = len(files)
  if num_files > 1:
      print(files)
      print('All of the above files match the input pattern. Please use a more specific pattern or rearrange files in different directories.')
      sys.exit(-1)
  elif num_files == 1:
      output = files[0]
      print('Accessing', output)
      return output

def loadFTLV_100(tile_names, features_path, feature_type):
  """
  Load feature tables and label vectors for given tile names.

  Parameters:
  - tile_names: list of str, names of the tiles.
  - features_path: str, path to the features directory.
  - feature_type: str, type of feature to load.

  Returns:
  - ft: numpy array, the feature table.
  - lv: numpy array, the label vector.
  """
  os.chdir(features_path)
  for t, tile_name in enumerate(tile_names):
      # Load Feature Tables and Labels Vector
      current_train_FT = glob_and_check(features_path, tile_name+'*'+feature_type+'_train_FT.npy')
      current_test_FT = glob_and_check(features_path, tile_name+'*'+feature_type+'_test_FT.npy')
      current_train_LV = glob_and_check(features_path, tile_name+'*_train_LV.npy')
      current_test_LV = glob_and_check(features_path, tile_name+'*_test_LV.npy')
      if t == 0:
          ft = np.load(current_train_FT)
          ft = np.vstack((ft, np.load(current_test_FT)))
          lv = np.load(current_train_LV).astype(np.uint8)
          lv = np.hstack((lv, np.load(current_test_LV).astype(np.uint8)))
      else:
          ft = np.vstack((ft, np.load(current_train_FT)))
          ft = np.vstack((ft, np.load(current_test_FT)))
          lv = np.hstack((lv, np.load(current_train_LV).astype(np.uint8)))
          lv = np.hstack((lv, np.load(current_test_LV).astype(np.uint8)))
      
      print(lv.shape)

  return ft, lv

if len(sys.argv) == 1:
  prog = os.path.basename(sys.argv[0])
  print(sys.argv[0] + ' [options]')
  print("Run: python3 ", prog, " --help")
  sys.exit(-1)
else:
  usage = "usage: %prog [options]"
  parser = OptionParser(usage=usage)
  # RandomForest parameters
  # -K, the number of trees; default 100
  # -m, the number of features randomly selected at each node; default sqrt(n_features)
  # --max_depth, the maximal depth of each tree; default none
  # --min_samples, the minimal number of samples per node; default 1
  parser.add_option("-K", dest="rf_n_estimators", action="store", type="int", help="the number of trees", default=100)
  parser.add_option("-m", dest="rf_n_features", action="store", type="string", help="the strategy for features randomly selected at each node", default="auto")
  parser.add_option("--max-depth", dest="rf_max_depth", action="store", type="int", help="the maximal depth of each tree; 0 means None", default=0)
  parser.add_option("--min-samples", dest="rf_min_samples_leaf", action="store", type="int", help="minimal number of samples per node", default=1)
  # Building parameters
  parser.add_option("-t", dest="train_tile_names", action="callback", type="string", help="Sentinel-2 tile name/s used for training the model (in format NNCCC e.g. 34SEG). If multiple separate by comma. All the reference data for this tile will be used for training.", default=None, callback=tile_callback) 
  parser.add_option("-v", dest="validation_tile_names", action="callback", type="string", help="Sentinel-2 tile name/s used for testing the model (in format NNCCC e.g. 34SEG). If multiple separate by comma. All the reference data for this tile will be used for testing", default=None, callback=tile_callback) 
  parser.add_option("-f", dest="feature_type", action="store", type="string", help="Feature type(ts,tf).", default=None)
  parser.add_option("-n", dest="nomenclature_xls_fullpath", action="store", type="string", help="Path to .xls file containing classes' nomenclature. Label names will be taken from 'Abbreviation' column header by default. Otherwise, set parameter -h to the desired column header", default=None)
  parser.add_option("-o", dest="output_path", action="store", type="string", help="Output path for this experiment's results. Will be created if it does not exist. Features will be saved in 'features' dir.", default=None)
  # Optional
  parser.add_option("-a", dest="auxiliary_features_labels", action="callback", type="string", help="Optional. Auxiliary feature labels to load from features directory. If multiple separate by comma. Will be stacked at the end of the Feature Tables, before training/testing.", default=None, callback=tile_callback) 
  parser.add_option("-l", dest="nomenclature_labels_header", action="store", type="string", help="Optional. Header for column containing classes' nomenclature. Default value: Abbreviation", default='Abbreviation')
  parser.add_option("-c", dest="classifier_pkl_filename", action="store", type="string", help="Optional. Classifier exact .pkl filename, must be present in 'models' directory of output_path. If not provided a new classifier will be trained. ", default=None)
  parser.add_option("--output-train-tiles-name", dest="output_train_tile_names", action="store", type="string", help="Optional. Normally the train tiles are concatenated by an underscore and become part of the output filenames. To override the train tiles set an alternative string using this option. ", default=None)
  (options, args) = parser.parse_args()
  # Checking required arguments for the script
  parser.check_required("-f")
  parser.check_required("-n")
  parser.check_required("-o")
  parser.check_required("-t")
  parser.check_required("-v")

t_start = time.time()

# Handle Input
output_path = options.output_path
feature_type = options.feature_type
nomenclature_xls_fullpath = options.nomenclature_xls_fullpath
nomenclature_labels_header = options.nomenclature_labels_header
classifier_pkl_filename = options.classifier_pkl_filename
train_tile_names = options.train_tile_names
output_train_tile_names = [options.output_train_tile_names] if options.output_train_tile_names else train_tile_names
auxiliary_features_labels = options.auxiliary_features_labels
rf_max_depth = None if options.rf_max_depth == 0 else options.rf_max_depth

test_tile_names = options.validation_tile_names

savename_list = ['K', options.rf_n_estimators, 'maxD', rf_max_depth, 'minS', options.rf_min_samples_leaf, 'tr100'] + output_train_tile_names + ['te100'] + test_tile_names
savename_list_clf = ['K', options.rf_n_estimators, 'maxD', rf_max_depth, 'minS', options.rf_min_samples_leaf, 'tr100'] + output_train_tile_names
if auxiliary_features_labels != None:
  savename_list += ['aux'] + auxiliary_features_labels
  savename_list_clf += ['aux'] + auxiliary_features_labels
savename_list += [feature_type, 'RF.xlsx']
savename_list_clf += [feature_type, 'RF.pkl']

cm_name = '_'.join(str(x) for x in savename_list)
classifier_filename = '_'.join(str(x) for x in savename_list_clf)

print(f"cm_name will be: {cm_name}")
if not classifier_pkl_filename:
  print(f"classifier_filename will be: {classifier_filename}")

# Set directories (make if non-existent)
features_path = os.path.join(output_path, 'features')
models_path = os.path.join(output_path, 'models')
results_path = os.path.join(output_path, 'results')
if not os.path.exists(models_path):
  os.mkdir(models_path)
if not os.path.exists(results_path):
  os.mkdir(results_path)

# Get features
train_FT, train_LV = loadFTLV_100(train_tile_names, features_path, feature_type)
test_FT, test_LV = loadFTLV_100(test_tile_names, features_path, feature_type)
test_LV = test_LV.astype(np.uint8)

# Get auxiliary features
if auxiliary_features_labels != None:
  for aux_feature in auxiliary_features_labels:
      aux_train_FT, _ = loadFTLV_100(train_tile_names, features_path, aux_feature)
      train_FT = np.hstack((train_FT, aux_train_FT))
      aux_test_FT, _ = loadFTLV_100(test_tile_names, features_path, aux_feature)
      test_FT = np.hstack((test_FT, aux_test_FT))

print(train_FT.shape)
print(test_FT.shape)
print(train_LV.shape)
print(test_LV.shape)

# Get class label names
df = pd.read_excel(nomenclature_xls_fullpath, sheet_name='Sheet1')  # Import as dataframe
try:
  label_names = df[nomenclature_labels_header].to_numpy()  # Get label names from Abbreviation column
except Exception as e:
  print(e)
  print('Please make sure the class label names are under the header', nomenclature_labels_header)
unique_classes = np.unique(np.concatenate((train_LV, test_LV)))  # unique class labels in data
label_names = label_names[unique_classes].tolist()  # subset label names
classes = len(unique_classes)  # Number of classes
print('Number of classes is:', classes)

# Create/Load Random Forest classifier object
os.chdir(models_path)
if classifier_pkl_filename == None:
  print('Initializing new classifier')
  n_jobs = multiprocessing.cpu_count() - 1
  clf_RF = ensemble.RandomForestClassifier(
      random_state=1,
      n_jobs=n_jobs,
      n_estimators=options.rf_n_estimators,
      max_features=options.rf_n_features,
      max_depth=rf_max_depth,
      min_samples_leaf=options.rf_min_samples_leaf,
  )
  print('... training Random Forest Classifier with', train_FT.shape[0], 'pixels')
  t0 = time.time()
  clf_RF.fit(train_FT, train_LV)  # Train Random Forest classifier
  t1 = time.time()
  print('Training completed in:', convert(n_jobs*(t1-t0)), 'core hours')
  print('Saving classifier')
  joblib.dump(clf_RF, classifier_filename)  # Save classifier to a file
else:
  try:
      print('Loading classifier')
      clf_RF = joblib.load(classifier_pkl_filename)
  except Exception as e:
      print(e)
      print('Model not already trained, or classifier .pkl file not found.')

print('... testing on', test_FT.shape[0], 'pixels')
predictions = clf_RF.predict(test_FT)  # Validate Random Forest Classifier
errors = np.where(predictions != test_LV)
error = float(errors[0].shape[0]) / test_LV.shape[0]
print('... Random Forest accuracy:  %2.2f' % (100*(1-error)), '%')

# Calculate set of labels present in exactly one of train or test dataset
# Union minus Intersection of sets -> Symmetric Difference
a = set(test_LV)
b = set(train_LV)
labels_present_only_in_one = np.array(list(a.symmetric_difference(b)))

# Save Confusion Matrix
os.chdir(results_path)
cm = confusion_matrix(test_LV, predictions, cm_name, label_names, labels_present_only_in_one, np.unique(train_LV))

t_end = time.time()

if classifier_pkl_filename == None:
  t_multi = n_jobs*(t1-t0)
  t_single = t_end - t_start - (t1-t0)
  print('Other processes completed in:', convert(t_single), 'core hours')
  print('Overall processing time:', convert(t_multi+t_single), 'core hours')
