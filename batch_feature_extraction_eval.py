# Extracts the features, labels, and normalizes the development and evaluation split features.
# NOTE: Change the dataset_dir and feat_label_dir path accordingly

import cls_feature_class

process_str = 'eval'  # 'dev' or 'eval' will extract features for the respective set accordingly
#  'dev, eval' will extract features of both sets together

dataset_name = 'mic'  # 'foa' -ambisonic or 'mic' - microphone signals
dataset_dir = 'audio_dataset/'   # Base folder containing the foa/mic and metadata folders
feat_label_dir = 'feature_spectrum_generated/'  # Directory to dump extracted features and labels

# -----------------------------Extract ONLY features for evaluation set-----------------------------
def batch_feature_extraction_eval():
    eval_feat_cls = cls_feature_class.FeatureClass(dataset=dataset_name, dataset_dir=dataset_dir,
                                                       feat_label_dir=feat_label_dir, is_eval=True)

    # Extract features and normalize them
    eval_feat_cls.extract_all_feature()
    eval_feat_cls.preprocess_features()

