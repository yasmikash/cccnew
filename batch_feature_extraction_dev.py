# Extracts the features, labels, and normalizes the development and evaluation split features.
# NOTE: Change the dataset_dir and feat_label_dir path accordingly

import cls_feature_class

process_str = 'dev'  # 'dev' or 'eval' will extract features for the respective set accordingly
#  'dev, eval' will extract features of both sets together

dataset_name = 'mic'  # 'foa' -ambisonic or 'mic' - microphone signals
dataset_dir = 'audio_dataset/'   # Base folder containing the foa/mic and metadata folders
feat_label_dir = 'feature_spectrum_generated/'  # Directory to dump extracted features and labels


def batch_feature_extraction_dev():
    if 'dev' in process_str:
        # -------------- Extract features and labels for development set -----------------------------
        dev_feat_cls = cls_feature_class.FeatureClass(dataset=dataset_name, dataset_dir=dataset_dir,
                                                      feat_label_dir=feat_label_dir)

        # Extract features and normalize them
        dev_feat_cls.extract_all_feature()
        dev_feat_cls.preprocess_features()





