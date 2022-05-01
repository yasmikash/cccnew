import os
import shutil

import numpy as np
import matplotlib.pyplot as plot
import cls_feature_class
import cls_data_generator
import evaluation_metrics
from keras.models import load_model
import parameter
import batch_feature_extraction_dev
import batch_feature_extraction_eval
import preprocess_cough_rate_audio

plot.switch_backend('agg')
params = parameter.get_params('3')

def predict():
    preprocess_cough_rate_audio.preprocess()
    shutil.copy('audio_dataset/mic_dev/split0_1.wav', 'audio_dataset/mic_eval/split0_1.wav')

    batch_feature_extraction_dev.batch_feature_extraction_dev()
    batch_feature_extraction_eval.batch_feature_extraction_eval()

    print('Loading testing dataset:')
    data_gen_test = cls_data_generator.DataGenerator(
        dataset=params['dataset'], split=0, batch_size=params['batch_size'], seq_len=params['sequence_length'],
        feat_label_dir=params['feat_label_dir'], shuffle=False, per_file=params['dcase_output'],
        is_eval=True if params['mode'] is 'eval' else False
    )

    print('\nLoading the best model and predicting results on the testing split')
    model = load_model('models/final_cough_event_model.h5')
    pred_test = model.predict_generator(
        generator=data_gen_test.generate(),
        steps=1,
        verbose=2
    )

    test_sed_pred = evaluation_metrics.reshape_3Dto2D(pred_test[0]) > 0.5
    test_doa_pred = evaluation_metrics.reshape_3Dto2D(pred_test[1])

    # rescaling the elevation data from [-180 180] to [-def_elevation def_elevation] for scoring purpose
    test_doa_pred[:, 4:] = test_doa_pred[:, 4:] / (180. / 50)


    # Dump results in DCASE output format for calculating final scores
    dcase_dump_folder = os.path.join(params['dcase_dir'])
    cls_feature_class.create_folder(dcase_dump_folder)
    print('Dumping recording-wise results in: {}'.format(dcase_dump_folder))

    test_filelist = data_gen_test.get_filelist()
    # Number of frames for a 60 second audio with 20ms hop length = 3000 frames
    max_frames_with_content = data_gen_test.get_nb_frames()

    # Number of frames in one batch (batch_size* sequence_length) consists of all the 3000 frames above with
    # zero padding in the remaining frames
    frames_per_file = data_gen_test.get_frame_per_file()

    for file_cnt in range(test_sed_pred.shape[0] // frames_per_file):
        output_file = os.path.join(dcase_dump_folder, test_filelist[file_cnt].replace('.npy', '.csv'))
        dc = file_cnt * frames_per_file
        output_dict = evaluation_metrics.regression_label_format_to_output_format(
            data_gen_test,
            test_sed_pred[dc:dc + max_frames_with_content, :],
            test_doa_pred[dc:dc + max_frames_with_content, :] * 180 / np.pi
        )
        evaluation_metrics.write_output_format_file(output_file, output_dict)











