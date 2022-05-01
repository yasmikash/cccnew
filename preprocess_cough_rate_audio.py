import sox

def preprocess():
    tfm = sox.Transformer()
    tfm.set_output_format(channels=4)
    tfm.build(
    input_filepath='audio_dataset/split0_1.wav',
    output_filepath='audio_dataset/mic_dev/split0_1.wav')


