import os
from pathlib import Path
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify
from flask_cors import CORS
import cough_rate_calculation
import predict

# define a flask app
app = Flask(__name__)
CORS(app)

@app.route('/cough', methods=['POST'])
def cough():
    # get the audio file from post request
    f = request.files['file']

    # save the file to location
    base_path = os.path.dirname(__file__)

    file_name = Path("audio_dataset/split0_1.wav")

    if file_name.exists():
        os.remove('audio_dataset/split0_1.wav')

    file_path = os.path.join(base_path, 'audio_dataset', secure_filename(f.filename))
    f.save(file_path)

    predict.predict()
    x = cough_rate_calculation.cough_rate_calculation()

    os.remove('audio_dataset/mic_dev/split0_1.wav')
    os.remove('audio_dataset/mic_eval/split0_1.wav')
    os.remove('feature_spectrum_generated/mic_dev/split0_1.npy')
    os.remove('feature_spectrum_generated/mic_dev_norm/split0_1.npy')
    os.remove('feature_spectrum_generated/mic_eval/split0_1.npy')
    os.remove('feature_spectrum_generated/mic_eval_norm/split0_1.npy')
    os.remove('feature_spectrum_generated/mic_wts')
    os.remove('results/split0_1.csv')
    os.remove('audio_dataset/split0_1.wav')

    if (x > 0.0):
        isCough = 1
    else:
        isCough = 0
    return jsonify(cough_rate=x, isCough=isCough)

if __name__ == '__main__':
    app.run(debug=True)
