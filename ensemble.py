import shutil

if __name__ == '__main__':
    import os

    gpu_use = "0"
    print('GPU use: {}'.format(gpu_use))
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


import numpy as np
import torch
import torch.nn as nn
import os
import argparse
import soundfile as sf

from demucs.states import load_model
from demucs import pretrained
from demucs.apply import apply_model
import onnxruntime as ort
from time import time
import librosa
import hashlib

from vocal_models_utils import get_models, demix_full


class EnsembleMusicSeparationModel(object):
    @property
    def instruments(self):
        return ['bass', 'drums', 'other']

    def __init__(self, options):
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        self.kim_model_1 = False
        self.input_audio = options.get('input_audio')
        self.sample_rate = options.get('sample_rate')
        self.output_folder = options.get('output_folder')
        # Define Overlaps
        self.overlap_large = 0.6
        self.overlap_small = 0.5
        model_folder = os.path.dirname(os.path.realpath(__file__)) + '/models/'
        remote_url = 'https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/04573f0d-f3cf25b2.th'
        self.model_path = model_folder + '04573f0d-f3cf25b2.th'
        if not os.path.isfile(self.model_path):
            torch.hub.download_url_to_file(remote_url, model_folder + '04573f0d-f3cf25b2.th')

        # Getting the models and initialize the weights respectively
        self.models = []
        self.weights_vocals = np.array([10, 1, 8, 9])
        self.weights_bass = np.array([19, 4, 5, 8])
        self.weights_drums = np.array([18, 2, 4, 9])
        self.weights_other = np.array([14, 2, 5, 10])

        model1 = pretrained.get_model('htdemucs_ft')
        model1.to(device)
        self.models.append(model1)

        model2 = pretrained.get_model('htdemucs')
        model2.to(device)
        self.models.append(model2)

        model3 = pretrained.get_model('htdemucs_6s')
        model3.to(device)
        self.models.append(model3)

        model4 = pretrained.get_model('hdemucs_mmi')
        model4.to(device)
        self.models.append(model4)

        # With this chunk_size it's possible run over a 8gb of vRam, use 1M with >8gb vRam
        self.chunk_size = 200000
        providers = ["CUDAExecutionProvider"]

        # Load Kim Model for vocal, creating InferenceSession from ONNX
        self.kim_vocal_model = get_models('tdf_extra', load=False, device=device, vocals_model_type=2)
        model_path_onnx1 = model_folder + 'Kim_Vocal_2.onnx'
        remote_url_onnx1 = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/Kim_Vocal_2.onnx'
        if not os.path.isfile(model_path_onnx1):
            torch.hub.download_url_to_file(remote_url_onnx1, model_path_onnx1)
        self.infer_session1 = ort.InferenceSession(
            model_path_onnx1,
            providers=providers,
            provider_options=[{"device_id": 0}],
        )

        # Load Kim Model for get instrumentals, same way that first kim model
        self.kim_instrumental_model = get_models('tdf_extra', load=False, device=device, vocals_model_type=2)
        model_path_onnx2 = model_folder + 'Kim_Inst.onnx'
        remote_url_onnx2 = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/Kim_Inst.onnx'
        if not os.path.isfile(model_path_onnx2):
            torch.hub.download_url_to_file(remote_url_onnx2, model_path_onnx2)
        self.infer_session2 = ort.InferenceSession(
            model_path_onnx2,
            providers=providers,
            provider_options=[{"device_id": 0}],
        )

        self.device = device

    def run_vocals_separation(self, mixed_sound_array, write_audio=True):
        audio = np.expand_dims(mixed_sound_array.T, axis=0)
        audio = torch.from_numpy(audio).type('torch.FloatTensor').to(self.device)

        overlap_large = self.overlap_large
        overlap_small = self.overlap_small

        # First Use Demucs
        model_vocals_demucs = load_model(self.model_path)
        model_vocals_demucs.to(self.device)

        shifts = 1
        overlap = overlap_large
        vocals_demucs = 0.5 * apply_model(model_vocals_demucs, audio, shifts=shifts, overlap=overlap)[0][3].cpu().numpy()

        vocals_demucs += 0.5 * -apply_model(model_vocals_demucs, -audio, shifts=shifts, overlap=overlap)[0][3].cpu().numpy()
        model_vocals_demucs = model_vocals_demucs.cpu()
        del model_vocals_demucs

        overlap = overlap_large
        kim_only_vocals = demix_full(
            mixed_sound_array.T,
            self.device,
            self.chunk_size,
            self.kim_vocal_model,
            self.infer_session1,
            overlap=overlap
        )[0]
        del self.infer_session1
        del self.kim_vocal_model

        kim_only_instrumental = -demix_full(
            -mixed_sound_array.T,
            self.device,
            self.chunk_size,
            self.kim_instrumental_model,
            self.infer_session2,
            overlap=overlap
        )[0]
        del self.infer_session2
        del self.kim_instrumental_model

        # it's only instrumental so we need to invert
        vocals_kim_instrumental = mixed_sound_array.T - kim_only_instrumental

        weights = np.array([12, 8, 3])
        vocals = (weights[0] * kim_only_vocals.T + weights[1] * vocals_kim_instrumental.T + weights[
            2] * vocals_demucs.T) / weights.sum()

        if write_audio:
            output_name = 'predicted_vocals.wav'
            sf.write(self.output_folder + '/' + output_name, vocals, self.sample_rate, subtype='FLOAT')
            print('File created: {}'.format(self.output_folder + '/' + output_name))

        return vocals

    def run_instrumental_separation(self, mixed_sound_array, vocals_sound_array, write_audio=True):
        instrumental_sound_array = mixed_sound_array - vocals_sound_array
        audio = np.expand_dims(instrumental_sound_array.T, axis=0)
        audio = torch.from_numpy(audio).type('torch.FloatTensor').to(self.device)
        shifts = 1

        all_outs = []
        for index, model in enumerate(self.models):
            overlap = self.overlap_large
            # In htdemucs_ft model use overlap small
            if index == 0:
                overlap = self.overlap_small
            out = (0.5 * apply_model(model, audio, shifts=shifts, overlap=overlap)[0].cpu().numpy() +
                   0.5 * -apply_model(model, -audio, shifts=shifts, overlap=overlap)[0].cpu().numpy())
            # htdemucs_6s have guitar,piano in the last results of vector, it's necessary sum to get results
            if index == 2:
                out[2] = out[2] + out[4] + out[5]
                out = out[:4]

            out[0] = self.weights_drums[index] * out[0]
            out[1] = self.weights_bass[index] * out[1]
            out[2] = self.weights_other[index] * out[2]
            out[3] = self.weights_vocals[index] * out[3]
            all_outs.append(out)
            model_cpu = model.cpu()
            del model_cpu

        out = np.array(all_outs).sum(axis=0)
        out[0] = out[0] / self.weights_drums.sum()
        out[1] = out[1] / self.weights_bass.sum()
        out[2] = out[2] / self.weights_other.sum()
        out[3] = out[3] / self.weights_vocals.sum()

        # other
        res = mixed_sound_array - vocals_sound_array - out[0].T - out[1].T
        res = np.clip(res, -1, 1)
        other = (2 * res + out[2].T) / 3.0

        # drums
        res = mixed_sound_array - vocals_sound_array - out[1].T - out[2].T
        res = np.clip(res, -1, 1)
        drums = (res + 2 * out[0].T.copy()) / 3.0

        # bass
        res = mixed_sound_array - vocals_sound_array - out[0].T - out[2].T
        res = np.clip(res, -1, 1)
        bass = (res + 2 * out[1].T) / 3.0

        separated_arrays = {}
        separated_arrays['other'] = mixed_sound_array - vocals_sound_array - bass - drums
        separated_arrays['drums'] = mixed_sound_array - vocals_sound_array - bass - other
        separated_arrays['bass'] = mixed_sound_array - vocals_sound_array - drums - other

        if write_audio:
            for instrument in self.instruments:
                output_name = f"predicted_{instrument}.wav"
                sf.write(self.output_folder + '/' + output_name, separated_arrays[instrument], self.sample_rate, subtype='FLOAT')
                print('File created: {}'.format(self.output_folder + '/' + output_name))

        return separated_arrays

def execute_ensemble(options):
    for input_audio in options['input_audio']:
        if not os.path.isdir(input_audio):
            print('Error. No such directory: {}. Please check path!'.format(input_audio))
            return
    output_folder = options['output_folder']
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    for filename in os.listdir(options['input_audio'][0]):
        source_path = os.path.join(options['input_audio'][0], filename)
        if os.path.isfile(source_path):
            shutil.copy2(source_path, output_folder)

    only_vocals = False
    if 'only_vocals' in options:
        if options['only_vocals'] is True:
            print('Generate only vocals')
            only_vocals = True

    options['input_audio'] = options['input_audio'][0]
    options['output_folder'] = output_folder
    mixed_sound_array, options['sample_rate'] = librosa.load(f"{options['input_audio']}/mixture.mp3", mono=False, sr=44100)

    if len(mixed_sound_array.shape) == 1:
        mixed_sound_array = np.stack([mixed_sound_array, mixed_sound_array], axis=0)
    ensemble_class = EnsembleMusicSeparationModel(options)
    vocals = ensemble_class.run_vocals_separation(mixed_sound_array.T)
    instrumental_separated = ensemble_class.run_instrumental_separation(mixed_sound_array.T, vocals)


def run_benchmark(source_path, output_folder):
    for index, dirname in enumerate(os.listdir(source_path[0])):
        output_folder_i = os.path.join(output_folder, dirname)
        options = dict(
            only_vocals=False,
            output_folder=output_folder_i,
            input_audio=[os.path.join(source_path[0], dirname)]
        )
        if os.path.exists(output_folder_i):
            print(f"Already exists: {output_folder_i}")
            continue

        execute_ensemble(options)



if __name__ == '__main__':
    start_time = time()
    m = argparse.ArgumentParser()
    m.add_argument("--input_audio", "-i", nargs='+', type=str, required=False)
    m.add_argument("--output_folder", "-r", type=str,  required=True)
    m.add_argument("--only_vocals", action='store_true')
    m.add_argument("--benchmark_input_audio", "-bi", nargs='+', type=str, required=False)

    options = m.parse_args().__dict__
    print(options)
    if options.get('benchmark_input_audio') is not None:
        run_benchmark(options['benchmark_input_audio'], options['output_folder'])
    else:
        execute_ensemble(options)
    print('Time: {:.0f} sec'.format(time() - start_time))
