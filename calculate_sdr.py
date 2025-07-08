import argparse
import csv
import os

from torch import randn
from torchmetrics.audio import SignalDistortionRatio
from torchmetrics.functional.audio import signal_distortion_ratio
import torchaudio
import matplotlib.pyplot as plt


def calculate_sdr(audio_predicted, audio_target):
    preds_waveform, sample_rate_preds = torchaudio.load(audio_predicted)
    target_waveform, sample_rate_target = torchaudio.load(audio_target)

    min_len = min(preds_waveform.shape[-1], target_waveform.shape[-1])
    preds_waveform = preds_waveform[..., :min_len]
    target_waveform = target_waveform[..., :min_len]

    sdr_metric = SignalDistortionRatio()
    sdr_value_module = sdr_metric(preds_waveform, target_waveform)
    print(f"SDR (Module): {sdr_value_module.item()}")

    # sdr_value_functional = signal_distortion_ratio(preds_waveform, target_waveform)
    # print(f"SDR (Functional): {sdr_value_functional}")

    return float(sdr_value_module.cpu().numpy().astype(float))


def get_sdr_of_generated_folder(source_path):
    headers = [
        'Name Song', 'Bass SDR', 'Vocals SDR', 'Others SDR', 'Drums SDR'
    ]
    data = [headers, ]
    for index, dirname in enumerate(os.listdir(source_path)):
        song_folder = os.path.join(source_path, dirname)
        bass_compare = calculate_sdr(
            os.path.join(song_folder, "bass.wav"), os.path.join(song_folder, "predicted_bass.wav")
        )
        vocals_compare = calculate_sdr(
            os.path.join(song_folder, "vocals.wav"), os.path.join(song_folder, "predicted_vocals.wav")
        )
        other_compare = calculate_sdr(
            os.path.join(song_folder, "other.wav"), os.path.join(song_folder, "predicted_other.wav")
        )
        drums_compare = calculate_sdr(
            os.path.join(song_folder, "drums.wav"), os.path.join(song_folder, "predicted_drums.wav")
        )
        data.append([os.path.basename(song_folder), bass_compare, vocals_compare, other_compare, drums_compare])

    with open('benchmark_musdb_hq.csv', 'w', newline='') as csvfile:
        # Create a csv.writer object
        writer = csv.writer(csvfile)

        # Write all rows from the list of lists
        writer.writerows(data)


if __name__ == "__main__":
    m = argparse.ArgumentParser()
    m.add_argument("--benchmark-dir", "-i", type=str,  required=True)

    options = m.parse_args().__dict__
    get_sdr_of_generated_folder(options['benchmark_dir'])