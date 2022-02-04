# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# MIT License

# Copyright (c) 2019 Jeongmin Liu

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import collections as py_collections
import json
import logging
import math
import re
import os
import pickle
import random
import shutil
import sys
from os.path import expanduser
from pathlib import Path
from typing import Any, Dict, Optional, Union

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from scipy.stats import betabinom
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn

from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.tts.helpers.helpers import dynamic_range_compression, dynamic_range_decompression, window_sumsquare
from nemo.collections.tts.helpers.text import text_to_sequence, cmudict, _clean_text, get_arpabet
from nemo.collections.common.parts.preprocessing import collections, parsers
from nemo.core.classes import Dataset
from nemo.core.neural_types.elements import *
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils import logging

DataDict = Dict[str, Any]


class AudioDataset(Dataset):
    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            "audio_signal": NeuralType(("B", "T"), AudioSignal()),
            "a_sig_length": NeuralType(tuple("B"), LengthsType()),
        }

    def __init__(
        self,
        manifest_filepath: Union[str, "pathlib.Path"],
        n_segments: int,
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        trim: Optional[bool] = False,
        truncate_to: Optional[int] = 1,
    ):
        """
        Mostly compliant with nemo.collections.asr.data.datalayers.AudioToTextDataset except it only returns Audio
        without text. Dataset that loads tensors via a json file containing paths to audio files, transcripts, and
        durations (in seconds). Each new line is a different sample. Note that text is required, but is ignored for
        AudioDataset. Example below:
        {"audio_filepath": "/path/to/audio.wav", "text_filepath":
        "/path/to/audio.txt", "duration": 23.147}
        ...
        {"audio_filepath": "/path/to/audio.wav", "text": "the
        transcription", "offset": 301.75, "duration": 0.82, "utt":
        "utterance_id", "ctm_utt": "en_4156", "side": "A"}
        Args:
            manifest_filepath (str, Path): Path to manifest json as described above. Can be comma-separated paths
                such as "train_1.json,train_2.json" which is treated as two separate json files.
            n_segments (int): The length of audio in samples to load. For example, given a sample rate of 16kHz, and
                n_segments=16000, a random 1 second section of audio from the clip will be loaded. The section will
                be randomly sampled everytime the audio is batched. Can be set to -1 to load the entire audio.
            max_duration (float): If audio exceeds this length in seconds, it is filtered from the dataset.
                Defaults to None, which does not filter any audio.
            min_duration(float): If audio is less than this length in seconds, it is filtered from the dataset.
                Defaults to None, which does not filter any audio.
            trim (bool): Whether to use librosa.effects.trim on the audio clip
            truncate_to (int): Ensures that the audio segment returned is a multiple of truncate_to.
                Defaults to 1, which does no truncating.
        """

        self.collection = collections.ASRAudioText(
            manifests_files=manifest_filepath.split(","),
            parser=parsers.make_parser(),
            min_duration=min_duration,
            max_duration=max_duration,
        )
        self.trim = trim
        self.n_segments = n_segments
        self.truncate_to = truncate_to

    def _collate_fn(self, batch):
        """
        Takes a batch: a lists of length batch_size, defined in the dataloader. Returns 2 padded and batched
        tensors corresponding to the audio and audio_length.
        """

        def find_max_len(seq, index):
            max_len = -1
            for item in seq:
                if item[index].size(0) > max_len:
                    max_len = item[index].size(0)
            return max_len

        batch_size = len(batch)

        audio_signal, audio_lengths = None, None
        if batch[0][0] is not None:
            if self.n_segments > 0:
                max_audio_len = self.n_segments
            else:
                max_audio_len = find_max_len(batch, 0)

            audio_signal = torch.zeros(batch_size, max_audio_len, dtype=torch.float)
            audio_lengths = []
            for i, sample in enumerate(batch):
                audio_signal[i].narrow(0, 0, sample[0].size(0)).copy_(sample[0])
                audio_lengths.append(sample[1])
            audio_lengths = torch.tensor(audio_lengths, dtype=torch.long)

        return audio_signal, audio_lengths

    def __getitem__(self, index):
        """
        Given a index, returns audio and audio_length of the corresponding element. Audio clips of n_segments are
        randomly chosen if the audio is longer than n_segments.
        """
        example = self.collection[index]
        features = AudioSegment.segment_from_file(example.audio_file, n_segments=self.n_segments, trim=self.trim,)
        features = torch.tensor(features.samples)
        audio, audio_length = features, torch.tensor(features.shape[0]).long()

        truncate = audio_length % self.truncate_to
        if truncate != 0:
            audio_length -= truncate.long()
            audio = audio[:audio_length]

        return audio, audio_length

    def __len__(self):
        return len(self.collection)


class MelAudioDataset(Dataset):
    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            "audio_signal": NeuralType(("B", "T"), AudioSignal()),
            "audio_length": NeuralType(tuple("B"), LengthsType()),
            "melspec": NeuralType(("B", "D", "T"), MelSpectrogramType()),
        }

    def __init__(
        self,
        manifest_filepath: Union[str, "pathlib.Path"],
        n_segments: int,
        max_duration: Optional[float] = float("inf"),
        min_duration: Optional[float] = 0,
        mel_hop_size=256,
        mel_load_func=np.load,
    ):
        """
        `manifest_filepath` should point to a file with the following structure:
        {"audio_filepath": "/path/to/audio.wav", "mel_filepath": "/path/to/mel.npy", "duration": 23.147}
        where `mel.npy` is a pre-computed tensor of a melspectrogram.
        this dataset class is used for fine-tuning vocoders with pre-computed mel-spectrograms.
        """
        self.collection = [json.loads(line) for line in open(manifest_filepath, "r")]
        self.collection = list(filter(lambda item: min_duration <= item["duration"] <= max_duration, self.collection))
        self.n_segments = n_segments
        self.mel_hop_size = mel_hop_size
        self.mel_load_func = mel_load_func

    def __getitem__(self, index):
        """
        Given a index, returns audio and audio_length of the corresponding element. Audio clips of n_segments are
        randomly chosen if the audio is longer than n_segments.
        """
        example = self.collection[index]
        audio_file = example["audio_filepath"]
        mel_file = example["mel_filepath"]

        audio, sr = sf.read(audio_file)
        audio = torch.FloatTensor(audio).unsqueeze(0)
        mel = self.mel_load_func(mel_file)

        frames = math.ceil(self.n_segments / self.mel_hop_size)
        if audio.shape[1] > self.n_segments:
            start = random.randint(0, mel.shape[1] - frames - 2)
            mel = mel[:, start : start + frames]
            audio = audio[:, start * self.mel_hop_size : (start + frames) * self.mel_hop_size]
        else:
            mel = np.pad(mel, ((0, 0), (0, frames - mel.shape[1])))
            audio = torch.nn.functional.pad(audio, (0, self.n_segments - audio.shape[1]))

        return audio.squeeze(0), audio.shape[1], mel

    def __len__(self):
        return len(self.collection)


class SplicedAudioDataset(Dataset):
    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(
        self,
        manifest_filepath: Union[str, 'pathlib.Path'],
        n_segments: int,
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        trim: Optional[bool] = False,
        truncate_to: Optional[int] = 1,
    ):
        """
        See above AudioDataset for details on dataset and manifest formats.

        Unlike the regular AudioDataset, which samples random segments from each audio array as an example,
        SplicedAudioDataset concatenates all audio arrays together and indexes segments as examples. This way,
        the model sees more data (about 9x for LJSpeech) per epoch.

        Note: this class is not recommended to be used in validation.

        Args:
            manifest_filepath (str, Path): Path to manifest json as described above. Can be comma-separated paths
                such as "train_1.json,train_2.json" which is treated as two separate json files.
            n_segments (int): The length of audio in samples to load. For example, given a sample rate of 16kHz, and
                n_segments=16000, a random 1 second section of audio from the clip will be loaded. The section will
                be randomly sampled everytime the audio is batched. Can be set to -1 to load the entire audio.
            max_duration (float): If audio exceeds this length in seconds, it is filtered from the dataset.
                Defaults to None, which does not filter any audio.
            min_duration(float): If audio is less than this length in seconds, it is filtered from the dataset.
                Defaults to None, which does not filter any audio.
            trim (bool): Whether to use librosa.effects.trim on the audio clip
            truncate_to (int): Ensures that the audio segment returned is a multiple of truncate_to.
                Defaults to 1, which does no truncating.
        """
        assert n_segments > 0

        collection = collections.ASRAudioText(
            manifests_files=manifest_filepath.split(','),
            parser=parsers.make_parser(),
            min_duration=min_duration,
            max_duration=max_duration,
        )
        self.trim = trim
        self.n_segments = n_segments
        self.truncate_to = truncate_to

        self.samples = []
        for index in range(len(collection)):
            example = collection[index]
            with sf.SoundFile(example.audio_file, 'r') as f:
                samples = f.read(dtype='float32').transpose()
                self.samples.append(samples)
        self.samples = np.concatenate(self.samples, axis=0)
        self.samples = self.samples[: self.samples.shape[0] - (self.samples.shape[0] % self.n_segments), ...]

    def __getitem__(self, index):
        """
        Given a index, returns audio and audio_length of the corresponding element. Audio clips of n_segments are
        randomly chosen if the audio is longer than n_segments.
        """
        audio_index = index * self.n_segments
        audio = self.samples[audio_index : audio_index + self.n_segments]

        return audio, self.n_segments

    def __len__(self):
        return self.samples.shape[0] // self.n_segments


class NoisySpecsDataset(Dataset):
    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            'x': NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            'mag': NeuralType(('B', 'any', 'D', 'T'), SpectrogramType()),
            'max_length': NeuralType(None, LengthsType()),
            'y': NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            'T_ys': NeuralType(tuple('B'), LengthsType()),
            'length': NeuralType(tuple('B'), LengthsType()),
            'path_speech': NeuralType(tuple('B'), StringType()),
        }

    def __init__(
        self, destination: Union[str, 'pathlib.Path'], subdir: str, n_fft: int, hop_length: int, num_snr: int,
    ):
        self.tar_dir = Path("%s/degli_data_%d_%dx%d/%s/" % (destination, n_fft, hop_length, num_snr, subdir))
        """
        A modified dataset for training deep-griffin-lim iteration. Contains MSTFT (mag), STFT (y) , and noisy STFT which is
        used for initial phase. By using different levels of noise, the Degli model can learn to improve any phase, and thus
        it can be used iteratively.

        Args:
            destination (str, Path): Path to a directory containing the main data set folder, Similar to the directory
            provided to the preprocessor script, which generates this dataset.
            subdir (str): Either 'train', or 'valid', when using the standard script for generation.
            n_fft (int): STFT parameter. Also detrmines the STFT filter length.
            hop_length (int): STFT parameter.
            num_snr (int): number of noisy samples per clean audio in the original dataset.
        """

        self._all_files = [f for f in os.listdir(self.tar_dir) if 'npz' in f]

    def __getitem__(self, index):

        file = Path(self.tar_dir / self._all_files[index])
        sample = dict()

        with np.load(file, mmap_mode='r') as npz_data:
            for k, v in npz_data.items():
                if k in ['x', 'y', 'y_mag']:
                    sample[k] = torch.from_numpy(v)
                elif k == "path_speech":
                    sample[k] = str(v)
                elif k in ['T_x', 'T_y', 'length']:
                    sample[k] = int(v)
                else:
                    sample[k] = v
        return sample

    def __len__(self):
        return len(self._all_files)

    @torch.no_grad()
    def _collate_fn(self, batch):
        """ return data with zero-padding

        Important data like x, y are all converted to Tensor(cpu).
        :param batch:
        :return: DataDict
            Values can be an Tensor(cpu), list of str, ndarray of int.
        """

        result = dict()
        T_xs = np.array([item.pop('T_x') for item in batch])
        idxs_sorted = np.argsort(T_xs)
        T_xs = T_xs[idxs_sorted].tolist()
        T_ys = [batch[idx].pop('T_y') for idx in idxs_sorted]
        length = [batch[idx].pop('length') for idx in idxs_sorted]

        result['T_xs'], result['T_ys'], result['length'] = T_xs, T_ys, length

        for key, value in batch[0].items():
            if type(value) == str:
                list_data = [batch[idx][key] for idx in idxs_sorted]
                set_data = set(list_data)
                if len(set_data) == 1:
                    result[key] = set_data.pop()
                else:
                    result[key] = list_data
            else:
                if len(batch) > 1:
                    # B, T, F, C
                    data = [batch[idx][key].permute(1, 0, 2) for idx in idxs_sorted]
                    data = pad_sequence(data, batch_first=True)
                    # B, C, F, T
                    data = data.permute(0, 3, 2, 1)
                else:  # B, C, F, T
                    data = batch[0][key].unsqueeze(0).permute(0, 3, 1, 2)

                result[key] = data.contiguous()

        x = result['x']
        mag = result['y_mag']
        max_length = max(result['length'])
        y = result['y']
        T_ys = result['T_ys']
        length = result['length']
        path_speech = result['path_speech']
        return x, mag, max_length, y, T_ys, length, path_speech

    @staticmethod
    @torch.no_grad()
    def decollate_padded(batch: DataDict, idx: int) -> DataDict:
        """ select the `idx`-th data, get rid of padded zeros and return it.

        Important data like x, y are all converted to ndarray.
        :param batch:
        :param idx:
        :return: DataDict
            Values can be an str or ndarray.
        """
        result = dict()
        for key, value in batch.items():
            if type(value) == str:
                result[key] = value
            elif type(value) == list:
                result[key] = value[idx]
            elif not key.startswith('T_'):
                T_xy = 'T_xs' if 'x' in key else 'T_ys'
                value = value[idx, :, :, : batch[T_xy][idx]]  # C, F, T
                value = value.permute(1, 2, 0).contiguous()  # F, T, C
                value = value.numpy()
                if value.shape[-1] == 2:
                    value = value.view(dtype=np.complex64)  # F, T, 1
                result[key] = value

        return result


def setup_noise_augmented_dataset(files_list, num_snr, kwargs_stft, dest, desc):

    os.makedirs(dest)
    with open(files_list, 'r') as list_file:
        all_lines = [line for line in list_file]
        list_file_pbar = tqdm(all_lines, desc=desc, dynamic_ncols=True)

        i_speech = 0
        for line in list_file_pbar:
            audio_file = line.split('|')[0]
            speech = sf.read(audio_file)[0].astype(np.float32)
            spec_clean = np.ascontiguousarray(librosa.stft(speech, **kwargs_stft))
            mag_clean = np.ascontiguousarray(np.abs(spec_clean)[..., np.newaxis])
            signal_power = np.mean(np.abs(speech) ** 2)

            y = spec_clean.view(dtype=np.float32).reshape((*spec_clean.shape, 2))
            ##y = torch.from_numpy(y)
            T_y = spec_clean.shape[1]
            ##mag_clean = torch.from_numpy(mag_clean)
            for k in range(num_snr):
                snr_db = -6 * np.random.rand()
                snr = librosa.db_to_power(snr_db)
                noise_power = signal_power / snr
                noisy = speech + np.sqrt(noise_power) * np.random.randn(len(speech))
                spec_noisy = librosa.stft(noisy, **kwargs_stft)
                spec_noisy = np.ascontiguousarray(spec_noisy)
                T_x = spec_noisy.shape[1]
                x = spec_noisy.view(dtype=np.float32).reshape((*spec_noisy.shape, 2))
                ##x = torch.from_numpy(x)
                mdict = dict(x=x, y=y, y_mag=mag_clean, path_speech=audio_file, length=len(speech), T_x=T_x, T_y=T_y)
                np.savez(
                    f"{dest}/audio_{i_speech}_{k}.npz", **mdict,
                )
                i_speech = i_speech + 1

    return i_speech


def preprocess_linear_specs_dataset(valid_filelist, train_filelist, n_fft, hop_length, num_snr, destination):
    kwargs_stft = dict(hop_length=hop_length, window='hann', center=True, n_fft=n_fft, dtype=np.complex64)

    tar_dir = "%s/degli_data_%d_%dx%d/" % (destination, n_fft, hop_length, num_snr)
    if not os.path.isdir(tar_dir):

        if valid_filelist == "none" or train_filelist == "none":
            logging.error(f"Director {tar_dir} does not exist. Filelists for validation and train must be provided.")
            raise NameError("Missing Argument")
        else:
            logging.info(
                f"Director {tar_dir} does not exist. Preprocessing audio files listed in {valid_filelist}, {train_filelist} to create new dataset."
            )
        os.makedirs(tar_dir)
        n_train = 0
        n_valid = 0
        try:
            n_train = setup_noise_augmented_dataset(
                train_filelist, num_snr, kwargs_stft, tar_dir + "train/", desc="Initializing Train Dataset"
            )
            n_valid = setup_noise_augmented_dataset(
                valid_filelist, num_snr, kwargs_stft, tar_dir + "valid/", desc="Initializing Validation Dataset"
            )
        except FileNotFoundError as err:
            shutil.rmtree(tar_dir)
            raise err
        except:
            e = sys.exc_info()[0]
            shutil.rmtree(tar_dir)
            raise e

        if n_train == 0:
            shutil.rmtree(tar_dir)
            raise EOFError("Dataset initialization failed. No files to preprocess train dataset")

        if n_valid == 0:
            shutil.rmtree(tar_dir)
            raise EOFError("Dataset initialization failed. No files to preprocess validation dataset")

    return tar_dir


class FastSpeech2Dataset(Dataset):
    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports."""
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'a_sig_length': NeuralType(('B'), LengthsType()),
            'transcripts': NeuralType(('B', 'T'), TokenIndex()),
            'transcript_length': NeuralType(('B'), LengthsType()),
            'durations': NeuralType(('B', 'T'), TokenDurationType()),
            'pitches': NeuralType(('B', 'T'), RegressionValuesType()),
            'energies': NeuralType(('B', 'T'), RegressionValuesType()),
        }

    def __init__(
        self,
        manifest_filepath: str,
        mappings_filepath: str,
        sample_rate: int,
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        ignore_file: Optional[str] = None,
        trim: bool = False,
        load_supplementary_values=True,  # Set to False for validation
    ):
        """
        Dataset that loads audio, phonemes and their durations, pitches per frame, and energies per frame
        for FastSpeech 2 from paths described in a JSON manifest (see the AudioDataset documentation for details
        on the manifest format), as well as a mappings file for word to phones and phones to indices.
        The text in the manifest is ignored; instead, the phoneme indices for prediction come from the
        duration files.

        For each sample, paths for duration, energy, and pitch files are inferred from the manifest's audio
        filepaths by replacing '/wavs' with '/phoneme_durations', '/pitches', and '/energies', and swapping out
        the file extension to '.pt', '.npy', and '.npy' respectively.
        For example, given manifest audio path `/data/LJSpeech/wavs/LJ001-0001.wav`, the inferred duration and
        phonemes file path would be `/data/LJSpeech/phoneme_durations/LJ001-0001.pt`.

        Note that validation datasets only need the audio files and phoneme & duration files, set
        `load_supplementary_values` to False for validation sets.

        Args:
            manifest_filepath (str): Path to the JSON manifest file that lists audio files.
            mappings_filepath (str): Path to a JSON mappings file that contains mappings "word2phones" and
                "phone2idx". The latter is used to determine the padding index.
            sample_rate (int): Target sample rate of the audio.
            max_duration (float): If audio exceeds this length in seconds, it is filtered from the dataset.
                Defaults to None, which does not filter any audio.
            min_duration (float): If audio is shorter than this length in seconds, it is filtered from the dataset.
                Defaults to None, which does not filter any audio.
            ignore_file (str): Optional pickled file which contains a list of files to ignore (e.g. files that
                contain OOV words).
                Defaults to None.
            trim (bool): Whether to use librosa.effects.trim on the audio clip.
                Defaults to False.
            load_supplementary_values (bool): Whether or not to load pitch and energy files. Set this to False for
                validation datasets.
                Defaults to True.
        """
        super().__init__()

        # Retrieve mappings from file
        with open(mappings_filepath, 'r') as f:
            mappings = json.load(f)
            self.word2phones = mappings['word2phones']
            self.phone2idx = mappings['phone2idx']

        # Load data from manifests
        audio_files = []
        total_duration = 0
        if isinstance(manifest_filepath, str):
            manifest_filepath = [manifest_filepath]
        for manifest_file in manifest_filepath:
            with open(expanduser(manifest_file), 'r') as f:
                logging.info(f"Loading dataset from {manifest_file}.")
                for line in f:
                    item = json.loads(line)
                    audio_files.append({"audio_filepath": item["audio_filepath"], "duration": item["duration"]})
                    total_duration += item["duration"]

        total_dataset_len = len(audio_files)
        logging.info(f"Loaded dataset with {total_dataset_len} files totalling {total_duration/3600:.2f} hours.")
        self.data = []
        if load_supplementary_values:
            dataitem = py_collections.namedtuple(
                typename='AudioTextEntity', field_names='audio_file duration text_tokens pitches energies'
            )
        else:
            dataitem = py_collections.namedtuple(
                typename='AudioTextEntity', field_names='audio_file duration text_tokens'
            )

        if ignore_file:
            logging.info(f"using {ignore_file} to prune dataset.")
            with open(ignore_file, "rb") as f:
                wavs_to_ignore = set(pickle.load(f))

        pruned_duration = 0
        pruned_items = 0
        for item in audio_files:
            audio_path = item['audio_filepath']
            LJ_id = os.path.splitext(os.path.basename(audio_path))[0]

            # Prune data according to min/max_duration & the ignore file
            if (min_duration and item["duration"] < min_duration) or (
                max_duration and item["duration"] > max_duration
            ):
                pruned_duration += item["duration"]
                pruned_items += 1
                continue
            if ignore_file and (LJ_id in wavs_to_ignore):
                pruned_items += 1
                pruned_duration += item["duration"]
                wavs_to_ignore.remove(LJ_id)
                continue

            # Else not pruned, load additional info

            # Phoneme durations and text token indices from durations file
            dur_path = audio_path.replace('/wavs/', '/phoneme_durations/').replace('.wav', '.pt')
            duration_info = torch.load(dur_path)
            durs = duration_info['token_duration']
            text_tokens = duration_info['text_encoded']

            if load_supplementary_values:
                # Load pitch file (F0s)
                pitch_path = audio_path.replace('/wavs/', '/pitches/').replace('.wav', '.npy')
                pitches = torch.from_numpy(np.load(pitch_path).astype(dtype='float32'))

                # Load energy file (L2-norm of the amplitude of each STFT frame of an utterance)
                energies_path = audio_path.replace('/wavs/', '/energies/').replace('.wav', '.npy')
                energies = torch.from_numpy(np.load(energies_path))

                self.data.append(
                    dataitem(
                        audio_file=item['audio_filepath'],
                        duration=durs,
                        pitches=torch.clamp(pitches, min=1e-5),
                        energies=energies,
                        text_tokens=text_tokens,
                    )
                )
            else:
                self.data.append(dataitem(audio_file=item['audio_filepath'], duration=durs, text_tokens=text_tokens,))

        logging.info(f"Pruned {pruned_items} files and {pruned_duration/3600:.2f} hours.")
        logging.info(
            f"Final dataset contains {len(self.data)} files and {(total_duration-pruned_duration)/3600:.2f} hours."
        )

        self.featurizer = WaveformFeaturizer(sample_rate=sample_rate)
        self.trim = trim
        self.load_supplementary_values = load_supplementary_values

    def __getitem__(self, index):
        sample = self.data[index]

        features = self.featurizer.process(sample.audio_file, trim=self.trim)
        f, fl = features, torch.tensor(features.shape[0]).long()
        t, tl = sample.text_tokens.long(), torch.tensor(len(sample.text_tokens)).long()

        if self.load_supplementary_values:
            return f, fl, t, tl, sample.duration, sample.pitches, sample.energies
        else:
            return f, fl, t, tl, sample.duration, None, None

    def __len__(self):
        return len(self.data)

    def _collate_fn(self, batch):
        pad_id = len(self.phone2idx)
        if self.load_supplementary_values:
            _, audio_lengths, _, tokens_lengths, duration, pitches, energies = zip(*batch)
        else:
            _, audio_lengths, _, tokens_lengths, duration, _, _ = zip(*batch)
        max_audio_len = 0
        max_audio_len = max(audio_lengths).item()
        max_tokens_len = max(tokens_lengths).item()
        max_durations_len = max([len(i) for i in duration])
        max_duration_sum = max([sum(i) for i in duration])
        if self.load_supplementary_values:
            max_pitches_len = max([len(i) for i in pitches])
            max_energies_len = max([len(i) for i in energies])
            if max_pitches_len != max_energies_len or max_pitches_len != max_duration_sum:
                logging.warning(
                    f"max_pitches_len: {max_pitches_len} != max_energies_len: {max_energies_len} != "
                    f"max_duration_sum:{max_duration_sum}. Your training run will error out!"
                )

        # Add padding where necessary
        audio_signal, tokens, duration_batched, pitches_batched, energies_batched = [], [], [], [], []
        for sample_tuple in batch:
            if self.load_supplementary_values:
                sig, sig_len, tokens_i, tokens_i_len, duration, pitch, energy = sample_tuple
            else:
                sig, sig_len, tokens_i, tokens_i_len, duration, _, _ = sample_tuple
            sig_len = sig_len.item()
            if sig_len < max_audio_len:
                pad = (0, max_audio_len - sig_len)
                sig = torch.nn.functional.pad(sig, pad)
            audio_signal.append(sig)
            tokens_i_len = tokens_i_len.item()
            if tokens_i_len < max_tokens_len:
                pad = (0, max_tokens_len - tokens_i_len)
                tokens_i = torch.nn.functional.pad(tokens_i, pad, value=pad_id)
            tokens.append(tokens_i)
            if len(duration) < max_durations_len:
                pad = (0, max_durations_len - len(duration))
                duration = torch.nn.functional.pad(duration, pad)
            duration_batched.append(duration)

            if self.load_supplementary_values:
                pitch = pitch.squeeze(0)
                if len(pitch) < max_pitches_len:
                    pad = (0, max_pitches_len - len(pitch))
                    pitch = torch.nn.functional.pad(pitch.squeeze(0), pad)
                pitches_batched.append(pitch)

                if len(energy) < max_energies_len:
                    pad = (0, max_energies_len - len(energy))
                    energy = torch.nn.functional.pad(energy, pad)
                energies_batched.append(energy)

        audio_signal = torch.stack(audio_signal)
        audio_lengths = torch.stack(audio_lengths)
        tokens = torch.stack(tokens)
        tokens_lengths = torch.stack(tokens_lengths)
        duration_batched = torch.stack(duration_batched)

        if self.load_supplementary_values:
            pitches_batched = torch.stack(pitches_batched)
            energies_batched = torch.stack(energies_batched)
            assert pitches_batched.shape == energies_batched.shape

            return (
                audio_signal,
                audio_lengths,
                tokens,
                tokens_lengths,
                duration_batched,
                pitches_batched,
                energies_batched,
            )
        return (audio_signal, audio_lengths, tokens, tokens_lengths, duration_batched, None, None)

def beta_binomial_prior_distribution(phoneme_count, mel_count,
                                     scaling_factor=1.0):
    """ #TODO """
    P, M = phoneme_count, mel_count
    x = np.arange(0, P)
    mel_text_probs = []
    for i in range(1, M+1):
        a, b = scaling_factor*i, scaling_factor*(M+1-i)
        rv = betabinom(P, a, b)
        mel_i_prob = rv.pmf(x)
        mel_text_probs.append(mel_i_prob)
    return torch.tensor(np.array(mel_text_probs))


def load_filepaths_and_text(filelist, split="|"):
    """ Reads the filelist and creates a list filepaths to wav files and their corresponding text """
    if isinstance(filelist, str):
        with open(filelist, encoding='utf-8') as f:
            filepaths_and_text = [line.strip().split(split) for line in f]
    else:
        filepaths_and_text = filelist
    return filepaths_and_text


def load_wav_to_torch(full_path):
    """ Loads wavdata into torch array """
    sampling_rate, data = read(full_path)
    return torch.from_numpy(data).float(), sampling_rate


class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=None):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(
            sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert(torch.min(y.data) >= -1)
        assert(torch.max(y.data) <= 1)

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output

"""
BSD 3-Clause License

Copyright (c) 2017, Prem Seetharaman
All rights reserved.

* Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from this
  software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.signal import get_window
from librosa.util import pad_center, tiny

class STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""
    def __init__(self, filter_length=800, hop_length=200, win_length=800,
                 window='hann'):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        if window is not None:
            assert(filter_length >= win_length)
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            # window the bases
            forward_basis *= fft_window
            inverse_basis *= fft_window

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        self.num_samples = num_samples

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode='reflect')
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(
            input_data,
            Variable(self.forward_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0)

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.autograd.Variable(
            torch.atan2(imag_part.data, real_part.data))

        return magnitude, phase

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat(
            [magnitude*torch.cos(phase), magnitude*torch.sin(phase)], dim=1)

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            Variable(self.inverse_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0)

        if self.window is not None:
            window_sum = window_sumsquare(
                self.window, magnitude.size(-1), hop_length=self.hop_length,
                win_length=self.win_length, n_fft=self.filter_length,
                dtype=np.float32)
            # remove modulation effects
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0])
            window_sum = torch.autograd.Variable(
                torch.from_numpy(window_sum), requires_grad=False)
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]

            # scale by hop ratio
            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[:, :, int(self.filter_length/2):]
        inverse_transform = inverse_transform[:, :, :-int(self.filter_length/2):]

        return inverse_transform

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction


class FlowtronData(Dataset):

    # @property
    # def output_types(self) -> Optional[Dict[str, NeuralType]]:
    #     """Returns definitions of module output ports."""
    #     return {
    #         'mel': NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
    #         'speaker_id': NeuralType(('any'), IntType()),
    #         'text_encoded': NeuralType(None, LengthsType()),
    #         'attn_prior': NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
    #     }

    def __init__(self, manifest_filepath, filter_length, hop_length, win_length,
                 sampling_rate, mel_fmin, mel_fmax, max_wav_value, p_arpabet,
                 cmudict_path, text_cleaners, speaker_ids=None,
                 use_attn_prior=False, n_frames_per_step=1, attn_prior_threshold=1e-4,
                 prior_cache_path="", betab_scaling_factor=1.0, randomize=True,
                 keep_ambiguous=False, seed=1234):
        self.max_wav_value = max_wav_value
        self.audiopaths_and_text = load_filepaths_and_text(manifest_filepath)
        self.use_attn_prior = use_attn_prior
        self.betab_scaling_factor = betab_scaling_factor
        self.attn_prior_threshold = attn_prior_threshold
        self.keep_ambiguous = keep_ambiguous
        self.n_frames_per_step = n_frames_per_step

        if speaker_ids is None or speaker_ids == '':
            self.speaker_ids = self.create_speaker_lookup_table(
                self.audiopaths_and_text)
        else:
            self.speaker_ids = speaker_ids

        self.stft = TacotronSTFT(filter_length=filter_length,
                                 hop_length=hop_length,
                                 win_length=win_length,
                                 sampling_rate=sampling_rate,
                                 mel_fmin=mel_fmin, mel_fmax=mel_fmax)
        self.sampling_rate = sampling_rate
        self.text_cleaners = text_cleaners
        self.p_arpabet = p_arpabet
        self.cmudict = cmudict.CMUDict(
            cmudict_path, keep_ambiguous=keep_ambiguous)
        if speaker_ids is None:
            self.speaker_ids = self.create_speaker_lookup_table(
                self.audiopaths_and_text)
        else:
            self.speaker_ids = speaker_ids

        # caching makes sense for p_phoneme=1.0
        # for other values, everytime text lengths will change
        self.prior_cache_path = prior_cache_path
        self.caching_enabled = False
        if (self.prior_cache_path is not None and
                self.prior_cache_path != "" and p_arpabet == 1.0):
            self.caching_enabled = True
        # make sure caching path exists
        if (self.caching_enabled and
                not os.path.exists(self.prior_cache_path)):
            os.makedirs(self.prior_cache_path)

        random.seed(seed)
        if randomize:
            random.shuffle(self.audiopaths_and_text)

    def compute_attention_prior(self, audiopath, mel_length, text_length):
        folder_path = audiopath.split('/')[-2]
        filename = os.path.basename(audiopath).split('.')[0]
        prior_path = os.path.join(
            self.prior_cache_path,
            folder_path + "_" + filename)

        prior_path += "_prior.pth"

        prior_loaded = False
        if self.caching_enabled and os.path.exists(prior_path):
            attn_prior = torch.load(prior_path)
            if (attn_prior.shape[1] == text_length and
                    attn_prior.shape[0] == mel_length):
                prior_loaded = True
            else:
                print("Prior size mismatch, recomputing")

        if not prior_loaded:
            attn_prior = beta_binomial_prior_distribution(
                                            text_length,
                                            mel_length,
                                            self.betab_scaling_factor)
            if self.caching_enabled:
                torch.save(attn_prior, prior_path)

        if self.attn_prior_threshold > 0:
            attn_prior = attn_prior.masked_fill(
                attn_prior < self.attn_prior_threshold, 0.0)

        return attn_prior

    def create_speaker_lookup_table(self, audiopaths_and_text):
        speaker_ids = np.sort(np.unique([x[2] for x in audiopaths_and_text]))
        d = {int(speaker_ids[i]): i for i in range(len(speaker_ids))}
        print("Number of speakers :", len(d))
        return d

    def get_mel(self, audio):
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        return melspec

    def get_speaker_id(self, speaker_id):
        return torch.LongTensor([self.speaker_ids[int(speaker_id)]])

    def get_text(self, text):
        text = _clean_text(text, self.text_cleaners)
        words = re.findall(r'\S*\{.*?\}\S*|\S+', text)
        text = ' '.join([get_arpabet(word, self.cmudict)
                         if random.random() < self.p_arpabet else word
                         for word in words])
        text_norm = torch.LongTensor(text_to_sequence(text))
        return text_norm

    def _collate_fn_(self, batch):
        """ Zero-pads model inputs and targets based on number of frames per step """
        """Collate's training batch from normalized text and mel-spectrogram """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[2]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0].item()

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][2]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mel_channels = batch[0][0].size(0)
        max_target_len = max([x[0].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded, gate padded and speaker ids
        mel_padded = torch.FloatTensor(
                len(batch), num_mel_channels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))

        attn_prior_padded = None
        if self.use_attn_prior:
            attn_prior_padded = torch.FloatTensor(
                len(batch), max_target_len, max_input_len)
            attn_prior_padded.zero_()
        speaker_ids = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][0]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)
            speaker_ids[i] = batch[ids_sorted_decreasing[i]][1]
            if self.use_attn_prior:
                cur_attn_prior = batch[ids_sorted_decreasing[i]][3]
                attn_prior_padded[
                    i,
                    :cur_attn_prior.size(0),
                    :cur_attn_prior.size(1)] = cur_attn_prior

        return (mel_padded, speaker_ids, text_padded, input_lengths,
                output_lengths, gate_padded, attn_prior_padded)
    
    def __getitem__(self, index):
        # Read audio and text
        audiopath, text, speaker_id = self.audiopaths_and_text[index]
        audio, sampling_rate = load_wav_to_torch(audiopath)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))

        mel = self.get_mel(audio)
        text_encoded = self.get_text(text)
        speaker_id = self.get_speaker_id(speaker_id)
        attn_prior = None
        if self.use_attn_prior:
            attn_prior = self.compute_attention_prior(
                audiopath, mel.shape[1], text_encoded.shape[0])

        return (mel, speaker_id, text_encoded, attn_prior)

    def __len__(self):
        return len(self.audiopaths_and_text)

