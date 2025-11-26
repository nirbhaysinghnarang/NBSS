# data_loaders/librispeech_plus.py
# Adapted from sms_wsj_plus.py to use LibriSpeech instead of WSJ

import json
import os
from os.path import *
import random
from pathlib import Path
from typing import *

import numpy as np
import soundfile as sf
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.rank_zero import (rank_zero_info, rank_zero_warn)
from torch.utils.data import DataLoader, Dataset
from scipy.signal import resample_poly

from data_loaders.utils.collate_func import default_collate_func
from data_loaders.utils.mix import *
from data_loaders.utils.my_distributed_sampler import MyDistributedSampler
from data_loaders.utils.diffuse_noise import (gen_desired_spatial_coherence, gen_diffuse_noise)
from data_loaders.utils.window import reverberation_time_shortening_window


class LibriSpeechPlusDataset(Dataset):

    def __init__(
        self,
        librispeech_dir: str,
        rir_dir: str,
        target: str,
        dataset: str,
        ovlp: str,
        speech_overlap_ratio: Tuple[float, float] = [0.1, 1.0],
        sir: Tuple[float, float] = [-5, 5],
        snr: Tuple[float, float] = [10, 20],
        audio_time_len: Optional[float] = None,
        sample_rate: int = 8000,
        num_spk: int = 2,
        noise_type: List[Literal['babble', 'white']] = ['babble', 'white'],
        return_noise: bool = False,
        return_rvbt: bool = False,
    ) -> None:
        """LibriSpeech-Plus dataset (adapted from SMS-WSJ-Plus)

        Args:
            librispeech_dir: dir containing LibriSpeech (train-clean-100, dev-clean, test-clean, etc.)
            rir_dir: dir containing generated RIRs with train/validation/test subdirs
            target: 'revb_image', 'direct_path', or 'RTS_0.1s'
            dataset: 'train', 'val', 'test'
            ovlp: overlap type - 'mid', 'headtail', 'startend', 'full', 'hms', 'fhms'
            speech_overlap_ratio: range of overlap ratios
            sir: signal-to-interference ratio range (dB) for 2-speaker case
            snr: signal-to-noise ratio range (dB)
            audio_time_len: cut audio to this length (seconds), None for full length
            sample_rate: target sample rate (should match RIR sample rate)
            num_spk: number of speakers (1 for enhancement, 2 for separation)
            noise_type: list of noise types to use
        """
        super().__init__()
        assert target in ['revb_image', 'direct_path'] or target.startswith('RTS'), target
        assert dataset in ['train', 'val', 'test'], dataset
        assert ovlp in ['mid', 'headtail', 'startend', 'full', 'hms', 'fhms'], ovlp
        assert num_spk in [1, 2], f'num_spk must be 1 or 2, got {num_spk}'
        assert len(set(noise_type) - set(['babble', 'white'])) == 0, noise_type

        if ovlp == 'full' and audio_time_len is None:
            rank_zero_warn(f'dataset {dataset} could not achieve full-overlap without giving a length, the overlap type will be one of startend/headtail/mid-overlap')
            ovlp = 'hms'

        self.librispeech_dir = Path(librispeech_dir).expanduser()
        self.target = target
        self.dataset = dataset
        self.ovlp = ovlp
        self.speech_overlap_ratio = speech_overlap_ratio
        self.sir = sir
        self.audio_time_len = audio_time_len
        self.sample_rate = sample_rate

        # Map dataset names to LibriSpeech subdirectories
        libri_subsets = {
            'train': ['train-clean-100', 'train-clean-360', 'train-other-500'],
            'val': ['dev-clean', 'dev-other'],
            'test': ['test-clean', 'test-other'],
        }

        # Scan for utterances
        self.utterances = []
        for subset in libri_subsets[dataset]:
            subset_path = self.librispeech_dir / subset
            if subset_path.exists():
                self.utterances.extend(list(subset_path.rglob('*.flac')))
        
        self.utterances = [str(u) for u in self.utterances]
        self.utterances.sort()
        
        if len(self.utterances) == 0:
            raise RuntimeError(
                f"No .flac files found in {librispeech_dir} for dataset '{dataset}'. "
                f"Expected subdirs: {libri_subsets[dataset]}"
            )
        
        rank_zero_info(f"LibriSpeech [{dataset}]: Found {len(self.utterances)} utterances")

        # Create pairs for 2-speaker case (or single utterances for 1-speaker)
        self.num_spk = num_spk
        if num_spk == 2:
            # Create random pairs - each utterance paired with another
            rng = np.random.default_rng(42)  # Fixed seed for reproducibility
            shuffled = self.utterances.copy()
            rng.shuffle(shuffled)
            # Ensure no utterance is paired with itself
            self.utterance_pairs = []
            for i, utt in enumerate(self.utterances):
                pair_utt = shuffled[i]
                if pair_utt == utt:
                    pair_utt = shuffled[(i + 1) % len(shuffled)]
                self.utterance_pairs.append((utt, pair_utt))
        else:
            self.utterance_pairs = [(utt,) for utt in self.utterances]

        self.return_rvbt = return_rvbt
        self.return_noise = return_noise
        self.noises = self.utterances.copy()  # Use utterances as babble noise source
        self.snr = snr

        # Load RIRs
        rir_subdir = {'train': 'train', 'val': 'validation', 'test': 'test'}[dataset]
        self.rir_dir = Path(rir_dir).expanduser() / rir_subdir
        self.rirs = [str(r) for r in list(self.rir_dir.rglob('*.npz'))]
        self.rirs.sort()
        
        if len(self.rirs) == 0:
            raise RuntimeError(f"No RIR .npz files found in {self.rir_dir}")
        
        rank_zero_info(f"LibriSpeech [{dataset}]: Found {len(self.rirs)} RIRs")

        # Load or generate diffuse noise parameters
        diffuse_paras_path = Path(rir_dir).expanduser() / 'diffuse.npz'
        nfft = 512 if sample_rate == 16000 else 256
        self.nfft = nfft
        
        if diffuse_paras_path.exists():
            self.Cs = np.load(diffuse_paras_path, allow_pickle=True)['Cs']
        else:
            pos_mics = np.load(self.rirs[0], allow_pickle=True)['pos_rcv']
            _, self.Cs = gen_desired_spatial_coherence(
                pos_mics=pos_mics, fs=self.sample_rate, 
                noise_field='spherical', c=343, nfft=nfft
            )
            try:
                np.savez(diffuse_paras_path, Cs=self.Cs)
                rank_zero_info(f"Saved diffuse parameters to {diffuse_paras_path}")
            except:
                pass

        self.shuffle_rir = (dataset == 'train')
        self.noise_type = noise_type

    def _load_audio(self, path: str, rng) -> np.ndarray:
        """Load audio file and resample if needed."""
        audio, sr = sf.read(path, dtype='float32')
        if sr != self.sample_rate:
            audio = resample_poly(audio, self.sample_rate, sr).astype(np.float32)
        return audio

    def __getitem__(self, index_seed: Tuple[int, int]):
        index, seed = index_seed

        rng = np.random.default_rng(np.random.PCG64(seed))
        num_spk = self.num_spk

        # Step 1: Load clean speech signals
        cleans, uttrs = [], []
        pair = self.utterance_pairs[index % len(self.utterance_pairs)]
        
        for i in range(num_spk):
            if i < len(pair):
                utt_path = pair[i]
            else:
                # Fallback: pick random utterance
                utt_path = self.utterances[rng.integers(0, len(self.utterances))]
            
            uttrs.append(utt_path)
            audio = self._load_audio(utt_path, rng)
            cleans.append(audio)

        # Step 2: Load RIRs
        if self.shuffle_rir:
            rir_this = self.rirs[rng.integers(low=0, high=len(self.rirs))]
        else:
            rir_this = self.rirs[index % len(self.rirs)]
        
        rir_dict = np.load(rir_this, allow_pickle=True)
        
        if 'fs' in rir_dict:
            sr_rir = int(rir_dict['fs'])
            assert sr_rir == self.sample_rate, f"RIR sample rate {sr_rir} != {self.sample_rate}"

        rir = rir_dict['rir'].astype(np.float64)  # [nsrc, nmic, time]
        assert rir.shape[0] >= num_spk, (rir.shape, num_spk)
        
        spk_rir_idxs = rng.choice(rir.shape[0], size=num_spk, replace=False).tolist()
        rir = rir[spk_rir_idxs, :, :]
        
        if self.target == 'direct_path':
            rir_target = rir_dict['rir_dp'].astype(np.float64)[spk_rir_idxs, :, :]
        elif self.target == 'revb_image':
            rir_target = rir.copy()
        elif self.target.startswith('RTS'):
            rts_time = float(self.target.replace('RTS_', '').replace('s', ''))
            win = reverberation_time_shortening_window(
                rir=rir, original_T60=float(rir_dict['RT60']), 
                target_T60=rts_time, sr=self.sample_rate
            )
            rir_target = win * rir
        else:
            raise NotImplementedError(f'Unknown target: {self.target}')
        
        num_mic = rir.shape[1]

        # Step 3: Decide overlap type and lengths
        if num_spk == 2:
            ovlp_type = sample_an_overlap(rng=rng, ovlp_type=self.ovlp, num_spk=num_spk)
            lens = [clean.shape[0] for clean in cleans]
            ovlp_ratio, lens, mix_frames = sample_ovlp_ratio_and_cal_length(
                rng=rng,
                ovlp_type=ovlp_type,
                ratio_range=self.speech_overlap_ratio,
                target_len=None if self.audio_time_len is None else int(self.audio_time_len * self.sample_rate),
                lens=lens,
            )
        else:
            ovlp_type = 'none'
            ovlp_ratio = 0.0
            if self.audio_time_len is not None:
                mix_frames = int(self.audio_time_len * self.sample_rate)
            else:
                mix_frames = cleans[0].shape[0] + rir.shape[2] - 1
            lens = [mix_frames]

        # Step 4: Pad or cut signals
        cleans = pad_or_cut(wavs=cleans, lens=lens, rng=rng)

        # Step 5: Convolve and overlap
        rvbts, targets = zip(*[
            convolve(wav=wav, rir=rir_spk, rir_target=rir_spk_t, ref_channel=0, align=True) 
            for (wav, rir_spk, rir_spk_t) in zip(cleans, rir, rir_target)
        ])
        
        if num_spk == 2:
            rvbts, targets = overlap2(
                rvbts=rvbts, targets=targets, 
                ovlp_type=ovlp_type, mix_frames=mix_frames, rng=rng
            )
        else:
            # Single speaker - just trim to mix_frames
            rvbts = [rvbts[0][:, :mix_frames]]
            targets = [targets[0][:, :mix_frames]]
            rvbts = np.stack(rvbts, axis=0)
            targets = np.stack(targets, axis=0)

        # Step 6: Rescale for SIR (2-speaker only)
        sir_this = None
        if self.sir is not None and num_spk == 2:
            sir_this = rng.uniform(low=self.sir[0], high=self.sir[1])
            coeff = cal_coeff_for_adjusting_relative_energy(wav1=rvbts[0], wav2=rvbts[1], target_dB=sir_this)
            if coeff is not None:
                rvbts[1][:] *= coeff
                if targets is not rvbts:
                    targets[1][:] *= coeff

        # Step 7: Generate noise
        noise_type = self.noise_type[rng.integers(low=0, high=len(self.noise_type))]
        mix = np.sum(rvbts, axis=0)
        
        if noise_type == 'babble':
            noises = []
            for i in range(num_mic):
                noise_i = np.zeros(shape=(mix_frames,), dtype=mix.dtype)
                for j in range(10):
                    noise_path = self.noises[rng.integers(low=0, high=len(self.noises))]
                    noise_ij = self._load_audio(noise_path, rng)
                    noise_i += pad_or_cut([noise_ij], lens=[mix_frames], rng=rng)[0]
                noises.append(noise_i)
            noise = np.stack(noises, axis=0).reshape(-1)
        elif noise_type == 'white':
            noise = rng.normal(size=mix.shape[0] * mix.shape[1]).astype(np.float32)
        
        noise = gen_diffuse_noise(noise=noise, L=mix_frames, Cs=self.Cs, nfft=self.nfft, rng=rng)

        # Apply SNR
        snr_this = rng.uniform(low=self.snr[0], high=self.snr[1])
        coeff = cal_coeff_for_adjusting_relative_energy(wav1=mix, wav2=noise, target_dB=snr_this)
        if coeff is not None:
            noise[:, :] *= coeff
        
        snr_real = 10 * np.log10(np.sum(mix**2) / (np.sum(noise**2) + 1e-8))
        mix = mix + noise

        # Scale to [-0.9, 0.9]
        max_val = max(np.max(np.abs(mix)), np.max(np.abs(targets)))
        if max_val > 0:
            scale_value = 0.9 / max_val
            mix *= scale_value
            targets *= scale_value

        # Generate save names
        saveto = [basename(u).replace('.flac', f'_{i}.wav') for i, u in enumerate(uttrs)]

        paras = {
            'index': index,
            'seed': seed,
            'saveto': saveto,
            'target': self.target,
            'sample_rate': self.sample_rate,
            'dataset': f'LibriSpeech-Plus/{self.dataset}',
            'noise_type': noise_type,
            'noise': noises if self.return_noise else None,
            'rvbt': rvbts if self.return_rvbt else None,
            'sir': float(sir_this) if sir_this is not None else None,
            'snr': float(snr_real),
            'ovlp_type': ovlp_type,
            'ovlp_ratio': float(ovlp_ratio),
            'audio_time_len': self.audio_time_len,
            'num_spk': num_spk,
            'rir': {
                'RT60': float(rir_dict['RT60']),
                'pos_src': rir_dict['pos_src'],
                'pos_rcv': rir_dict['pos_rcv'],
            }
        }

        return torch.as_tensor(mix, dtype=torch.float32), torch.as_tensor(targets, dtype=torch.float32), paras

    def __len__(self):
        return len(self.utterance_pairs)


class LibriSpeechPlusDataModule(LightningDataModule):

    def __init__(
        self,
        librispeech_dir: str = '~/datasets/LibriSpeech',
        rir_dir: str = '~/datasets/librispeech_rirs',
        target: str = "direct_path",
        datasets: Tuple[str, str, str, str] = ['train', 'val', 'test', 'test'],
        audio_time_len: Tuple[Optional[float], ...] = [4.0, 4.0, None, None],
        ovlp: Union[str, Tuple[str, str, str, str]] = "mid",
        speech_overlap_ratio: Tuple[float, float] = [0.1, 1.0],
        sir: Optional[Tuple[float, float]] = [-5, 5],
        snr: Tuple[float, float] = [0, 20],
        num_spk: int = 1,
        noise_type: List[Literal['babble', 'white']] = ['babble', 'white'],
        sample_rate: int = 16000,
        return_noise: bool = False,
        return_rvbt: bool = False,
        batch_size: List[int] = [4, 4],
        num_workers: int = 8,
        collate_func_train: Callable = default_collate_func,
        collate_func_val: Callable = default_collate_func,
        collate_func_test: Callable = default_collate_func,
        seeds: Tuple[Optional[int], int, int, int] = [None, 2, 3, 3],
        pin_memory: bool = True,
        prefetch_factor: int = 5,
        persistent_workers: bool = False,
    ):
        super().__init__()
        self.librispeech_dir = librispeech_dir
        self.rir_dir = rir_dir
        self.target = target
        self.datasets = datasets
        self.audio_time_len = audio_time_len
        self.ovlp = [ovlp] * 4 if isinstance(ovlp, str) else ovlp
        self.speech_overlap_ratio = speech_overlap_ratio
        self.sir = sir
        self.snr = snr
        self.num_spk = num_spk
        self.noise_type = noise_type
        self.sample_rate = sample_rate
        self.return_noise = return_noise
        self.return_rvbt = return_rvbt
        self.persistent_workers = persistent_workers

        self.batch_size = batch_size
        while len(self.batch_size) < 4:
            self.batch_size.append(1)

        rank_zero_info("dataset: LibriSpeech-Plus")
        rank_zero_info(f'train/val/test/predict: {self.datasets}')
        rank_zero_info(f'batch size: train/val/test/predict = {self.batch_size}')
        rank_zero_info(f'audio_time_length: train/val/test/predict = {self.audio_time_len}')
        rank_zero_info(f'target: {self.target}')
        rank_zero_info(f'sample_rate: {self.sample_rate}')
        rank_zero_info(f'num_spk: {self.num_spk}')

        self.num_workers = num_workers
        self.collate_func = [collate_func_train, collate_func_val, collate_func_test, default_collate_func]

        self.seeds = []
        for seed in seeds:
            self.seeds.append(seed if seed is not None else random.randint(0, 1000000))

        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor

    def setup(self, stage=None):
        self.current_stage = stage

    def construct_dataloader(self, dataset, ovlp, audio_time_len, seed, shuffle, batch_size, collate_fn):
        ds = LibriSpeechPlusDataset(
            librispeech_dir=self.librispeech_dir,
            rir_dir=self.rir_dir,
            target=self.target,
            dataset=dataset,
            ovlp=ovlp,
            speech_overlap_ratio=self.speech_overlap_ratio,
            sir=self.sir,
            snr=self.snr,
            audio_time_len=audio_time_len,
            sample_rate=self.sample_rate,
            num_spk=self.num_spk,
            noise_type=self.noise_type,
            return_noise=self.return_noise,
            return_rvbt=self.return_rvbt,
        )

        return DataLoader(
            ds,
            sampler=MyDistributedSampler(ds, seed=seed, shuffle=shuffle),
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def train_dataloader(self) -> DataLoader:
        return self.construct_dataloader(
            dataset=self.datasets[0],
            ovlp=self.ovlp[0],
            audio_time_len=self.audio_time_len[0],
            seed=self.seeds[0],
            shuffle=True,
            batch_size=self.batch_size[0],
            collate_fn=self.collate_func[0],
        )

    def val_dataloader(self) -> DataLoader:
        return self.construct_dataloader(
            dataset=self.datasets[1],
            ovlp=self.ovlp[1],
            audio_time_len=self.audio_time_len[1],
            seed=self.seeds[1],
            shuffle=False,
            batch_size=self.batch_size[1],
            collate_fn=self.collate_func[1],
        )

    def test_dataloader(self) -> DataLoader:
        return self.construct_dataloader(
            dataset=self.datasets[2],
            ovlp=self.ovlp[2],
            audio_time_len=self.audio_time_len[2],
            seed=self.seeds[2],
            shuffle=False,
            batch_size=self.batch_size[2],
            collate_fn=self.collate_func[2],
        )

    def predict_dataloader(self) -> DataLoader:
        return self.construct_dataloader(
            dataset=self.datasets[3],
            ovlp=self.ovlp[3],
            audio_time_len=self.audio_time_len[3],
            seed=self.seeds[3],
            shuffle=False,
            batch_size=self.batch_size[3],
            collate_fn=self.collate_func[3],
        )


if __name__ == '__main__':
    """python -m data_loaders.librispeech_plus"""
    from jsonargparse import ArgumentParser
    parser = ArgumentParser("")
    parser.add_class_arguments(LibriSpeechPlusDataModule, nested_key='data')
    parser.add_argument('--save_dir', type=str, default='dataset')
    parser.add_argument('--dataset', type=str, default='predict')
    parser.add_argument('--gen_unprocessed', type=bool, default=True)
    parser.add_argument('--gen_target', type=bool, default=True)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if not args.gen_unprocessed and not args.gen_target:
        exit()

    args_dict = args.data
    args_dict['num_workers'] = 1
    datamodule = LibriSpeechPlusDataModule(**args_dict)
    datamodule.setup()

    if args.dataset.startswith('train'):
        dataloader = datamodule.train_dataloader()
    elif args.dataset.startswith('val'):
        dataloader = datamodule.val_dataloader()
    elif args.dataset.startswith('test'):
        dataloader = datamodule.test_dataloader()
    else:
        dataloader = datamodule.predict_dataloader()

    for idx, (noisy, tar, paras) in enumerate(dataloader):
        print(f'{idx}/{len(dataloader)}', end=' ')
        print(noisy.shape, tar.shape, paras[0]['snr'], paras[0]['noise_type'])
        if idx > 10:
            break
