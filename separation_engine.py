"""
Core Audio Separation Engine - Phase 1
Handles: Input → Separate → Export
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import librosa
import soundfile as sf
from demucs.pretrained import get_model
from demucs.apply import apply_model


class StemSeparationEngine:
    """High-performance audio stem separator using Demucs model."""
    
    def __init__(self, model_name: str = "htdemucs", device: Optional[str] = None):
        """
        Initialize the separation engine.
        
        Args:
            model_name: Model to use (htdemucs, mdx_extra, etc.)
            device: Device to run on ('cuda' or 'cpu'). Auto-detects if None.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Loading {model_name} model on {self.device}...")
        try:
            self.model = get_model(model_name).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_name}': {e}. Check internet connection for model download.")
        self.model_name = model_name
        self.stems = ["drums", "bass", "other", "vocals"]
        
    def load_audio(self, audio_path: str, sr: int = 44100) -> Tuple[np.ndarray, int]:
        """
        Load audio file with automatic resampling.
        
        Returns audio as (channels, samples) numpy array.
        
        Args:
            audio_path: Path to audio file
            sr: Target sample rate
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Load audio using librosa for compatibility
        # Returns (samples,) for mono or (samples,) for mono loaded from stereo
        y, sr_original = librosa.load(str(audio_path), sr=sr, mono=False)
        
        # Ensure stereo format (channels, samples)
        if y.ndim == 1:
            y = np.stack([y, y])
        
        return y, sr
    
    def separate(self, audio: np.ndarray, sr: int = 44100) -> Dict[str, np.ndarray]:
        """
        Separate audio into individual stems.
        
        Args:
            audio: Audio array with shape (channels, samples)
            sr: Sample rate
            
        Returns:
            Dictionary mapping stem names to audio arrays with shape (channels, samples)
        """
        # Convert numpy to torch tensor
        waveform = torch.from_numpy(audio).float().to(self.device)
        
        # Ensure proper shape (channels, samples)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        
        # Add batch dimension if needed (batch, channels, samples)
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        
        # Apply separation model
        try:
            with torch.no_grad():
                separated = apply_model(
                    self.model,
                    waveform,
                    device=self.device,
                    progress=True,
                    num_workers=0,
                )
        except Exception as e:
            raise RuntimeError(f"Model separation failed: {e}")
        
        # Validate output shape: (batch, stems, channels, samples)
        if separated.shape[1] != len(self.stems):
            raise ValueError(
                f"Model output {separated.shape[1]} stems, expected {len(self.stems)}. "
                f"Model may not be compatible."
            )
        
        # Process output - extract first batch item
        stems_dict = {}
        for idx, stem_name in enumerate(self.stems):
            stem_audio = separated[0, idx].cpu().numpy()  # (channels, samples)
            stems_dict[stem_name] = stem_audio
        
        return stems_dict
    
    def export_stems(
        self,
        stems: Dict[str, np.ndarray],
        output_dir: str,
        sr: int = 44100,
        bit_depth: int = 16
    ) -> List[str]:
        """
        Export separated stems to WAV files.
        
        Args:
            stems: Dictionary of stem names and audio arrays with shape (channels, samples)
            output_dir: Directory to save stems
            sr: Sample rate
            bit_depth: Bit depth (16 or 24)
            
        Returns:
            List of output file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_files = []
        subtype = "PCM_16" if bit_depth == 16 else "PCM_24"
        
        for stem_name, audio in stems.items():
            # Normalize per-channel to prevent clipping and maintain channel balance
            normalized_audio = audio.copy()
            for ch in range(normalized_audio.shape[0]):
                max_val = np.abs(normalized_audio[ch]).max()
                if max_val > 0:
                    normalized_audio[ch] = normalized_audio[ch] / max_val * 0.95
            
            stem_file = output_path / f"{stem_name}.wav"
            
            # soundfile.write expects (samples, channels) for multichannel
            # Our audio is (channels, samples), so transpose
            try:
                sf.write(
                    str(stem_file),
                    normalized_audio.T,  # Convert (channels, samples) → (samples, channels)
                    sr,
                    subtype=subtype
                )
                output_files.append(str(stem_file))
            except Exception as e:
                raise RuntimeError(f"Failed to write stem file {stem_file}: {e}")
        
        return output_files
    
    def process(
        self,
        input_path: str,
        output_dir: str,
        sr: int = 44100
    ) -> Dict[str, str]:
        """
        Complete pipeline: load → separate → export.
        
        Args:
            input_path: Path to input audio file
            output_dir: Directory for output stems
            sr: Sample rate
            
        Returns:
            Dictionary with status and output file paths
        """
        try:
            print(f"🎵 Loading audio from {input_path}...")
            audio, sr = self.load_audio(input_path, sr=sr)
            print(f"   ✓ Loaded audio: {audio.shape[0]} channels, {audio.shape[1]} samples")
            
            print(f"🎸 Separating stems using {self.model_name}...")
            stems = self.separate(audio, sr=sr)
            print(f"   ✓ Separated into {len(stems)} stems")
            
            print(f"💾 Exporting stems to {output_dir}...")
            output_files = self.export_stems(stems, output_dir, sr=sr)
            print(f"   ✓ Exported {len(output_files)} stem files")
            
            return {
                "status": "success",
                "input_file": input_path,
                "output_dir": output_dir,
                "stems": output_files,
                "sample_rate": sr,
                "device": self.device
            }
        except Exception as e:
            return {
                "status": "error",
                "error_message": str(e),
                "input_file": input_path,
                "output_dir": output_dir,
            }


def main():
    """CLI interface for stem separation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Sonic Stems Splitter - Phase 1")
    parser.add_argument("input", help="Input audio file (WAV, MP3, FLAC, etc.)")
    parser.add_argument("-o", "--output", default="./stems", help="Output directory")
    parser.add_argument("-m", "--model", default="htdemucs", help="Model name")
    parser.add_argument("-sr", "--sample-rate", type=int, default=44100, help="Sample rate")
    
    args = parser.parse_args()
    
    # Initialize engine
    engine = StemSeparationEngine(model_name=args.model)
    
    # Process audio
    result = engine.process(args.input, args.output, sr=args.sample_rate)
    
    # Report results
    print("\n" + "="*60)
    if result["status"] == "success":
        print("✅ SEPARATION COMPLETE!")
        print("="*60)
        print(f"Output directory: {result['output_dir']}")
        for stem_file in result['stems']:
            print(f"  ✓ {Path(stem_file).name}")
        print(f"\nDevice: {result['device']}")
        print(f"Sample Rate: {result['sample_rate']} Hz")
    else:
        print("❌ SEPARATION FAILED!")
        print("="*60)
        print(f"Error: {result.get('error_message', 'Unknown error')}")
    print("="*60)


if __name__ == "__main__":
    main()
