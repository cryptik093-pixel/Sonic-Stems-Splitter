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
        self.model = get_model(model_name).to(self.device)
        self.model_name = model_name
        self.stems = ["drums", "bass", "other", "vocals"]
        
    def load_audio(self, audio_path: str, sr: int = 44100) -> Tuple[np.ndarray, int]:
        """
        Load audio file with automatic resampling.
        
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
        y, sr_original = librosa.load(str(audio_path), sr=sr, mono=False)
        
        # Ensure stereo
        if y.ndim == 1:
            y = np.stack([y, y])
        
        return y, sr
    
    def separate(self, audio: np.ndarray, sr: int = 44100) -> Dict[str, np.ndarray]:
        """
        Separate audio into individual stems.
        
        Args:
            audio: Audio array (channels, samples)
            sr: Sample rate
            
        Returns:
            Dictionary mapping stem names to audio arrays
        """
        # Convert numpy to torch tensor
        waveform = torch.from_numpy(audio).float().to(self.device)
        
        # Ensure proper shape (channels, samples)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        
        # Add batch dimension if needed
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        
        # Apply separation model
        with torch.no_grad():
            separated = apply_model(
                self.model,
                waveform,
                device=self.device,
                progress=True,
                num_workers=0,
            )
        
        # Process output
        stems_dict = {}
        for idx, stem_name in enumerate(self.stems):
            stem_audio = separated[0, idx].cpu().numpy()
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
            stems: Dictionary of stem names and audio arrays
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
            # Normalize to prevent clipping
            max_val = np.abs(audio).max()
            if max_val > 0:
                audio = audio / max_val * 0.95
            
            stem_file = output_path / f"{stem_name}.wav"
            sf.write(
                str(stem_file),
                audio.T,  # Transpose for librosa compatibility (samples, channels)
                sr,
                subtype=subtype
            )
            output_files.append(str(stem_file))
        
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
        print(f"🎵 Loading audio from {input_path}...")
        audio, sr = self.load_audio(input_path, sr=sr)
        
        print(f"🎸 Separating stems using {self.model_name}...")
        stems = self.separate(audio, sr=sr)
        
        print(f"💾 Exporting stems to {output_dir}...")
        output_files = self.export_stems(stems, output_dir, sr=sr)
        
        return {
            "status": "success",
            "input_file": input_path,
            "output_dir": output_dir,
            "stems": output_files,
            "sample_rate": sr,
            "device": self.device
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
    
    print("\n✅ Separation complete!")
    print(f"Output directory: {result['output_dir']}")
    for stem_file in result['stems']:
        print(f"  - {Path(stem_file).name}")


if __name__ == "__main__":
    main()
