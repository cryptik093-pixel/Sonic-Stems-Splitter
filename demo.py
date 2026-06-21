"""
Simple demo script to test the separation engine.
Drop a WAV file in and get stems out.
"""

import sys
from pathlib import Path
from separation_engine import StemSeparationEngine


def demo():
    """Run a quick demo of the separation engine."""
    
    # Check for input file
    if len(sys.argv) < 2:
        print("📍 Usage: python demo.py <audio_file.wav>")
        print("\nExample: python demo.py my_song.wav")
        print("\nThis will create a 'stems' directory with:")
        print("  - drums.wav")
        print("  - bass.wav")
        print("  - other.wav")
        print("  - vocals.wav")
        sys.exit(1)
    
    input_file = sys.argv[1]
    input_path = Path(input_file)
    
    # Validate input
    if not input_path.exists():
        print(f"❌ Error: File not found - {input_file}")
        sys.exit(1)
    
    if input_path.suffix.lower() not in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
        print(f"❌ Error: Unsupported format - {input_path.suffix}")
        print("Supported: WAV, MP3, FLAC, OGG, M4A")
        sys.exit(1)
    
    print("🚀 Sonic Stems Splitter - Phase 1 Demo")
    print("=" * 50)
    
    # Initialize engine
    print("📦 Initializing separation engine...")
    engine = StemSeparationEngine()
    
    # Process
    output_dir = "./stems"
    result = engine.process(str(input_path), output_dir)
    
    # Report results
    print("\n" + "=" * 50)
    print("✅ SUCCESS - Audio separated into stems!")
    print("=" * 50)
    print(f"\nInput:  {result['input_file']}")
    print(f"Output: {result['output_dir']}")
    print(f"\nGenerated stems:")
    for stem_path in result['stems']:
        stem_name = Path(stem_path).stem
        print(f"  ✓ {stem_name}.wav")
    
    print(f"\nDevice: {result['device']}")
    print(f"Sample Rate: {result['sample_rate']} Hz")


if __name__ == "__main__":
    demo()
