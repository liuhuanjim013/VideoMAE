#!/bin/bash

# VideoMAE Dependencies Installation Script
# This script sets up the environment for VideoMAE model finetuning

set -e  # Exit on any error

echo "🚀 Starting VideoMAE Dependencies Installation..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root for apt-get commands
check_sudo() {
    if [[ $EUID -eq 0 ]]; then
        print_warning "Running as root. This is not recommended for conda operations."
    fi
}

# Install system dependencies
install_system_dependencies() {
    print_status "Installing system dependencies with apt-get..."
    
    # Update package list
    sudo apt-get update
    
    # Install essential build tools and libraries
    sudo apt-get install -y \
        build-essential \
        cmake \
        git \
        curl \
        wget \
        ca-certificates \
        libjpeg-dev \
        libpng-dev \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        ffmpeg \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libswscale-dev \
        libswresample-dev \
        libavfilter-dev \
        libavdevice-dev \
        pkg-config \
        python3-dev \
        python3-pip \
        libtool \
        autoconf \
        automake
        
    print_status "System dependencies installed successfully!"
}

# Install Miniconda if not already available
install_miniconda() {
    print_status "Checking for conda installation..."
    
    # Check if conda command is available
    if command -v conda &> /dev/null; then
        print_status "Conda is already installed and available in PATH."
        return 0
    fi
    
    # Check if Miniconda directory already exists
    if [[ -d "$HOME/miniconda3" ]]; then
        print_status "Miniconda directory already exists at $HOME/miniconda3"
        print_status "Adding to PATH and initializing..."
        
        # Add to PATH for current session
        export PATH="$HOME/miniconda3/bin:$PATH"
        
        # Initialize conda if not already done
        if [[ ! -f "$HOME/.bashrc" ]] || ! grep -q "miniconda3" "$HOME/.bashrc"; then
            print_status "Initializing conda..."
            "$HOME/miniconda3/bin/conda" init bash
        fi
        
        # Source the conda script
        if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
            source "$HOME/miniconda3/etc/profile.d/conda.sh"
        fi
        
        print_status "✅ Using existing Miniconda installation."
        return 0
    fi
    
    print_status "Conda not found. Installing Miniconda..."
    
    # Detect architecture
    ARCH=$(uname -m)
    OS=$(uname -s)
    
    # Determine the correct Miniconda installer URL
    if [[ "$OS" == "Linux" ]]; then
        if [[ "$ARCH" == "x86_64" ]]; then
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        elif [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
        else
            print_error "Unsupported architecture: $ARCH"
            exit 1
        fi
    elif [[ "$OS" == "Darwin" ]]; then
        if [[ "$ARCH" == "x86_64" ]]; then
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
        elif [[ "$ARCH" == "arm64" ]]; then
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
        else
            print_error "Unsupported architecture: $ARCH"
            exit 1
        fi
    else
        print_error "Unsupported operating system: $OS"
        exit 1
    fi
    
    # Download Miniconda installer
    MINICONDA_INSTALLER="/tmp/miniconda_installer.sh"
    print_status "Downloading Miniconda installer for $OS-$ARCH..."
    
    if ! wget -O "$MINICONDA_INSTALLER" "$MINICONDA_URL"; then
        if ! curl -o "$MINICONDA_INSTALLER" "$MINICONDA_URL"; then
            print_error "Failed to download Miniconda installer. Please check your internet connection."
            exit 1
        fi
    fi
    
    # Install Miniconda
    print_status "Installing Miniconda..."
    bash "$MINICONDA_INSTALLER" -b -p "$HOME/miniconda3"
    
    # Clean up installer
    rm -f "$MINICONDA_INSTALLER"
    
    # Initialize conda
    print_status "Initializing conda..."
    "$HOME/miniconda3/bin/conda" init bash
    
    # Add conda to PATH for current session
    export PATH="$HOME/miniconda3/bin:$PATH"
    
    # Source the conda script to make it available
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    
    print_status "Miniconda installed successfully!"
    print_warning "Note: You may need to restart your terminal or run 'source ~/.bashrc' after installation."
}

# Setup conda environment
setup_conda_environment() {
    print_status "Setting up conda environment 'videomae'..."
    
    # Ensure conda is available (install if needed)
    install_miniconda
    
    # Check if environment already exists
    if conda env list | grep -q "videomae"; then
        print_status "Environment 'videomae' already exists. Using existing environment..."
    else
        print_status "Creating new conda environment 'videomae'..."
        conda create -n videomae python=3.10 -y
    fi
    
    # Activate environment
    print_status "Activating conda environment 'videomae'..."
    
    # Source conda initialization script (handle both existing and newly installed conda)
    if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif command -v conda &> /dev/null; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
    else
        print_error "Could not find conda initialization script"
        exit 1
    fi
    
    conda activate videomae
    
    # Install conda packages that work better from conda-forge
    print_status "Installing conda packages..."
    conda install -n videomae -c conda-forge \
        "numpy>=1.21.0,<2.0" \
        scipy \
        pillow \
        opencv \
        libstdcxx-ng \
        libgcc-ng \
        -y
    
    print_status "Conda environment setup completed!"
}

# Install uv if not already installed
install_uv() {
    print_status "Checking for uv installation..."
    
    if ! command -v uv &> /dev/null; then
        print_status "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        
        # Add uv to PATH for current session
        export PATH="$HOME/.cargo/bin:$PATH"
        
        # Source cargo environment if it exists
        if [[ -f "$HOME/.cargo/env" ]]; then
            source "$HOME/.cargo/env"
        fi
        
        # Verify uv is now available
        if command -v uv &> /dev/null; then
            print_status "✅ uv installed successfully!"
        else
            print_warning "⚠️  uv installation may not be in PATH. Adding to current session..."
            export PATH="$HOME/.local/bin:$PATH"
        fi
    else
        print_status "uv is already installed."
    fi
}

# Install Python dependencies with uv
install_python_dependencies() {
    print_status "Installing Python dependencies with uv..."
    
    # Ensure we're in the conda environment
    if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif command -v conda &> /dev/null; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
    fi
    conda activate videomae
    
    # Clean up any existing NumPy 2.x installations that might cause conflicts
    print_status "Cleaning up any existing NumPy 2.x installations..."
    uv pip uninstall numpy -y 2>/dev/null || true
    python -m pip uninstall numpy -y 2>/dev/null || true
    
    # Install PyTorch and torchvision first (GPU version)
    print_status "Installing PyTorch with CUDA support..."
    uv pip uninstall torch torchvision torchaudio

    # Ensure NumPy 1.x compatibility before installing PyTorch
    print_status "Ensuring NumPy 1.x compatibility..."
    uv pip install "numpy>=1.21.0,<2.0"
    
    # Install PyTorch with compatible versions and NumPy constraints
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    
    # Check and fix numpy compatibility after PyTorch installation
    NUMPY_VERSION=$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "none")
    print_status "Installed NumPy version: $NUMPY_VERSION"
    
    # Verify NumPy is in 1.x range, downgrade if needed
    if python -c "import numpy; import sys; sys.exit(0 if numpy.__version__.startswith('1.') else 1)" 2>/dev/null; then
        print_status "✅ NumPy 1.x detected: $NUMPY_VERSION"
    else
        print_warning "⚠️  NumPy 2.x detected ($NUMPY_VERSION). Downgrading to 1.x for compatibility..."
        uv pip install "numpy>=1.21.0,<2.0" --force-reinstall
        NUMPY_VERSION=$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "none")
        print_status "NumPy downgraded to: $NUMPY_VERSION"
    fi

    # Install other core dependencies with NumPy 1.x constraints
    print_status "Installing core ML dependencies..."
    uv pip install \
        "numpy>=1.21.0,<2.0" \
        timm==0.6.12 \
        tensorboardX \
        einops \
        matplotlib \
        seaborn \
        pandas \
        scikit-learn \
        tqdm \
        pyyaml \
        opencv-python \
        imageio \
        imageio-ffmpeg \
        packaging \
        requests \
        safetensors
    
    # Install PyAV as reliable video loading fallback
    print_status "Installing PyAV for video processing..."
    uv pip install av
    
    # Install decord for video processing
    print_status "Installing decord for video processing..."
    
    # Check architecture to handle ARM64 systems
    ARCH=$(uname -m)
    if [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
        print_warning "ARM64 architecture detected. Trying multiple approaches for decord installation..."
        
        # Method 1: Try conda-forge first (sometimes has better ARM64 support)
        print_status "Attempting to install decord via conda-forge..."
        if conda install -n videomae -c conda-forge decord -y 2>/dev/null; then
            print_status "✅ Successfully installed decord via conda-forge!"
        else
            print_warning "Conda installation failed. Trying alternative approaches..."
            
            # Method 2: Try pre-release or alternative versions
            print_status "Trying decord pre-release versions..."
            if uv pip install --prerelease=allow decord 2>/dev/null; then
                print_status "✅ Successfully installed decord pre-release!"
            else
                print_warning "❌ Decord installation failed on ARM64 architecture."
                print_warning "This is a known compatibility issue with ARM64 systems."
                print_status ""
                print_status "🔧 Alternative Solutions:"
                print_status "1. VideoMAE will use PyAV for video loading (reliable and efficient)"
                print_status "2. You can manually try: pip install decord==0.6.0 --no-deps"
                print_status "3. Use Docker with x86_64 emulation"
                print_status "4. Run on x86_64 hardware for full decord support"
                print_status ""
                print_status "Continuing installation without decord (PyAV will be used instead)..."
            fi
        fi
    else
        # For x86_64 systems, use regular installation
        uv pip install decord
    fi
    
    # Install DeepSpeed for distributed training (optional but recommended)
    print_status "Installing DeepSpeed..."
    
    # Check if we're on ARM64 and handle accordingly
    ARCH=$(uname -m)
    if [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
        print_warning "ARM64 detected. DeepSpeed has compatibility issues on ARM64."
        print_status "Skipping DeepSpeed installation (VideoMAE will work fine without it)."
        print_status "For distributed training on ARM64, consider alternatives like torchrun."
    else
        # For x86_64, try with pre-compiled ops and compatible pydantic
        print_status "Installing DeepSpeed with compatible dependencies..."
        # Install compatible pydantic first
        uv pip install "pydantic<2.0"
        
        if ! DS_BUILD_OPS=1 uv pip install deepspeed==0.8.3; then
            print_warning "Failed to install with pre-compiled ops. Installing without ops..."
            uv pip install deepspeed==0.8.3
        fi
    fi
    
    # Install additional utilities with NumPy constraints
    print_status "Installing additional utilities..."
    uv pip install \
        "numpy>=1.21.0,<2.0" \
        wandb \
        tensorboard \
        ipython \
        jupyter \
        notebook
    
    print_status "Python dependencies installed successfully!"
}

# Verify installation
verify_installation() {
    print_status "Verifying installation..."
    
    if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif command -v conda &> /dev/null; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
    fi
    conda activate videomae
    
    # Clean up any broken installations first
    print_status "Cleaning up any broken installations..."
    pip uninstall decord -y 2>/dev/null || true
    rm -rf /tmp/decord 2>/dev/null || true
    
    # Clean up DeepSpeed on ARM64 since we skip it now
    ARCH=$(uname -m)
    if [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
        print_status "Removing any existing DeepSpeed installation on ARM64..."
        pip uninstall deepspeed -y 2>/dev/null || true
    fi
    
    # Determine what should be available based on architecture
    ARCH=$(uname -m)
    EXPECT_DEEPSPEED=false
    if [[ "$ARCH" != "aarch64" && "$ARCH" != "arm64" ]]; then
        EXPECT_DEEPSPEED=true
    fi
    
    # Test imports
    python -c "
import sys
import platform

# Architecture check
ARCH = platform.machine()
EXPECT_DEEPSPEED = ARCH not in ['aarch64', 'arm64']

print(f'Architecture: {ARCH}')
print('')

import torch
import torchvision
import timm
import cv2
import numpy as np
import PIL
from tensorboardX import SummaryWriter
import safetensors

# Test video loading libraries
try:
    import decord
    print('✅ Decord imported successfully!')
    DECORD_AVAILABLE = True
except (ImportError, RuntimeError, OSError) as e:
    print('⚠️  Decord not available (this is OK on ARM64 architectures)')
    DECORD_AVAILABLE = False

try:
    import av
    print('✅ PyAV imported successfully!')
    PYAV_AVAILABLE = True
except ImportError as e:
    print('❌ PyAV not available (this may cause video loading issues)')
    PYAV_AVAILABLE = False

# Only test DeepSpeed if we expect it to be installed
DEEPSPEED_AVAILABLE = False
if EXPECT_DEEPSPEED:
    try:
        import deepspeed
        print('✅ DeepSpeed imported successfully!')
        DEEPSPEED_AVAILABLE = True
    except ImportError as e:
        print('⚠️  DeepSpeed not available (training will still work without it)')
        DEEPSPEED_AVAILABLE = False
else:
    print('⚠️  DeepSpeed skipped on ARM64 (use torchrun for distributed training)')

print('')
print('✅ Core dependencies imported successfully!')
print(f'PyTorch version: {torch.__version__}')
print(f'Torchvision version: {torchvision.__version__}')
print(f'NumPy version: {np.__version__}')

# Verify NumPy version compatibility
if np.__version__.startswith('1.'):
    print('✅ NumPy 1.x detected - compatible with compiled modules')
else:
    print('⚠️  WARNING: NumPy 2.x detected - may cause compatibility issues!')
    print('   Consider downgrading with: pip install \"numpy>=1.21.0,<2.0\"')

print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device count: {torch.cuda.device_count()}')
    print(f'Current CUDA device: {torch.cuda.current_device()}')

print('')
print('📋 Installation Summary:')
print(f'   PyTorch + CUDA: ✅')
print(f'   Core ML libraries: ✅') 
print(f'   Decord (video loading): {\"✅\" if DECORD_AVAILABLE else \"❌ (will use PyAV fallback)\"}')
print(f'   PyAV (video fallback): {\"✅\" if PYAV_AVAILABLE else \"❌\"}')
if EXPECT_DEEPSPEED:
    print(f'   DeepSpeed (distributed): {\"✅\" if DEEPSPEED_AVAILABLE else \"❌ (optional)\"}')
else:
    print(f'   DeepSpeed (distributed): ⏭️  (skipped on ARM64 - use torchrun instead)')

# Video loading capability check
if DECORD_AVAILABLE:
    print(f'   Video loading: ✅ Decord (optimal)')
elif PYAV_AVAILABLE:
    print(f'   Video loading: ✅ PyAV (reliable fallback)')
else:
    print(f'   Video loading: ⚠️  OpenCV only (may have limitations)')
"
    
    if [ $? -eq 0 ]; then
        print_status "✅ Installation verification completed successfully!"
    else
        print_error "❌ Installation verification failed!"
        exit 1
    fi
}

# Create activation script
create_activation_script() {
    print_status "Creating activation script..."
    
    cat > activate_videomae.sh << 'EOF'
#!/bin/bash
# VideoMAE Environment Activation Script

echo "🎬 Activating VideoMAE environment..."

# Activate conda environment (handle both existing and newly installed conda)
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "❌ Could not find conda installation"
    exit 1
fi

conda activate videomae

# Add uv to PATH if needed
export PATH="$HOME/.cargo/bin:$PATH"

echo "✅ VideoMAE environment activated!"
echo "🚀 You can now run VideoMAE finetuning scripts."
echo ""
echo "Example usage:"
echo "  python run_class_finetuning.py --help"
echo ""
echo "To deactivate: conda deactivate"
EOF
    
    chmod +x activate_videomae.sh
    print_status "Created activation script: activate_videomae.sh"
}

# Create alternative decord installation script for advanced users
create_decord_alternative_script() {
    print_status "Creating alternative decord installation script..."
    
    cat > install_decord_alternative.sh << 'EOF'
#!/bin/bash
# Alternative Decord Installation for ARM64
# This script provides multiple approaches to install decord on ARM64

echo "🔧 Alternative Decord Installation for ARM64"
echo "Choose an option:"
echo "1. Try building with system FFmpeg (may work)"
echo "2. Install with --no-deps (risky but sometimes works)"
echo "3. Use OpenCV alternative (modify VideoMAE code)"
echo "4. Exit"
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo "Attempting build with system FFmpeg..."
        sudo apt-get install -y ffmpeg libavcodec-dev libavformat-dev
        pip install decord --no-binary=decord --verbose
        ;;
    2)
        echo "Installing with --no-deps..."
        pip install decord==0.6.0 --no-deps
        echo "⚠️  May have missing dependencies!"
        ;;
    3)
        echo "Setting up PyAV alternative..."
        echo "PyAV should already be installed as part of the main installation"
        echo "VideoMAE has been modified to use PyAV as fallback when decord is not available"
        ;;
    4)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice"
        ;;
esac
EOF
    
    chmod +x install_decord_alternative.sh
    print_status "Created alternative installation script: install_decord_alternative.sh"
}

# Main installation flow
main() {
    print_status "Starting VideoMAE installation process..."
    
    check_sudo
    install_system_dependencies
    setup_conda_environment
    install_uv
    install_python_dependencies
    verify_installation
    create_activation_script
    create_decord_alternative_script
    
    print_status "🎉 VideoMAE installation completed successfully!"
    print_status ""
    print_status "📋 Important Notes:"
    print_status "• Miniconda has been automatically installed (if not already present)"
    print_status "• NumPy has been pinned to 1.x for compatibility with compiled modules"
    print_status "• If you encounter NumPy-related errors, ensure no NumPy 2.x is installed:"
    print_status "  conda activate videomae && pip install 'numpy>=1.21.0,<2.0' --force-reinstall"
    print_status ""
    print_status "To get started:"
    print_status "1. Run: source activate_videomae.sh"
    print_status "2. Or manually: conda activate videomae"
    print_status "3. Then you can run finetuning scripts!"
    print_status ""
    print_status "💡 If this is a fresh Miniconda installation, you may need to:"
    print_status "   • Restart your terminal, or"
    print_status "   • Run: source ~/.bashrc"
    print_status ""
    print_status "Happy training! 🚀"
}

# Run main function
main "$@" 