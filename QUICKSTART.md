# Round 0 MVP - Quick Start Guide

## Overview
This is the Minimum Viable Prototype (MVP) for the Neuro-Controlled Robotic Limb and Exoskeleton project. It demonstrates the core concept using simulated data.

## Features Demonstrated
1. ✅ Simulated EMG signal generator with noise and activation spikes
2. ✅ Simulated IMU orientation generator
3. ✅ Basic AI Model (SVM) for intent classification (rest, flex, strong flex)
4. ✅ Simple robotic limb simulation (single-joint elbow animation)
5. ✅ Real-time graph plotting with Matplotlib

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the MVP
```bash
python round0_mvp.py
```

## What You'll See

When you run the program, a window will open with 4 real-time plots:

1. **Live EMG Graph** (Top Left)
   - Shows simulated EMG signal with muscle activation spikes
   - Updates in real-time

2. **Live Intent Prediction** (Top Right)
   - Shows AI model's prediction: Rest, Flex, or Strong Flex
   - Displays confidence level

3. **Elbow Angle Over Time** (Bottom Left)
   - Shows how the elbow angle changes based on predictions
   - 180° = straight arm, 0° = fully flexed

4. **Virtual Limb Animation** (Bottom Right)
   - Visual representation of the robotic limb
   - Elbow angle updates based on AI predictions
   - Shows shoulder, elbow, and wrist joints

## How It Works

1. **EMG Simulation**: Generates realistic EMG signals with:
   - Multiple frequency components (50-450 Hz)
   - Gaussian noise
   - Occasional activation spikes

2. **Intent Classification**: 
   - SVM model extracts features from EMG signals
   - Classifies into 3 states: rest, flex, strong_flex
   - Model trains automatically on first run

3. **Limb Control**:
   - Elbow angle changes based on predicted intent
   - Rest → 180° (straight)
   - Flex → 90° (half flexed)
   - Strong Flex → 45° (fully flexed)

4. **IMU Simulation**:
   - Generates orientation data based on elbow angle
   - Simulates roll, pitch, yaw angles

## Program Flow

```
EMG Simulator → Feature Extraction → SVM Classifier → Intent Prediction
                                                           ↓
IMU Simulator ← Limb Simulator ← Angle Update ← Intent Mapping
```

## Stopping the Program

Simply close the matplotlib window to stop the program.

## Troubleshooting

### Import Errors
If you get import errors, make sure you're running from the project root directory:
```bash
cd "prosthetic Exoskeleton"
python round0_mvp.py
```

### Missing Dependencies
If you get module not found errors:
```bash
pip install --upgrade -r requirements.txt
```

### Display Issues
If the plots don't update:
- Make sure you have a display/GUI environment
- On Linux, you may need: `export DISPLAY=:0`

## Next Steps (Future Rounds)

- Round 1: Real sensor integration
- Round 2: Advanced AI models (CNN, LSTM)
- Round 3: Multi-joint control
- Round 4: Haptic feedback integration
- Round 5: Full exoskeleton system

## Team Information

**Team Name:** Astra  
**Team Leader:** Supritika Bhardwaj  
**Team Members:** Anushka Yadav, Shreya Bhardwaj  
**Institution:** Graphic Era Hill University, Dehradun, India

