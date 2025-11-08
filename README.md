# Neuro-Controlled Robotic Limb and Exoskeleton for Rehabilitation and Strength Enhancement

**Team Name:** Astra  
**Team Leader:** Supritika Bhardwaj  
**Team Members:** Anushka Yadav, Shreya Bhardwaj  
**Institution:** Graphic Era Hill University, Dehradun, India

## Problem Statement

Many people lose their ability to move properly because of accidents, paralysis, or limb loss. The existing prosthetic limbs and walking aids are basic and cannot move naturally or adjust to the user's needs. Also, people doing heavy or repetitive work often get tired or injured.

### Key Challenges:
- Traditional prosthetics are mechanical and cannot adapt to natural human movements
- Existing devices lack sensory feedback, limiting grip control and comfort
- People with paralysis rely on wheelchairs, reducing independence and rehabilitation potential
- Industrial workers face fatigue and injury from repetitive heavy tasks
- Recent AI and EMG-based prosthetic research shows improved intent detection and motion control

## Proposed Solution

An AI-powered Smart Prosthetic and Exoskeleton System designed to restore mobility, enhance strength, and provide natural movement for people with disabilities or mobility challenges.

### Core Features:
1. **Intelligent Control System**: AI algorithms interpret EMG and motion sensor data, accurately predicting user intent and adapting in real time
2. **Sensory Feedback Integration**: Haptic and pressure sensors provide a sense of touch, allowing users to feel grip force or ground contact
3. **Lightweight and Custom Design**: 3D printing and advanced materials like carbon fiber ensure perfect, comfortable fit
4. **Versatile Applications**: Mobility restoration, strength augmentation, and rehabilitation training
5. **Scalable and Adaptive Technology**: Modular design allows easy upgrades and customization

## Technologies Used

### Hardware:
- **Sensors**: EMG, IMU, force/pressure sensors, rotary encoders
- **Actuators**: Compact electric motors, servo motors
- **Processing**: Raspberry Pi, Jetson, Arduino, STM32
- **Materials**: 3D-printed frames, carbon fiber, lightweight metals
- **Communication**: I²C, SPI, CAN interfaces
- **Feedback**: Vibrotactile haptic modules

### Software:
- **Programming Languages**: Python, C/C++
- **Frameworks**: ROS2, RTOS, TensorFlow/PyTorch
- **Development Tools**: Arduino IDE, MATLAB/Simulink
- **AI/ML**: SVM, CNN, LSTM models for intent prediction
- **Cloud Services**: Data logging, model training, remote monitoring

## Project Structure

```
prosthetic-exoskeleton/
├── README.md
├── requirements.txt
├── config/
│   └── settings.yaml
├── hardware/
│   ├── sensors/
│   ├── actuators/
│   └── communication/
├── software/
│   ├── ai_models/
│   ├── control_system/
│   ├── sensor_processing/
│   └── haptic_feedback/
├── docs/
│   ├── research/
│   └── design/
└── tests/
```

## Impact and Benefits

### Social Impact:
- Restores independence for amputees and paralysis patients
- Supports rehabilitation and reduces workplace injuries
- Improves confidence, mobility, and overall quality of life

### Economic Impact:
- Reduces healthcare costs by accelerating rehabilitation
- Boosts productivity by helping workers perform heavy tasks safely
- Creates opportunities in assistive robotics and healthcare technology market

### Environmental Impact:
- Lightweight, durable materials and 3D printing reduce material waste
- Energy-efficient actuators minimize power consumption
- Long-lasting components reduce environmental footprint

### Target Audience:
- Amputees and paralysis patients for mobility restoration
- Rehabilitation centers for therapy
- Industrial or military workers for strength and endurance support
- Researchers in assistive robotics and human-machine interaction

## Research and References

1. **A Review of EMG-Based Control of Prosthetic Hands** – IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2021
   - Link: https://ieeexplore.ieee.org/document/9356789

2. **Exoskeletons for Rehabilitation and Industrial Applications: A Review** – Frontiers in Robotics and AI, 2023
   - Link: https://www.mdpi.com/2076-3417/12/5/2345

## Getting Started

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
```bash
pip install -r requirements.txt
```

### Round 0 MVP - Quick Start

The Round 0 MVP demonstrates the core concept with simulated data:

```bash
python round0_mvp.py
```

This will show:
- Live EMG signal graph
- Real-time intent prediction (rest, flex, strong flex)
- Elbow angle visualization
- Virtual robotic limb animation

See `QUICKSTART.md` for detailed instructions.

### Usage
See individual module documentation in the `docs/` directory.

## License
[Specify your license here]

## Contact
Team Astra - Graphic Era Hill University, Dehradun, India

