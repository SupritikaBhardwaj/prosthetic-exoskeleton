"""
ROUND 0 - Minimum Viable Prototype (MVP)
Neuro-Controlled Robotic Limb and Exoskeleton

This program demonstrates:
1. Simulated EMG signal generation
2. Simulated IMU orientation generation
3. Basic AI Model (SVM) for intent classification
4. Simple robotic limb simulation (single-joint elbow)
5. Real-time graph plotting with Matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import sys
import os

# Add project paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from software.sensor_processing.emg_simulator import EMGSimulator
from software.sensor_processing.imu_simulator import IMUSimulator
from software.ai_models.intent_classifier import IntentClassifier
from software.control_system.limb_simulator import LimbSimulator


class MVP:
    """
    Main MVP class that integrates all components
    """
    
    def __init__(self):
        """Initialize MVP components"""
        # Initialize simulators
        self.emg_sim = EMGSimulator(sampling_rate=1000, channels=1)
        self.imu_sim = IMUSimulator()
        
        # Initialize AI classifier
        self.classifier = IntentClassifier()
        
        # Initialize limb simulator
        self.limb_sim = LimbSimulator(initial_angle=180)
        
        # Data buffers for plotting
        self.emg_buffer = []
        self.time_buffer = []
        self.prediction_buffer = []
        self.angle_buffer = []
        self.buffer_size = 500  # Keep last 500 samples
        
        # Current state
        self.current_intent = 'rest'
        self.current_confidence = 0.0
        self.current_time = 0.0
        
        # Signal window for classification
        self.signal_window = []
        self.window_size = 100  # 100 samples for classification
        
        # Training flag
        self.model_trained = False
        
    def train_model(self):
        """Train the SVM classifier"""
        print("\n" + "="*50)
        print("Training SVM Classifier...")
        print("="*50)
        
        # Generate training data
        X_train, y_train = self.classifier.generate_training_data(
            self.emg_sim, 
            samples_per_class=200,
            window_size=self.window_size
        )
        
        # Train model
        self.classifier.train(X_train, y_train)
        self.model_trained = True
        
        print("\nModel training completed!")
        print("="*50 + "\n")
    
    def update(self, frame):
        """
        Update function called by animation
        
        Args:
            frame: Animation frame number
        """
        # Generate new EMG sample
        # Simulate changing intent over time for demonstration
        t = self.current_time
        if int(t) % 5 == 0 and int(t) % 10 < 3:
            intent_type = 'strong_flex'
        elif int(t) % 5 == 0:
            intent_type = 'flex'
        else:
            intent_type = 'rest'
        
        emg_sample = self.emg_sim.generate_sample(intent_type)
        
        # Update time
        dt = 1.0 / 100  # 100 Hz update rate for display
        self.current_time += dt
        
        # Add to signal window
        self.signal_window.append(emg_sample)
        if len(self.signal_window) > self.window_size:
            self.signal_window.pop(0)
        
        # Predict intent if we have enough samples and model is trained
        if len(self.signal_window) == self.window_size and self.model_trained:
            prediction, confidence = self.classifier.predict(np.array(self.signal_window))
            self.current_intent = prediction
            self.current_confidence = confidence
            
            # Update limb angle based on prediction
            self.limb_sim.update_angle(prediction, dt)
        
        # Get IMU orientation
        elbow_angle = self.limb_sim.current_angle
        imu_data = self.imu_sim.generate_orientation(elbow_angle)
        
        # Update buffers
        self.emg_buffer.append(emg_sample)
        self.time_buffer.append(self.current_time)
        self.prediction_buffer.append(self.current_intent)
        self.angle_buffer.append(elbow_angle)
        
        # Keep buffer size limited
        if len(self.emg_buffer) > self.buffer_size:
            self.emg_buffer.pop(0)
            self.time_buffer.pop(0)
            self.prediction_buffer.pop(0)
            self.angle_buffer.pop(0)
        
        # Update plots
        self.update_plots()
    
    def update_plots(self):
        """Update all matplotlib plots"""
        # Clear all axes
        for ax in self.axes:
            ax.clear()
        
        # Plot 1: EMG Signal
        if len(self.time_buffer) > 0:
            self.axes[0].plot(self.time_buffer, self.emg_buffer, 'b-', linewidth=1.5)
            self.axes[0].set_xlabel('Time (s)')
            self.axes[0].set_ylabel('EMG Signal (V)')
            self.axes[0].set_title('Live EMG Signal', fontweight='bold')
            self.axes[0].grid(True, alpha=0.3)
            self.axes[0].set_xlim(max(0, self.current_time - 5), self.current_time + 0.5)
        
        # Plot 2: Intent Prediction
        if len(self.time_buffer) > 0:
            # Convert predictions to numeric for plotting
            pred_numeric = []
            for pred in self.prediction_buffer:
                if pred == 'rest':
                    pred_numeric.append(0)
                elif pred == 'flex':
                    pred_numeric.append(1)
                elif pred == 'strong_flex':
                    pred_numeric.append(2)
                else:
                    pred_numeric.append(0)
            
            self.axes[1].plot(self.time_buffer, pred_numeric, 'g-', linewidth=2, label='Prediction')
            self.axes[1].fill_between(self.time_buffer, 0, pred_numeric, alpha=0.3)
            self.axes[1].set_xlabel('Time (s)')
            self.axes[1].set_ylabel('Intent')
            self.axes[1].set_title('Live Intent Prediction', fontweight='bold')
            self.axes[1].set_yticks([0, 1, 2])
            self.axes[1].set_yticklabels(['Rest', 'Flex', 'Strong Flex'])
            self.axes[1].grid(True, alpha=0.3)
            self.axes[1].set_xlim(max(0, self.current_time - 5), self.current_time + 0.5)
            self.axes[1].set_ylim(-0.2, 2.5)
            
            # Add current prediction text
            pred_text = f'Current: {self.current_intent} (Confidence: {self.current_confidence:.2f})'
            self.axes[1].text(0.02, 0.95, pred_text, transform=self.axes[1].transAxes,
                            fontsize=10, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Plot 3: Elbow Angle
        if len(self.time_buffer) > 0:
            self.axes[2].plot(self.time_buffer, self.angle_buffer, 'r-', linewidth=2)
            self.axes[2].fill_between(self.time_buffer, 0, self.angle_buffer, alpha=0.3, color='red')
            self.axes[2].set_xlabel('Time (s)')
            self.axes[2].set_ylabel('Elbow Angle (degrees)')
            self.axes[2].set_title('Elbow Angle Over Time', fontweight='bold')
            self.axes[2].grid(True, alpha=0.3)
            self.axes[2].set_ylim(0, 190)
            self.axes[2].set_xlim(max(0, self.current_time - 5), self.current_time + 0.5)
        
        # Plot 4: Robotic Limb Visualization
        self.limb_sim.draw_limb(self.axes[3])
        
        # Adjust layout
        plt.tight_layout()
    
    def run(self):
        """Run the MVP demonstration"""
        print("\n" + "="*60)
        print("ROUND 0 - Minimum Viable Prototype (MVP)")
        print("Neuro-Controlled Robotic Limb and Exoskeleton")
        print("="*60)
        print("\nTeam: Astra")
        print("Graphic Era Hill University, Dehradun, India\n")
        
        # Train model first
        if not self.model_trained:
            self.train_model()
        
        # Create figure and axes
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.suptitle('Neuro-Controlled Robotic Limb - Real-Time Monitoring', 
                         fontsize=16, fontweight='bold')
        
        # Start animation
        print("Starting real-time visualization...")
        print("Close the window to stop.\n")
        
        self.ani = FuncAnimation(self.fig, self.update, interval=10, blit=False)
        plt.show()


def main():
    """Main entry point"""
    mvp = MVP()
    mvp.run()


if __name__ == "__main__":
    main()

