"""
Main GUI Application for 6-DOF Robotic Arm Control
====================================================
Entry point: python main_gui.py

Wires together:
  ThreadedRobotSimulation  → PyBullet physics backend
  NNController             → NN warm-start + DLS IK refinement
  RobotControlGUI          → PyQt6 industry-grade dark UI
"""
import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont

from simulation_threaded import ThreadedRobotSimulation
from gui_control_nn      import NNController
from ui                  import RobotControlGUI, DARK_STYLESHEET


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_STYLESHEET)

    # Use a clean monospace default font
    font = QFont("Consolas", 10)
    app.setFont(font)

    # Start simulation backend
    sim = ThreadedRobotSimulation()
    sim.start()

    # Initialise IK controller
    controller = NNController(sim)

    # Launch GUI
    window = RobotControlGUI(sim=sim, controller=controller)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
