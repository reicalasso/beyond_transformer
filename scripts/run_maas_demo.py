#!/usr/bin/env python
"""
Quick start script for PULSE + MaaS demo

Runs a complete demonstration of the Memory-as-a-Service system.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from examples.maas_integration_demo import simulate_chatbot_session

if __name__ == "__main__":
    print("Starting PULSE + MaaS Integration Demo...\n")
    simulate_chatbot_session()
