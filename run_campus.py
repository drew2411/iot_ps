import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from campus.main_campus import main

if __name__ == "__main__":
    print("Starting Fixed Campus Energy Ensemble Analysis...")
    print(f"Project root: {project_root}")
    main()
